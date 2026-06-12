"""
Data storage: SQLite for metadata/indexes, Parquet for time-series data.
"""
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

from quanti.config import settings
from quanti.data.schema import ETFDailyBar


class DataStorage:
    """
    Three-layer storage:
    - raw/    : Immutable, as-received Parquet files
    - clean/  : Validated, normalized Parquet files
    - SQLite  : Metadata, ingestion log, schema version
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.DATA_DIR)
        self.raw_dir = self.base_dir / "raw"
        self.clean_dir = self.base_dir / "clean"
        self.features_dir = self.base_dir / "features"
        self.db_path = Path(settings.DB_PATH)

        # Ensure directories exist
        for d in [self.raw_dir, self.clean_dir, self.features_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize SQLite metadata database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,       -- 'tushare' | 'akshare'
                    symbol TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    bar_count INTEGER,
                    status TEXT,                -- 'success' | 'error'
                    error_message TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_freshness (
                    symbol TEXT PRIMARY KEY,
                    last_date TEXT NOT NULL,
                    last_ingested_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                INSERT OR IGNORE INTO schema_version (version, applied_at)
                VALUES (1, ?)
            """, (datetime.now().isoformat(),))

    def log_ingestion(
        self,
        source: str,
        symbol: str,
        start_date: str,
        end_date: str,
        bar_count: int,
        status: str,
        error_message: str = "",
    ) -> None:
        """Record ingestion run in the log."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT INTO ingestion_log
                   (timestamp, source, symbol, start_date, end_date, bar_count, status, error_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), source, symbol, start_date, end_date,
                 bar_count, status, error_message),
            )

    def update_freshness(self, symbol: str, last_date: str) -> None:
        """Update the last-known data date for a symbol."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO data_freshness (symbol, last_date, last_ingested_at)
                   VALUES (?, ?, ?)""",
                (symbol, last_date, datetime.now().isoformat()),
            )

    def check_freshness(self, symbol: str) -> tuple[str | None, float]:
        """Return (last_date, hours_since_ingest) for a symbol."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT last_date, last_ingested_at FROM data_freshness WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None:
            return (None, float("inf"))
        last_dt = datetime.fromisoformat(row[1])
        hours = (datetime.now() - last_dt).total_seconds() / 3600
        return (row[0], hours)

    def save_bars_raw(self, symbol: str, bars: list[ETFDailyBar]) -> str:
        """Save raw bars to Parquet. Returns file path."""
        df = self._bars_to_df(bars)
        path = self.raw_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(path, index=False)
        return str(path)

    def save_bars_clean(self, symbol: str, bars: list[ETFDailyBar]) -> str:
        """Save validated/cleaned bars to Parquet. Returns file path.

        Strips .SH/.SZ suffix so all data is persisted under the bare ETF code
        (e.g. 510300, 159915). This gives load_bars() a single canonical path.
        """
        df = self._bars_to_df(bars)
        base = symbol.replace(".SH", "").replace(".SZ", "").replace(".sh", "").replace(".sz", "")
        path = self.clean_dir / f"{base}.parquet"
        # Append if exists, deduplicate by trade_date
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["trade_date"], keep="last")
            df = df.sort_values("trade_date")
        df.to_parquet(path, index=False)
        return str(path)

    def load_bars(self, symbol: str) -> list[ETFDailyBar]:
        """Load cleaned bars for a symbol from Parquet.

        Tries multiple suffix patterns to accommodate inconsistent naming
        from different data sources (bare codes, .SH, .SZ):
          1. {symbol}.parquet       (bare code, e.g. 510300.parquet)
          2. {symbol}.SH.parquet   (Shanghai suffix, e.g. 510300.SH.parquet)
          3. {symbol}.SZ.parquet   (Shenzhen suffix, e.g. 159915.SZ.parquet)
        """
        # Strip any caller-provided suffix so we probe with our own
        base = symbol.replace(".SH", "").replace(".SZ", "").replace(".sh", "").replace(".sz", "")
        for sfx in ["", ".SH", ".SZ"]:
            path = self.clean_dir / f"{base}{sfx}.parquet"
            if path.exists():
                break
        else:
            return []
        df = pd.read_parquet(path)
        bars = []
        for _, row in df.iterrows():
            bars.append(ETFDailyBar(
                symbol=row["symbol"],
                trade_date=row["trade_date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                amount=float(row.get("amount", 0)),
            ))
        return sorted(bars, key=lambda b: b.trade_date)

    @staticmethod
    def _bars_to_df(bars: list[ETFDailyBar]) -> pd.DataFrame:
        """Convert ETFDailyBar list to DataFrame."""
        records = []
        for b in bars:
            records.append({
                "symbol": b.symbol,
                "trade_date": b.trade_date,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "amount": b.amount,
            })
        return pd.DataFrame(records)

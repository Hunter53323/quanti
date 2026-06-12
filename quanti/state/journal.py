"""
Durable position and order journal (SQLite).
Every position change and order lifecycle event is written here.
The journal is the source of truth for crash recovery.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from quanti.config import settings


class Journal:
    """SQLite-based position and order journal."""

    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or settings.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    # ---- Table Creation ----

    def _init_tables(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS position_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    order_id TEXT,
                    status TEXT NOT NULL,
                    reason TEXT
                );

                CREATE TABLE IF NOT EXISTS order_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    order_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filled_qty INTEGER DEFAULT 0,
                    avg_fill_price REAL,
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    snapshot TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_position_journal_ts
                    ON position_journal(timestamp);
                CREATE INDEX IF NOT EXISTS idx_order_journal_ts
                    ON order_journal(timestamp);
                CREATE INDEX IF NOT EXISTS idx_order_journal_status
                    ON order_journal(status);
            """)

    # ---- Position Journal ----

    def record_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_id: str = "",
        reason: str = "",
        status: str = "confirmed",
    ) -> int:
        """Record a position change. Returns journal id."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                """INSERT INTO position_journal
                   (timestamp, symbol, side, quantity, price, order_id, status, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), symbol, side, quantity, price,
                 order_id, status, reason),
            )
            return cur.lastrowid or 0

    def get_positions(self, since: str | None = None) -> list[dict]:
        """Get position records since a timestamp."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if since:
                rows = conn.execute(
                    "SELECT * FROM position_journal WHERE timestamp >= ? ORDER BY timestamp",
                    (since,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM position_journal ORDER BY timestamp"
                ).fetchall()
            return [dict(r) for r in rows]

    def compute_current_positions(self) -> dict[str, dict]:
        """Replay journal to compute current positions."""
        positions: dict[str, dict] = {}
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT symbol, side, quantity, price FROM position_journal "
                "WHERE status = 'confirmed' ORDER BY id"
            ).fetchall()

        for sym, side, qty, price in rows:
            if sym not in positions:
                positions[sym] = {"quantity": 0, "total_cost": 0.0}

            if side == "buy":
                positions[sym]["quantity"] += qty
                positions[sym]["total_cost"] += qty * price
            elif side == "sell":
                positions[sym]["quantity"] -= qty
                # Reduce cost proportionally
                if positions[sym]["quantity"] > 0:
                    ratio = qty / (positions[sym]["quantity"] + qty)
                    positions[sym]["total_cost"] *= (1 - ratio)

        # Remove zero positions
        return {s: p for s, p in positions.items() if p["quantity"] > 0}

    # ---- Order Journal ----

    def record_order_event(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        status: str,
        price: float | None = None,
        order_type: str = "limit",
        filled_qty: int = 0,
        avg_fill_price: float | None = None,
        retry_count: int = 0,
        last_error: str = "",
    ) -> None:
        """Insert or update an order in the journal."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO order_journal
                   (timestamp, order_id, symbol, side, quantity, price, order_type,
                    status, filled_qty, avg_fill_price, retry_count, last_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), order_id, symbol, side, quantity,
                 price, order_type, status, filled_qty, avg_fill_price,
                 retry_count, last_error),
            )

    def get_order(self, order_id: str) -> dict | None:
        """Get a single order by ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM order_journal WHERE order_id = ?", (order_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_pending_orders(self) -> list[dict]:
        """Get all orders not in a terminal state."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM order_journal
                   WHERE status NOT IN ('filled', 'cancelled', 'rejected')
                   ORDER BY timestamp"""
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- Checkpoints ----

    def save_checkpoint(self, snapshot: dict) -> None:
        """Save a portfolio snapshot as a recovery checkpoint."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO checkpoints (timestamp, snapshot) VALUES (?, ?)",
                (datetime.now().isoformat(), json.dumps(snapshot)),
            )

    def get_last_checkpoint(self) -> tuple[str, dict] | None:
        """Get the most recent checkpoint. Returns (timestamp, snapshot) or None."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT timestamp, snapshot FROM checkpoints ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return (row[0], json.loads(row[1]))

    def get_journal_since(self, timestamp: str) -> list[dict]:
        """Get all position journal entries since a timestamp."""
        return self.get_positions(since=timestamp)

    # ---- Maintenance ----

    def prune(self, retention_days: int | None = None) -> int:
        """Delete journal entries older than retention period. Returns deleted count."""
        days = retention_days if retention_days is not None else settings.JOURNAL_RETENTION_DAYS
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if days == 0:
            cutoff = datetime.now() + timedelta(seconds=1)
        else:
            cutoff = cutoff - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                "DELETE FROM position_journal WHERE timestamp < ?", (cutoff_str,)
            )
            pos_deleted = cur.rowcount
            cur = conn.execute(
                "DELETE FROM order_journal WHERE timestamp < ? AND status IN ('filled', 'cancelled', 'rejected')",
                (cutoff_str,),
            )
            ord_deleted = cur.rowcount
            return pos_deleted + ord_deleted






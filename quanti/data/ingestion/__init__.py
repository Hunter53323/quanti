"""
Daily data ingestion pipeline.
Orchestrates Tushare + AkShare fetchers with fallback logic and persistence.
"""

from datetime import datetime

from quanti.config import settings
from quanti.data.storage import DataStorage
from quanti.data.ingestion.tushare_fetcher import TushareETFetcher
from quanti.data.ingestion.akshare_fetcher import AkShareETFetcher


def run_daily_ingest(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[bool, list[str]]:
    """
    Run daily ingestion for all configured ETF symbols.

    Priority: Tushare (primary) -> AkShare (fallback).

    Args:
        symbols: ETF symbols to fetch. Defaults to settings.ETF_UNIVERSE.
        start_date: YYYYMMDD. Defaults to 30 days ago.
        end_date: YYYYMMDD. Defaults to today.

    Returns:
        (ok, errors): True if at least one source succeeded for each symbol,
                      list of error messages.
    """
    symbols = symbols or settings.ETF_UNIVERSE
    storage = DataStorage()
    errors: list[str] = []
    overall_ok = True

    # Try Tushare first
    try:
        tushare = TushareETFetcher()
        tushare_ok = True
    except ValueError as e:
        errors.append(f"Tushare unavailable: {e}")
        tushare_ok = False

    akshare = AkShareETFetcher()

    for sym in symbols:
        symbol_ok = False

        # Primary: Tushare
        if tushare_ok:
            try:
                bars = tushare.fetch_daily(sym, start_date, end_date)
                if bars:
                    storage.save_bars_raw(sym, bars)
                    storage.save_bars_clean(sym, bars)
                    last_date = bars[-1].trade_date
                    storage.update_freshness(sym, last_date)
                    storage.log_ingestion("tushare", sym, start_date or "auto",
                                          end_date or "auto", len(bars), "success")
                    print(f"OK: {sym} -- {len(bars)} bars from Tushare (last: {last_date})")
                    symbol_ok = True
                else:
                    print(f"WARN: {sym} -- Tushare returned 0 bars, trying AkShare...")
            except Exception as e:
                msg = f"Tushare error for {sym}: {e}"
                errors.append(msg)
                storage.log_ingestion("tushare", sym, "", "", 0, "error", msg)

        # Fallback: AkShare
        if not symbol_ok:
            try:
                bars = akshare.fetch_daily(sym, start_date, end_date)
                if bars:
                    storage.save_bars_raw(sym, bars)
                    storage.save_bars_clean(sym, bars)
                    last_date = bars[-1].trade_date
                    storage.update_freshness(sym, last_date)
                    storage.log_ingestion("akshare", sym, start_date or "auto",
                                          end_date or "auto", len(bars), "success")
                    print(f"OK: {sym} -- {len(bars)} bars from AkShare (last: {last_date})")
                    symbol_ok = True
                else:
                    msg = f"No data from any source for {sym}"
                    errors.append(msg)
                    storage.log_ingestion("akshare", sym, "", "", 0, "error", msg)
                    overall_ok = False
            except Exception as e:
                msg = f"AkShare error for {sym}: {e}"
                errors.append(msg)
                storage.log_ingestion("akshare", sym, "", "", 0, "error", msg)
                overall_ok = False

    return (overall_ok, errors)

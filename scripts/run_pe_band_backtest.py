"""
PE-Band Dynamic Allocation Backtest Runner.

Backtests the PE-band strategy against historical data:
1. Fetches CSI300 PE time series (or loads from cache)
2. Fetches ETF bars for equity/bond/gold symbols
3. Runs backtest engine with PE data injection
4. Compares against buy-and-hold CSI300

Usage:
    python scripts/run_pe_band_backtest.py

Requires:
    - TUSHARE_TOKEN in .env
    - ETF data ingested (run_full_pipeline.py or fetch_history.py first)
"""

import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quanti.backtest.engine import BacktestEngine
from quanti.config import settings
from quanti.data.index_pe import IndexPEFetcher
from quanti.data.storage import DataStorage
from quanti.strategy.pe_band import PEBandAllocation


def get_pe_provider(source_index="000300.SH"):
    """
    Returns a callable that provides PE stats for any date in the backtest window.

    Pre-fetches the full PE history, then creates a rolling window lookup:
    For each backtest date, computes percentile over a trailing N-year window.
    This simulates what the live system would see (only past data).
    """
    fetcher = IndexPEFetcher()
    raw = fetcher.fetch_history(source_index)
    if not raw:
        print("ERROR: No PE data available. Check Tushare token.")
        return None

    print(f"  PE data: {len(raw)} records from {raw[0]['trade_date']} to {raw[-1]['trade_date']}")

    def pe_stats(date_str: str) -> dict | None:
        """Compute PE percentile using only data available up to date_str."""
        pe_vals = np.array([r["pe"] for r in raw if r["trade_date"] <= date_str and r["pe"] > 0], dtype=np.float64)
        if len(pe_vals) < 30:
            return None

        # Find the latest entry at or before date_str
        latest_pe = None
        for r in reversed(raw):
            if r["trade_date"] <= date_str:
                latest_pe = r["pe"]
                latest_pb = r["pb"]
                latest_date = r["trade_date"]
                break

        if latest_pe is None:
            return None

        # Use trailing 10-year window (or full history if shorter)
        window_start = datetime.strptime(date_str, "%Y%m%d")
        cutoff = window_start.strftime("%Y%m%d")
        ten_years_ago = window_start.replace(year=window_start.year - 10).strftime("%Y%m%d")

        # Use max(ten_years_ago, first available date)
        first_date = raw[0]["trade_date"]
        window_start_str = max(ten_years_ago, first_date)

        window_pe = np.array(
            [r["pe"] for r in raw
             if window_start_str <= r["trade_date"] <= cutoff and r["pe"] > 0],
            dtype=np.float64
        )

        if len(window_pe) < 20:
            window_pe = pe_vals  # Fallback to full history

        pe_pctile = float(np.sum(window_pe <= latest_pe) / len(window_pe)) * 100
        pe_pctile = max(5.0, min(95.0, pe_pctile))

        # Equity allocation: cheapest -> max equity, expensive -> min equity
        equity_max = settings.PE_BAND_EQUITY_MAX
        equity_min = settings.PE_BAND_EQUITY_MIN
        equity_pct = equity_max - (pe_pctile / 100.0) * (equity_max - equity_min)
        equity_pct = max(equity_min, min(equity_max, equity_pct)) * 100

        return {
            "trade_date": latest_date,
            "pe": float(latest_pe),
            "pe_percentile": round(pe_pctile, 1),
            "pb": float(latest_pb) if latest_pb else 0,
            "pb_percentile": 50.0,
            "equity_allocation_pct": round(equity_pct, 1),
        }

    return pe_stats


def main():
    print("=" * 60)
    print("PE-BAND DYNAMIC ALLOCATION BACKTEST")
    print("=" * 60)

    # 1. Load ETF data
    store = DataStorage()
    symbols = ["510300", "518880", "511880"]
    print("\nLoading ETF data...")
    bars_dict = {}
    for sym in symbols:
        bars = store.load_bars(sym)
        if not bars:
            print(f"  WARN: No data for {sym}. Run fetch_history.py or full_pipeline first.")
        else:
            print(f"  {sym}: {len(bars)} bars ({bars[0].trade_date} to {bars[-1].trade_date})")
            bars_dict[sym] = bars

    if len(bars_dict) < 2:
        print("\nERROR: Need at least equity + bond ETF data. Run data ingestion first.")
        print("  python scripts/fetch_history.py")
        return

    # 2. Build PE provider (with rolling percentile)
    print("\nLoading CSI300 PE data...")
    pe_provider = get_pe_provider("000300.SH")
    if pe_provider is None:
        return

    # Test PE provider on first and last dates
    all_dates = sorted({b.trade_date for v in bars_dict.values() for b in v})
    if all_dates:
        first_pe = pe_provider(all_dates[max(0, len(all_dates) - 1000)])
        if first_pe:
            print(f"  Sample PE stats (mid-period): PE={first_pe['pe']:.1f} ({first_pe['pe_percentile']:.0f}th pct) -> equity={first_pe['equity_allocation_pct']:.0f}%")
        last_pe = pe_provider(all_dates[-1])
        if last_pe:
            print(f"  Latest PE stats: PE={last_pe['pe']:.1f} ({last_pe['pe_percentile']:.0f}th pct) -> equity={last_pe['equity_allocation_pct']:.0f}%")

    # 3. Run PE-band backtest
    print("\nRunning PE-band backtest (2019-2025)...")

    # Split: in-sample 2019-2023, out-of-sample 2024-2025
    in_bars = {sym: [b for b in bars if "20190101" <= b.trade_date <= "20231231"] for sym, bars in bars_dict.items()}
    out_bars = {sym: [b for b in bars if "20240101" <= b.trade_date <= "20251231"] for sym, bars in bars_dict.items()}

    # Test multiple rebalance frequencies
    for freq_name, freq_days in [("Monthly", 21), ("Quarterly", 63), ("Semi-Annual", 126)]:
        print(f"\n  --- {freq_name} Rebalance ---")

        engine = BacktestEngine(
            strategy_class=PEBandAllocation,
            params=dict(rebalance_freq=freq_days),
            initial_capital=90000,
            pe_provider=pe_provider,
        )

        if in_bars and any(in_bars.values()):
            in_result = engine.run(list(in_bars.keys()), in_bars, period_label=f"in-sample-2019-2023-{freq_name}")
            print(f"    In-Sample:  {in_result.summarize()}")

        if out_bars and any(out_bars.values()):
            engine2 = BacktestEngine(
                strategy_class=PEBandAllocation,
                params=dict(rebalance_freq=freq_days),
                initial_capital=90000,
                pe_provider=pe_provider,
            )
            out_result = engine2.run(list(out_bars.keys()), out_bars, period_label=f"out-of-sample-2024-2025-{freq_name}")
            print(f"    OOS:        {out_result.summarize()}")

    # 4. Compare: buy-and-hold CSI300
    print("\n  --- Benchmark: Buy-and-Hold CSI300 (510300) ---")
    eq_bars = bars_dict.get("510300", [])
    if len(eq_bars) > 2:
        bh_start = eq_bars[0].close
        bh_end = eq_bars[-1].close
        bh_dates = len(eq_bars)
        bh_years = bh_dates / 252
        bh_return = (bh_end - bh_start) / bh_start * 100
        bh_cagr = ((bh_end / bh_start) ** (1 / max(bh_years, 0.1)) - 1) * 100
        bh_maxdd = 0.0
        peak = bh_start
        for b in eq_bars:
            if b.close > peak:
                peak = b.close
            dd = (peak - b.close) / peak * 100
            if dd > bh_maxdd:
                bh_maxdd = dd
        print(f"    Period: {eq_bars[0].trade_date} to {eq_bars[-1].trade_date}")
        print(f"    Total Return: {bh_return:.1f}%")
        print(f"    CAGR: {bh_cagr:.1f}%")
        print(f"    MaxDD: {bh_maxdd:.1f}%")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

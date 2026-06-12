"""
Dividend Barbell Strategy Backtest Runner.

Backtests the 510880 (中证红利) + bond + gold barbell strategy:
1. Loads ETF bars for 510880/518880/bond ETF
2. Runs backtest engine with quarterly rebalance
3. Compares against buy-and-hold 510880 and CSI300

Usage:
    python scripts/run_dividend_barbell_backtest.py

Requires:
    - ETF data ingested (run_full_pipeline.py or fetch_history.py first)
    - 510880.SH data available in clean/ storage
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quanti.backtest.engine import BacktestEngine
from quanti.data.storage import DataStorage
from quanti.strategy.dividend_barbell import DividendBarbell


def compute_bh_stats(bars, label: str) -> str:
    """Compute buy-and-hold return, CAGR, and max drawdown for a bar series."""
    if len(bars) < 10:
        return f"{label}: insufficient data"
    closes = [b.close for b in bars]
    dates = len(bars)
    years = dates / 252
    start_p = closes[0]
    end_p = closes[-1]
    total_ret = (end_p - start_p) / start_p * 100
    cagr = ((end_p / start_p) ** (1 / max(years, 0.1)) - 1) * 100 if start_p > 0 else 0
    peak = start_p
    max_dd = 0.0
    for c in closes:
        if c > peak:
            peak = c
        dd = (peak - c) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return (
        f"  {label}: {bars[0].trade_date} to {bars[-1].trade_date}\n"
        f"    Return: {total_ret:+.1f}%  CAGR: {cagr:+.1f}%  MaxDD: {max_dd:.1f}%  "
        f"Years: {years:.1f}"
    )


def main():
    print("=" * 60)
    print("DIVIDEND BARBELL STRATEGY BACKTEST")
    print("=" * 60)

    # 1. Load ETF data
    store = DataStorage()
    symbols = ["510880", "518880", "511880"]
    print("\nLoading ETF data...")
    bars_dict = {}
    for sym in symbols:
        bars = store.load_bars(sym)
        if not bars:
            print(f"  WARN: No data for {sym}. Run fetch_history.py first.")
        else:
            print(f"  {sym}: {len(bars)} bars ({bars[0].trade_date} to {bars[-1].trade_date})")
            bars_dict[sym] = bars

    if len(bars_dict) < 2:
        print("\nERROR: Need at least dividend + bond ETF data. Run data ingestion first.")
        return

    # 2. Split into IS/OOS periods
    in_bars = {
        sym: [b for b in bars if "20190101" <= b.trade_date <= "20231231"]
        for sym, bars in bars_dict.items()
    }
    out_bars = {
        sym: [b for b in bars if "20240101" <= b.trade_date <= "20251231"]
        for sym, bars in bars_dict.items()
    }

    # 3. Run barbell strategy
    engine_args = dict(initial_capital=90000)
    test_configs = [
        ("Quarterly (static)",   dict(rebalance_freq=63, dynamic_tilt=False)),
        ("Quarterly (tilt)",     dict(rebalance_freq=63, dynamic_tilt=True)),
        ("Monthly (static)",     dict(rebalance_freq=21, dynamic_tilt=False)),
        ("Semi-Annual (static)", dict(rebalance_freq=126, dynamic_tilt=False)),
    ]

    for label, params in test_configs:
        print(f"\n  --- {label} ---")
        for period_name, period_bars in [("In-Sample 2019-2023", in_bars),
                                          ("OOS 2024-2025", out_bars)]:
            if not any(period_bars.values()):
                continue
            engine = BacktestEngine(
                strategy_class=DividendBarbell,
                params=params,
                **engine_args,
            )
            result = engine.run(list(period_bars.keys()), period_bars, period_label=period_name)
            print(f"    {period_name}: CAGR={result.cagr_pct:+.1f}%  Sharpe={result.sharpe_ratio:.2f}  "
                  f"MaxDD={result.max_drawdown_pct:.1f}%  "
                  f"Trades={len(result.trades)}  "
                  f"WinRate={result.win_rate_pct:.1f}%")

    # 4. Buy-and-hold benchmarks
    print("\n  --- Benchmarks ---")
    eq_bars = store.load_bars("510300")
    if eq_bars:
        print(compute_bh_stats(eq_bars, "CSI300 Buy & Hold"))

    div_bars = store.load_bars("510880")
    if div_bars:
        print(compute_bh_stats(div_bars, "510880 Buy & Hold"))

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

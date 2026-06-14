"""Multi-sector ETF rotation backtest (2015-2025)."""
import sys, os, time
sys.path.insert(0, r"C:\study\AIWorkspace\quanti")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage
from quanti.backtest.engine import BacktestEngine, BacktestResult
from quanti.strategy.etf_rotation import ETFRotationStrategy


def main():
    t0 = time.time()
    storage = DataStorage()

    # --- Load available ETFs ---
    clean_dir = storage.clean_dir
    target_codes = [
        "510300","510500","159915","588000",
        "512880","512800","512480","515070","515880","512720",
        "515790","516160","516110",
        "159928","512010",
        "512660",
        "159825","516780","512400","516020",
        "512980","159869",
        "510880","518880","511880",
    ]

    bars_dict: dict[str, list] = {}
    for code in target_codes:
        bars = storage.load_bars(code)
        if bars and len(bars) >= 140:
            bars_dict[code] = bars
    print(f"Loaded {len(bars_dict)}/{len(target_codes)} ETFs with data")

    # --- Legacy Backtest (6-ETF baseline) ---
    print("\n" + "="*60)
    print("LEGACY BASELINE (6-ETF)")
    print("="*60)
    legacy_engine = BacktestEngine(
        strategy_class=ETFRotationStrategy,
        params=dict(use_multi_sector=False, top_n=3, equity_mandate=True),
    )
    legacy_result = legacy_engine.run(
        symbols=list(bars_dict.keys()),
        bars_dict=bars_dict,
        period_label="legacy-2015-2025",
    )
    print(f"  CAGR: {legacy_result.cagr_pct:+.2f}%  Sharpe: {legacy_result.sharpe_ratio:.3f}  MaxDD: {legacy_result.max_drawdown_pct:.1f}%  Trades: {len(legacy_result.trades)}")

    # --- Multi-Sector Backtest ---
    print("\n" + "="*60)
    print("MULTI-SECTOR (25-ETF dynamic pool)")
    print("="*60)

    multi_engine = BacktestEngine(
        strategy_class=ETFRotationStrategy,
        params=dict(use_multi_sector=True, top_n=3, max_per_sector=2, equity_mandate=False),
    )
    multi_result = multi_engine.run(
        symbols=list(bars_dict.keys()),
        bars_dict=bars_dict,
        period_label="multi-2015-2025",
    )
    print(f"  CAGR: {multi_result.cagr_pct:+.2f}%  Sharpe: {multi_result.sharpe_ratio:.3f}  MaxDD: {multi_result.max_drawdown_pct:.1f}%  Trades: {len(multi_result.trades)}")

    # --- Walk-Forward OOS ---
    print("\n" + "="*60)
    print("WALK-FORWARD OUT-OF-SAMPLE")
    print("="*60)

    # In-sample: 2015-2021, Out-of-sample: 2022-2025
    in_bars = {}
    out_bars = {}
    for sym, bars in bars_dict.items():
        in_bars[sym] = [b for b in bars if "20150101" <= b.trade_date <= "20211231"]
        out_bars[sym] = [b for b in bars if "20220101" <= b.trade_date <= "20251231"]

    is_eng = BacktestEngine(
        strategy_class=ETFRotationStrategy,
        params=dict(use_multi_sector=True, top_n=3, max_per_sector=2, equity_mandate=False),
    )
    oos_eng = BacktestEngine(
        strategy_class=ETFRotationStrategy,
        params=dict(use_multi_sector=True, top_n=3, max_per_sector=2, equity_mandate=False),
    )

    is_result = is_eng.run(
        symbols=list(in_bars.keys()),
        bars_dict=in_bars,
        period_label="in-sample-2015-2021",
    )
    oos_result = oos_eng.run(
        symbols=list(out_bars.keys()),
        bars_dict=out_bars,
        period_label="out-of-sample-2022-2025",
    )

    print(f"\n  In-Sample  (2015-2021):  CAGR={is_result.cagr_pct:+.2f}%  Sharpe={is_result.sharpe_ratio:.3f}  MaxDD={is_result.max_drawdown_pct:.1f}%  Trades={len(is_result.trades)}")
    print(f"  Out-of-Sample (2022-2025): CAGR={oos_result.cagr_pct:+.2f}%  Sharpe={oos_result.sharpe_ratio:.3f}  MaxDD={oos_result.max_drawdown_pct:.1f}%  Trades={len(oos_result.trades)}")

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    delta_cagr = multi_result.cagr_pct - legacy_result.cagr_pct
    delta_sharpe = multi_result.sharpe_ratio - legacy_result.sharpe_ratio
    delta_mdd = multi_result.max_drawdown_pct - legacy_result.max_drawdown_pct
    print(f"  Multi vs Legacy (full period):")
    print(f"    CAGR:  {delta_cagr:+.2f}%  ({legacy_result.cagr_pct:+.2f}% -> {multi_result.cagr_pct:+.2f}%)")
    print(f"    Sharpe: {delta_sharpe:+.3f}  ({legacy_result.sharpe_ratio:.3f} -> {multi_result.sharpe_ratio:.3f})")
    print(f"    MaxDD:  {delta_mdd:+.1f}%  ({legacy_result.max_drawdown_pct:.1f}% -> {multi_result.max_drawdown_pct:.1f}%)")

    oos_delta = oos_result.cagr_pct - 7.7
    print(f"\n  OOS CAGR vs original baseline (+7.7%): {oos_delta:+.2f}%")
    print(f"  OOS CAGR vs legacy OOS:              N/A (legacy OOS not run separately)")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

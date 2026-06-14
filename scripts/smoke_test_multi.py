"""End-to-end smoke test: multi-sector rotation on 2 months of data."""
import sys; sys.path.insert(0, r"C:\study\AIWorkspace\quanti")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.strategy.etf_rotation import ETFRotationStrategy
from quanti.data.storage import DataStorage
from quanti.backtest.engine import BacktestEngine


def main():
    storage = DataStorage()

    # Load all available sector ETFs
    target = [
        "510300","510500","159915","588000",
        "512880","512800","512480","515070","515880","512720",
        "515790","516160","516110",
        "159928","512010",
        "512660",
        "159825","516780","512400","516020",
        "512980","159869",
        "510880","518880","511880",
    ]

    bars_dict = {}
    for code in target:
        bars = storage.load_bars(code)
        if bars and len(bars) >= 140:
            bars_dict[code] = bars

    print(f"Loaded {len(bars_dict)} ETFs")

    if len(bars_dict) >= 8:
        # Run full backtest
        print("\n=== Legacy 6-ETF Baseline ===")
        eng = BacktestEngine(
            strategy_class=ETFRotationStrategy,
            params=dict(use_multi_sector=False, top_n=3, equity_mandate=True),
        )
        r = eng.run(list(bars_dict.keys()), bars_dict, period_label="legacy")
        print(f"  CAGR: {r.cagr_pct:+.2f}%  Sharpe: {r.sharpe_ratio:.3f}  MaxDD: {r.max_drawdown_pct:.1f}%")

        print("\n=== Multi-Sector (with concentration) ===")
        eng2 = BacktestEngine(
            strategy_class=ETFRotationStrategy,
            params=dict(use_multi_sector=True, top_n=3, max_per_sector=2),
        )
        r2 = eng2.run(list(bars_dict.keys()), bars_dict, period_label="multi")
        print(f"  CAGR: {r2.cagr_pct:+.2f}%  Sharpe: {r2.sharpe_ratio:.3f}  MaxDD: {r2.max_drawdown_pct:.1f}%")

        # Show sector diversity in trades
        if r2.trades:
            buys = [t for t in r2.trades if t["side"] == "buy"]
            from quanti.config.etf_universe import get_sector
            from collections import Counter
            sector_counts = Counter()
            for t in buys:
                sector_counts[get_sector(t["symbol"])] += 1
            print(f"\n  Sector distribution in buys: {dict(sector_counts)}")

        print(f"\n  Delta vs Legacy: CAGR={r2.cagr_pct-r.cagr_pct:+.2f}%")
    else:
        print("Not enough ETFs loaded for backtest")


if __name__ == "__main__":
    main()

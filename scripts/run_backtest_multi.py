"""
Backtest runner for multi-industry ETF rotation strategy.
Compares progressive enrollment vs static all-in (look-ahead bias test).

Usage:
    python scripts/run_backtest_multi.py
"""
import sys, time

_PROJECT_ROOT = r"C:\study\AIWorkspace\quanti"
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage
from quanti.backtest.engine import BacktestEngine
from quanti.strategy.etf_rotation_multi import ETFRotationMultiStrategy
from quanti.strategy.etf_universe import ALL_ETF_SYMBOLS


def filter_bars_by_date(bars_dict, start=None, end=None):
    """Return new bars_dict with bars filtered to [start, end] inclusive."""
    filtered = {}
    for sym, bars in bars_dict.items():
        fb = bars
        if start:
            fb = [b for b in fb if b.trade_date >= start]
        if end:
            fb = [b for b in fb if b.trade_date <= end]
        if fb:
            filtered[sym] = fb
    return filtered


def make_engine():
    """Create a standard BacktestEngine for this strategy."""
    return BacktestEngine(
        strategy_class=ETFRotationMultiStrategy,
        params=dict(top_n=3, max_per_category=2),
        initial_capital=90000,
        commission_rate=0.00025,
        slippage_bps=5,
    )


def run_backtest(engine, symbols, bars_dict, label, patch_universe=False):
    """Run a single backtest with optional get_eligible_etfs patching.

    When *patch_universe* is True, ``get_eligible_etfs`` is patched on the
    strategy module so it always returns ALL_ETF_SYMBOLS (static all-in).
    Progress is printed at 10% intervals during the run.
    """
    all_dates = sorted({b.trade_date for v in bars_dict.values() for b in v})
    total = len(all_dates)

    # Monkey-patch _mk_md for progress tracking
    original_mk_md = BacktestEngine._mk_md
    counter = {"n": 0, "total": total, "last_pct": 0}

    def _mk_md_progress(self, date, symbols, bars_dict, index_bars=None):
        counter["n"] += 1
        pct = int(counter["n"] / counter["total"] * 100)
        if pct >= counter["last_pct"] + 10:
            counter["last_pct"] = pct
            print(f"    Progress: {counter['n']}/{counter['total']} ({pct}%)")
        return original_mk_md(self, date, symbols, bars_dict, index_bars)

    BacktestEngine._mk_md = _mk_md_progress

    if patch_universe:
        import quanti.strategy.etf_rotation_multi as strat_mod
        original_get_eligible = strat_mod.get_eligible_etfs
        strat_mod.get_eligible_etfs = lambda dt: list(ALL_ETF_SYMBOLS)
    else:
        original_get_eligible = None

    try:
        print(f"  Running {label} ({total} trading days)...")
        t1 = time.time()
        result = engine.run(symbols, bars_dict, period_label=label)
        elapsed = time.time() - t1
        print(f"  Done in {elapsed:.0f}s, {len(result.trades)} trades")
        return result
    finally:
        BacktestEngine._mk_md = original_mk_md
        if original_get_eligible is not None:
            strat_mod.get_eligible_etfs = original_get_eligible


def run_period(bars_dict, period_name, start, end):
    """Run both variants for a given date range and print side-by-side comparison."""
    filtered = filter_bars_by_date(bars_dict, start=start, end=end)
    if not filtered:
        print(f"  [SKIP] {period_name}: no data")
        return None, None

    symbols = list(ALL_ETF_SYMBOLS)

    print(f"\n{'='*70}")
    print(f"  {period_name}")
    print(f"{'='*70}")

    eng = make_engine()
    prog_result = run_backtest(eng, symbols, filtered, f"progressive-{period_name}")

    eng2 = make_engine()
    static_result = run_backtest(
        eng2, symbols, filtered, f"static-{period_name}", patch_universe=True
    )

    print_side_by_side(period_name, prog_result, static_result)
    return prog_result, static_result


def print_side_by_side(label, pr, sr):
    """Print a clean side-by-side comparison table with deltas."""

    fields = [
        ("CAGR (%)",       "cagr_pct",             ".2f"),
        ("Sharpe",         "sharpe_ratio",          ".3f"),
        ("Max DD (%)",     "max_drawdown_pct",       ".2f"),
        ("Calmar",         "calmar_ratio",           ".3f"),
        ("Win Rate (%)",   "win_rate_pct",           ".2f"),
        ("Profit Factor",  "profit_factor",           ".3f"),
        ("Ann Turn (%)",   "annual_turnover_pct",    ".2f"),
    ]

    print(f"\n  {label}")
    print(f"  {'Metric':<20s} {'Progressive':>14s} {'Static':>14s} {'Delta':>12s}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*12}")

    for name, attr, fmt in fields:
        pv = getattr(pr, attr, 0.0) or 0.0
        sv = getattr(sr, attr, 0.0) or 0.0
        d = sv - pv
        pv_s = f"{pv:{fmt}}"
        sv_s = f"{sv:{fmt}}"
        d_s = f"{d:+{fmt}}"
        print(
            f"  {name:<20s} {pv_s:>14s} {sv_s:>14s} {d_s:>12s}"
        )

    # Trades (integer)
    nt_p = len(pr.trades)
    nt_s = len(sr.trades)
    print(f"  {'Num Trades':<20s} {nt_p:>14d} {nt_s:>14d} {nt_s - nt_p:+12d}")

    # Final equity
    fe_p = pr.equity_curve[-1] if pr.equity_curve else 0.0
    fe_s = sr.equity_curve[-1] if sr.equity_curve else 0.0
    print(f"  {'Final Eq ($)':<20s} {fe_p:>14.2f} {fe_s:>14.2f} {fe_s - fe_p:+12.2f}")


def print_delta_block(name, pr, sr):
    """Print a focused look-ahead bias delta summary."""
    if pr is None or sr is None:
        return
    print(f"\n  {name} — Look-ahead Bias (Static - Progressive):")
    print(f"    CAGR:           {sr.cagr_pct - pr.cagr_pct:+.2f}%")
    print(f"    Sharpe:         {sr.sharpe_ratio - pr.sharpe_ratio:+.4f}")
    print(f"    Max DD:         {sr.max_drawdown_pct - pr.max_drawdown_pct:+.2f}%")
    print(f"    Trades:         {len(sr.trades) - len(pr.trades):+d}")


def main():
    t0 = time.time()

    # ── 1. Load data ──
    print("=" * 70)
    print("  Loading ETF data from data/clean/...")
    print("=" * 70)
    storage = DataStorage()
    symbols = list(ALL_ETF_SYMBOLS)
    print(f"  Universe: {len(symbols)} symbols")

    bars_dict = {}
    for i, sym in enumerate(symbols):
        bars = storage.load_bars(sym)
        if bars:
            bars_dict[sym] = bars
        else:
            print(f"  WARNING: No data for {sym}")

    print(f"  Loaded {len(bars_dict)}/{len(symbols)} symbols with data")

    all_dates = sorted({b.trade_date for v in bars_dict.values() for b in v})
    if all_dates:
        print(f"  Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} trading days)")

    if not bars_dict:
        print("  ERROR: No data loaded. Aborting.")
        sys.exit(1)

    # ── 2. Validation period: 2022-2025 ──
    # (Full-period 2015-2025 is available via --full flag)
    period_name = "Validation (2022-2025)"
    prog_result, static_result = run_period(
        bars_dict, period_name, "20220101", "20251231"
    )

    # ── 3. Summary: Look-ahead bias deltas ──
    print(f"\n{'='*80}")
    print(f"  LOOK-AHEAD BIAS SUMMARY (Static - Progressive)")
    print(f"{'='*80}")
    print_delta_block(period_name, prog_result, static_result)

    print(f"\n{'='*70}")
    print(f"  TOTAL TIME: {time.time() - t0:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

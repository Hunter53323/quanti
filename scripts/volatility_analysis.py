"""
Volatility stress test for all 25 ETFs in the multi-industry rotation pool.
Phase 4.1: Are existing risk parameters adequate for volatile sector ETFs?

Risk parameters under review:
  - HWM trailing stop: -10%
  - DD breaker: -15% monthly
  - ATR trailing stop: 2x ATR(14)

For each ETF, computes:
  - Annualized volatility (rolling 60d, averaged)
  - Max single-day drawdown (close-to-close)
  - 95th/99th percentile daily range
  - % days with daily range > 5% (gap risk days)
  - % days with daily drop > 3% (large down days)
  - % of 60d rolling windows where HWM -10% stop would trigger
  - Average True Range as % of price
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quanti.data.storage import DataStorage


# ── ETF Pool: copied from audit_etf_listing.py ──
CATEGORIES = {
    "Broad":     ["510300", "510500", "159915", "588000"],
    "Finance":   ["512000", "512800"],
    "Tech":      ["512480", "515070", "515880", "512720"],
    "NewEnergy": ["516160", "516880", "516110"],
    "Consumer":  ["159928", "512010"],
    "Resources": ["159825", "516810", "516310", "516320"],
    "TMT":       ["512980", "159869"],
    "Defense":   ["512660"],
    "Defensive": ["510880", "518880", "511880"],
}

ALL_ETFS = [sym for syms in CATEGORIES.values() for sym in syms]

# ── Risk thresholds from existing config ──
HWM_STOP_PCT = 10.0         # -10% from high water mark
DD_BREAKER_PCT = 15.0       # -15% monthly drawdown breaker
ATR_MULT = 2.0              # ATR trailing stop multiplier
ATR_PERIOD = 14
TRADING_DAYS_PER_YEAR = 242


def load_bars(storage: DataStorage, symbol: str) -> np.ndarray:
    """Load OHLCV from clean parquet and return structured array.

    Returns None if no data found.
    """
    bars = storage.load_bars(symbol)
    if not bars:
        return None

    n = len(bars)
    out = np.zeros(n, dtype=[
        ("trade_date", "U10"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("volume", "f8"),
    ])
    for i, b in enumerate(bars):
        out[i] = (b.trade_date, b.open, b.high, b.low, b.close, b.volume)
    return out


def compute_metrics(data: np.ndarray) -> dict:
    """Compute all volatility metrics for a single ETF's daily bars."""
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    n = len(closes)

    if n < 252:
        return {"error": f"Only {n} trading days (need 252+ for reliable estimates)"}

    # ── Daily returns (close-to-close, %) ──
    daily_returns = np.diff(closes) / closes[:-1] * 100.0   # percent

    # ── 1. Annualized volatility (rolling 60d, then average) ──
    if len(daily_returns) >= 60:
        rolling_vol_60 = np.array([
            np.std(daily_returns[i:i+60]) * np.sqrt(TRADING_DAYS_PER_YEAR)
            for i in range(len(daily_returns) - 60 + 1)
        ])
        avg_vol = float(np.mean(rolling_vol_60))
    else:
        avg_vol = float(np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR))

    # ── 2. Max single-day drawdown (worst close-to-close drop) ──
    if len(daily_returns) > 0:
        max_single_day_dd = float(np.min(daily_returns))
    else:
        max_single_day_dd = 0.0

    # ── 3 & 4. 95th/99th percentile daily range ──
    daily_range_pct = (highs - lows) / closes * 100.0
    p95_range = float(np.percentile(daily_range_pct, 95))
    p99_range = float(np.percentile(daily_range_pct, 99))

    # ── 5. % of days where daily range > 5% (gap risk days) ──
    gap_risk_days_pct = float(np.mean(daily_range_pct > 5.0) * 100.0)

    # ── 6. % of days where daily drop > 3% (large down days) ──
    large_down_days_pct = float(np.mean(daily_returns < -3.0) * 100.0)

    # ── 7. % of time HWM -10% stop would trigger ──
    # Simulate: for each 60d rolling window, track peak close,
    # and see if any subsequent close falls >=10% below that peak.
    if len(daily_returns) >= 60:
        hwm_triggered = 0
        hwm_windows = len(daily_returns) - 60 + 1
        for i in range(hwm_windows):
            window_closes = closes[i:i+61]       # 61 close prices = 60 returns + current
            peak = np.maximum.accumulate(window_closes)
            dd_from_peak = (window_closes - peak) / peak * 100.0
            if np.any(dd_from_peak <= -HWM_STOP_PCT):
                hwm_triggered += 1
        hwm_trigger_pct = float(hwm_triggered / hwm_windows * 100.0)
    else:
        hwm_trigger_pct = float("nan")

    # ── 8. Average True Range as % of close ──
    # ATR = EMA of True Range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    if len(tr) >= ATR_PERIOD:
        # Simple approach: SMA of first ATR_PERIOD, then EMA
        atr_vals = np.zeros(len(tr))
        atr_vals[ATR_PERIOD - 1] = np.mean(tr[:ATR_PERIOD])
        alpha = 2.0 / (ATR_PERIOD + 1)
        for j in range(ATR_PERIOD, len(tr)):
            atr_vals[j] = alpha * tr[j] + (1 - alpha) * atr_vals[j - 1]
        # Use last value from EMA series as current ATR
        # But for a summary metric, average ATR over the whole history
        avg_atr = float(np.mean(atr_vals[ATR_PERIOD - 1:]))
    else:
        avg_atr = float(np.mean(tr)) if len(tr) > 0 else 0.0

    avg_close = float(np.mean(closes))
    avg_atr_pct = avg_atr / avg_close * 100.0 if avg_close > 0 else 0.0

    return {
        "avg_vol_pct": round(avg_vol, 1),
        "max_single_dd_pct": round(max_single_day_dd, 1),
        "p95_range_pct": round(p95_range, 2),
        "p99_range_pct": round(p99_range, 2),
        "gap_risk_days_pct": round(gap_risk_days_pct, 1),
        "large_down_days_pct": round(large_down_days_pct, 1),
        "hwm_trigger_pct": round(hwm_trigger_pct, 1),
        "avg_atr_pct": round(avg_atr_pct, 2),
    }


def print_summary_table(results: dict):
    """Print category-grouped summary table and flag risky categories."""
    print()
    print("=" * 120)
    print("VOLATILITY STRESS TEST -- CATEGORY SUMMARY")
    print("=" * 120)

    # Header
    header = f"{'Category':<14} {'ETFs':>5} {'AvgVol%':>9} {'MaxDD%':>8} {'P95Rng%':>8} {'P99Rng%':>8} {'GapDays%':>9} {'BigDrop%':>9} {'HWM10%':>8} {'AvgATR%':>8}  {'FLAG'}"
    print(header)
    print("-" * 120)

    category_data = {}
    for cat, symbols in CATEGORIES.items():
        cat_results = {}
        etf_count = 0
        for sym in symbols:
            r = results.get(sym, {})
            if "error" not in r:
                cat_results[sym] = r
                etf_count += 1

        if etf_count == 0:
            continue

        # Average across ETFs in category
        avg_vol = np.mean([r["avg_vol_pct"] for r in cat_results.values()])
        avg_maxdd = np.mean([r["max_single_dd_pct"] for r in cat_results.values()])
        avg_p95 = np.mean([r["p95_range_pct"] for r in cat_results.values()])
        avg_p99 = np.mean([r["p99_range_pct"] for r in cat_results.values()])
        avg_gap = np.mean([r["gap_risk_days_pct"] for r in cat_results.values()])
        avg_bigdrop = np.mean([r["large_down_days_pct"] for r in cat_results.values()])
        avg_hwm = np.mean([r["hwm_trigger_pct"] for r in cat_results.values()])
        avg_atr = np.mean([r["avg_atr_pct"] for r in cat_results.values()])

        # Flag: HWM -10% triggers >20% of windows
        flag = ""
        if avg_hwm > 20:
            flag = "<<< HWM TOO TIGHT"

        row = (
            f"{cat:<14} {etf_count:>5} {avg_vol:>9.1f} {avg_maxdd:>8.1f} "
            f"{avg_p95:>8.2f} {avg_p99:>8.2f} {avg_gap:>9.1f} {avg_bigdrop:>9.1f} "
            f"{avg_hwm:>8.1f} {avg_atr:>8.2f}  {flag}"
        )
        print(row)

        category_data[cat] = {
            "etf_count": etf_count,
            "avg_vol_pct": round(avg_vol, 1),
            "avg_max_dd_pct": round(avg_maxdd, 1),
            "avg_p95_range_pct": round(avg_p95, 2),
            "avg_p99_range_pct": round(avg_p99, 2),
            "avg_gap_risk_pct": round(avg_gap, 1),
            "avg_large_down_pct": round(avg_bigdrop, 1),
            "avg_hwm_trigger_pct": round(avg_hwm, 1),
            "avg_atr_pct": round(avg_atr, 2),
        }

    print("=" * 120)
    print()

    # ── Flagged categories ──
    print(">> FLAGGED CATEGORIES (HWM -10% triggers on > 20% of rolling windows)")
    print("-" * 60)
    flagged = [(cat, d) for cat, d in category_data.items() if d["avg_hwm_trigger_pct"] > 20]
    if flagged:
        for cat, d in sorted(flagged, key=lambda x: -x[1]["avg_hwm_trigger_pct"]):
            print(f"  {cat:12s}: HWM trigger={d['avg_hwm_trigger_pct']:5.1f}%  AvgVol={d['avg_vol_pct']:5.1f}%  GapRisk={d['avg_gap_risk_pct']:5.1f}%")
    else:
        print("  None -- all categories have HWM trigger <= 20%")
    print()

    return category_data


def print_detailed_table(results: dict):
    """Print per-ETF detailed results."""
    print()
    print("=" * 120)
    print("PER-ETF DETAILED RESULTS")
    print("=" * 120)

    header = f"{'Symbol':<10} {'Category':<12} {'AvgVol%':>9} {'MaxDD%':>8} {'P95Rng%':>8} {'P99Rng%':>8} {'GapDays%':>9} {'BigDrop%':>9} {'HWM10%':>8} {'AvgATR%':>8}"
    print(header)
    print("-" * 120)

    cat_for_sym = {}
    for cat, syms in CATEGORIES.items():
        for s in syms:
            cat_for_sym[s] = cat

    for sym in ALL_ETFS:
        r = results.get(sym, {})
        if "error" in r:
            print(f"{sym:<10} {'ERROR':<12} {r['error']}")
            continue
        cat = cat_for_sym.get(sym, "??")
        row = (
            f"{sym:<10} {cat:<12} {r['avg_vol_pct']:>9.1f} {r['max_single_dd_pct']:>8.1f} "
            f"{r['p95_range_pct']:>8.2f} {r['p99_range_pct']:>8.2f} {r['gap_risk_days_pct']:>9.1f} "
            f"{r['large_down_days_pct']:>9.1f} {r['hwm_trigger_pct']:>8.1f} {r['avg_atr_pct']:>8.2f}"
        )
        print(row)
    print("=" * 120)
    print()


def print_recommendations(category_data: dict):
    """Print risk parameter recommendations."""
    print("=" * 80)
    print("RISK PARAMETER RECOMMENDATIONS")
    print("=" * 80)

    # Identify categories needing adjustments
    for cat, d in sorted(category_data.items(), key=lambda x: -x[1]["avg_hwm_trigger_pct"]):
        hwm = d["avg_hwm_trigger_pct"]
        vol = d["avg_vol_pct"]
        max_dd = d["avg_max_dd_pct"]

        if hwm > 20:
            # Suggest wider HWM stop
            suggested_hwm = round(min(vol * 0.5, 20.0), 1)
            suggested_hwm = max(suggested_hwm, 12.0)
            print(f"\n  [{cat}] HIGH VOLATILITY SECTOR")
            print(f"    Current HWM stop: -10%  |  Trigger rate: {hwm:.1f}% of windows")
            print(f"    Avg annualized vol: {vol:.1f}%")
            print(f"    SUGGESTED HWM stop: -{suggested_hwm:.0f}%")
            print(f"    Current ATR trailing: 2x | Avg ATR%: {d['avg_atr_pct']:.2f}%")
            if d['avg_atr_pct'] > 3:
                widened_atr = round(ATR_MULT * 1.5, 1)
                print(f"    SUGGESTED ATR multiplier: {widened_atr:.1f}x (Avg ATR% = {d['avg_atr_pct']:.2f}%)")
            if d['avg_gap_risk_pct'] > 10:
                print(f"    NOTE: {d['avg_gap_risk_pct']:.1f}% of days have range > 5%. Consider gap-fill kill switch.")
        elif vol > 30 and hwm > 15:
            print(f"\n  [{cat}] MODERATELY ELEVATED VOLATILITY")
            print(f"    HWM trigger: {hwm:.1f}% -- monitor, may need loosening to -12%")

    # Summary recommendation
    print()
    print("-" * 80)
    print("GRAND SUMMARY")
    print("-" * 80)

    high_vol_cats = [d for d in category_data.values() if d["avg_vol_pct"] > 30]
    med_vol_cats = [d for d in category_data.values() if 20 < d["avg_vol_pct"] <= 30]
    low_vol_cats = [d for d in category_data.values() if d["avg_vol_pct"] <= 20]

    print(f"  High-volatility categories (AvgVol > 30%): {len(high_vol_cats)}")
    print(f"  Medium-volatility categories (20-30%):     {len(med_vol_cats)}")
    print(f"  Low-volatility categories (< 20%):         {len(low_vol_cats)}")

    # Overall assessment
    all_hwm = [d["avg_hwm_trigger_pct"] for d in category_data.values()]
    avg_hwm_all = np.mean(all_hwm)
    max_hwm_all = max(all_hwm)

    print()
    if avg_hwm_all > 15:
        print("  VERDICT: Current -10% HWM stop is TOO TIGHT for the rotation pool.")
        print("           Recommend category-specific HWM stops (10-15%) or a pool-wide -12% to -15%.")
    elif avg_hwm_all > 10:
        print("  VERDICT: Current -10% HWM stop is acceptable but borderline for certain sectors.")
        print("           Recommend monitoring, with -12% for Tech/NewEnergy categories.")
    else:
        print("  VERDICT: Current -10% HWM stop appears adequate for this pool.")

    if any(d["avg_atr_pct"] > 3.0 for d in category_data.values()):
        print("           Some ETFs have ATR% > 3% -- consider 3x ATR multiplier for those sectors.")
    print()


def main():
    print("=" * 80)
    print("VOLATILITY STRESS TEST -- Phase 4.1")
    print("25-ETF Multi-Industry Rotation Pool")
    print("=" * 80)

    storage = DataStorage()

    results = {}
    for i, sym in enumerate(ALL_ETFS, 1):
        print(f"  [{i:2d}/{len(ALL_ETFS)}] Loading {sym} ...", end=" ", flush=True)

        data = load_bars(storage, sym)
        if data is None:
            print("NO DATA")
            results[sym] = {"error": "No data found"}
            continue

        metrics = compute_metrics(data)
        if "error" in metrics:
            print(f"SKIP: {metrics['error']}")
            results[sym] = metrics
            continue

        results[sym] = metrics
        print(
            f"vol={metrics['avg_vol_pct']:5.1f}%  "
            f"maxDD={metrics['max_single_dd_pct']:6.1f}%  "
            f"HWM_trig={metrics['hwm_trigger_pct']:4.1f}%  "
            f"ATR={metrics['avg_atr_pct']:4.2f}%"
        )

    # ── Print tables ──
    print_detailed_table(results)
    category_data = print_summary_table(results)
    print_recommendations(category_data)

    # ── Save to JSON ──
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_path = output_dir / "volatility_analysis_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "analysis_timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_parameters": {
                "hwm_stop_pct": HWM_STOP_PCT,
                "dd_breaker_pct": DD_BREAKER_PCT,
                "atr_multiplier": ATR_MULT,
                "atr_period": ATR_PERIOD,
            },
            "per_etf": results,
            "by_category": category_data,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()

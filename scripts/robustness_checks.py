"""Robustness checks for multi-sector ETF rotation strategy.

1. Correlation heatmap of monthly returns
2. Weight sensitivity analysis
3. Top-N sensitivity
"""
import sys, os
sys.path.insert(0, r"C:\study\AIWorkspace\quanti")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from quanti.data.storage import DataStorage
from quanti.backtest.engine import BacktestEngine
from quanti.strategy.etf_rotation import ETFRotationStrategy


def correlation_heatmap(bars_dict):
    """Compute monthly return correlation matrix for all ETFs."""
    from datetime import datetime

    # Build monthly returns matrix
    all_dates = sorted({b.trade_date for v in bars_dict.values() for b in v})
    if not all_dates:
        return {}

    # Group into months
    months = {}
    for d in all_dates:
        m = d[:6]
        months.setdefault(m, []).append(d)

    # For each ETF, compute monthly close (last trading day of month)
    monthly_returns = {}
    for sym, bars in bars_dict.items():
        price_map = {b.trade_date: b.close for b in bars}
        month_closes = []
        month_keys = []
        for m, dates in sorted(months.items()):
            last_date = dates[-1]
            if last_date in price_map:
                month_closes.append(price_map[last_date])
                month_keys.append(m)
        if len(month_closes) >= 24:  # need at least 2 years
            rets = np.diff(month_closes) / np.array(month_closes[:-1])
            monthly_returns[sym] = (month_keys[1:], rets)

    # Compute correlation matrix
    syms = sorted(monthly_returns.keys())
    n = len(syms)
    if n < 3:
        return {}

    # Align returns to common months
    common_months = set(monthly_returns[syms[0]][0])
    for s in syms[1:]:
        common_months &= set(monthly_returns[s][0])
    common_months = sorted(common_months)

    if len(common_months) < 12:
        return {}

    ret_matrix = np.zeros((len(common_months), n))
    for j, sym in enumerate(syms):
        month_map = dict(zip(monthly_returns[sym][0], monthly_returns[sym][1]))
        for i, m in enumerate(common_months):
            ret_matrix[i, j] = month_map.get(m, 0.0)

    corr = np.corrcoef(ret_matrix.T)

    # Find high-correlation pairs
    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > 0.80:
                high_pairs.append((syms[i], syms[j], corr[i, j]))

    return {
        "syms": syms,
        "corr": corr,
        "high_pairs": high_pairs,
        "n_months": len(common_months),
    }


def weight_sensitivity(bars_dict):
    """Test CAGR sensitivity to scoring weight changes."""
    results = []

    for w_trend in [0.30, 0.35, 0.40, 0.45]:
        for w_adx in [0.35, 0.40, 0.45]:
            w_mom = 1.0 - w_trend - w_adx
            if w_mom < 0.15 or w_mom > 0.35:
                continue

            engine = BacktestEngine(
                strategy_class=ETFRotationStrategy,
                params=dict(
                    use_multi_sector=True, top_n=3, max_per_sector=2,
                    w_trend=w_trend, w_adx=w_adx, w_momentum=w_mom,
                ),
            )
            result = engine.run(
                symbols=list(bars_dict.keys()),
                bars_dict=bars_dict,
                period_label=f"w{w_trend:.0f}_{w_adx:.0f}_{w_mom:.0f}",
            )
            results.append({
                "w_trend": w_trend, "w_adx": w_adx, "w_momentum": w_mom,
                "cagr": result.cagr_pct,
                "sharpe": result.sharpe_ratio,
                "maxdd": result.max_drawdown_pct,
            })

    return results


def top_n_sensitivity(bars_dict):
    """Test sensitivity to number of positions (N=2,3,4,5)."""
    results = []

    for n in [2, 3, 4, 5]:
        for max_per in [1, 2]:
            if max_per > n:
                continue
            engine = BacktestEngine(
                strategy_class=ETFRotationStrategy,
                params=dict(
                    use_multi_sector=True, top_n=n, max_per_sector=max_per,
                ),
            )
            result = engine.run(
                symbols=list(bars_dict.keys()),
                bars_dict=bars_dict,
                period_label=f"top{n}_max{max_per}",
            )
            results.append({
                "top_n": n, "max_per_sector": max_per,
                "cagr": result.cagr_pct,
                "sharpe": result.sharpe_ratio,
                "maxdd": result.max_drawdown_pct,
            })

    return results


def main():
    storage = DataStorage()
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

    bars_dict = {}
    for code in target_codes:
        bars = storage.load_bars(code)
        if bars and len(bars) >= 140:
            bars_dict[code] = bars
    print(f"Loaded {len(bars_dict)} ETFs")

    # 1. Correlation
    print("\n" + "="*50)
    print("1. CORRELATION HEATMAP")
    print("="*50)
    corr_result = correlation_heatmap(bars_dict)
    if corr_result:
        print(f"  Months: {corr_result['n_months']}  ETFs: {len(corr_result['syms'])}")
        if corr_result["high_pairs"]:
            print(f"  High-correlation pairs (|r| > 0.80): {len(corr_result['high_pairs'])}")
            for s1, s2, r in sorted(corr_result["high_pairs"], key=lambda x: -abs(x[2]))[:10]:
                print(f"    {s1} <-> {s2}: r={r:+.3f}")
        else:
            print("  No pairs with |r| > 0.80")

    # 2. Weight Sensitivity
    print("\n" + "="*50)
    print("2. WEIGHT SENSITIVITY")
    print("="*50)
    ws_results = weight_sensitivity(bars_dict)
    if ws_results:
        cagrs = [r["cagr"] for r in ws_results]
        print(f"  Weight combos tested: {len(ws_results)}")
        print(f"  CAGR range: {min(cagrs):+.2f}% ~ {max(cagrs):+.2f}%")
        print(f"  CAGR std:   {np.std(cagrs):.2f}%")
        best = max(ws_results, key=lambda x: x["cagr"])
        print(f"  Best: trend={best['w_trend']:.2f} adx={best['w_adx']:.2f} mom={best['w_momentum']:.2f} -> CAGR={best['cagr']:+.2f}%")
        # Find default
        default = next((r for r in ws_results if r["w_trend"] == 0.35 and r["w_adx"] == 0.40), None)
        if default:
            print(f"  Default (0.35/0.40/0.25): CAGR={default['cagr']:+.2f}%")

    # 3. Top-N Sensitivity
    print("\n" + "="*50)
    print("3. TOP-N SENSITIVITY")
    print("="*50)
    tn_results = top_n_sensitivity(bars_dict)
    if tn_results:
        print(f"{'N':>3s} {'Max/Sec':>7s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>7s}")
        for r in sorted(tn_results, key=lambda x: (x["top_n"], x["max_per_sector"])):
            print(f"{r['top_n']:3d} {r['max_per_sector']:7d} {r['cagr']:+7.2f}% {r['sharpe']:6.3f} {r['maxdd']:6.1f}%")


if __name__ == "__main__":
    main()

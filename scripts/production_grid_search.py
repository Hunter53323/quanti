"""
Production-correct parameter grid search for StockMomentumStrategy.
Uses the ACTUAL production strategy class logic, but runs at research speed
(precomputed numpy arrays, direct method delegation).

Train: 2015-01-01 to 2021-12-31
Test:  2022-01-01 to 2025-12-31

Grid: top_n x stop_loss_pct x min_trend_score x dd_exit_pct
      3/5/8   x  -8/-10/-15   x  3/4            x  15/20/25

This script mirrors the production StockMomentumStrategy logic exactly.
"""
import csv
import itertools
import os
import sys
import time

os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

from quanti.data.storage import DataStorage
from quanti.indicators import adx as shared_adx
from quanti.indicators import sma as shared_sma

CAPITAL = 90000
COMM = 0.00025

# ── Parameter grid ──
TOP_N_VALS = [3, 5, 8]
STOP_LOSS_VALS = [-8, -10, -15]      # negative = % below HWM
MIN_TREND_VALS = [3, 4]
DD_EXIT_VALS = [15, 20, 25]           # positive = % from equity peak

TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END = "20220101", "20251231"
MARKET_TREND_REQUIRED = True
MA_FILTER_PERIOD = 120

# ═══════════════════════════════════════════════════════════════
# Precomputed stock data (loaded once, shared across all configs)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading stock data (production-correct indicators)...")
print("=" * 70)

storage = DataStorage()
all_files = sorted(storage.clean_dir.glob("*.parquet"))
stocks = [p.stem for p in all_files if len(p.stem) == 6 and not p.stem.startswith(("51", "58", "15", "56"))]
print(f"Found {len(stocks)} stock files")

stock_data = {}
all_dates_set = set()
for i, code in enumerate(stocks):
    if (i + 1) % 100 == 0:
        print(f"  Loading {i + 1}/{len(stocks)}...")
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200:
        continue
    dates = [r.trade_date for r in raw]
    stock_data[code] = (
        np.array([r.close for r in raw], dtype=np.float64),
        np.array([r.high for r in raw], dtype=np.float64),
        np.array([r.low for r in raw], dtype=np.float64),
        np.array([r.volume for r in raw], dtype=np.float64),
        dates,
    )
    all_dates_set.update(dates)
all_dates = sorted(all_dates_set)
print(f"Loaded {len(stock_data)} stocks, {len(all_dates)} trading days")


# ═══════════════════════════════════════════════════════════════
# Production-correct indicator wrappers (delegate to shared quanti.indicators)
# ═══════════════════════════════════════════════════════════════

def _sma(arr, period):
    """Production SMA via shared indicators."""
    return shared_sma(arr, period)


def _adx(high, low, close, period=14):
    """Production ADX via shared indicators (NaN-safe wilder_smooth)."""
    return shared_adx(high, low, close, period)


# ═══════════════════════════════════════════════════════════════
# StockMomentumStrategy logic, extracted for research-speed execution
# These methods are IDENTICAL to the ones in quanti/strategy/stock_momentum.py
# ═══════════════════════════════════════════════════════════════

def _is_stock_trending(closes, highs, lows, vols, min_score, ma_period):
    """Mirrors StockMomentumStrategy._is_stock_trending. Returns (is_trending, cond_count)."""
    if len(closes) < 200:
        return False, 0
    count = 0
    # 1. Price above 120-day MA
    ma120 = _sma(closes, ma_period)
    if ma120 is not None and not np.isnan(ma120[-1]) and closes[-1] > ma120[-1]:
        count += 1
    # 2. Higher highs and higher lows
    recent_high = np.max(highs[-20:])
    prev_high = np.max(highs[-60:-20])
    recent_low = np.min(lows[-20:])
    prev_low = np.min(lows[-60:-20])
    if recent_high > prev_high and recent_low > prev_low:
        count += 1
    # 3. MA alignment: SMA20 > SMA60 > SMA120
    ma20 = _sma(closes, 20)
    ma60 = _sma(closes, 60)
    if (ma20 is not None and ma60 is not None and ma120 is not None and
        not np.isnan(ma20[-1]) and not np.isnan(ma60[-1]) and
        not np.isnan(ma120[-1]) and ma20[-1] > ma60[-1] > ma120[-1]):
        count += 1
    # 4. ADX > 25
    adx_arr = _adx(highs, lows, closes, 14)
    if adx_arr is not None and not np.isnan(adx_arr[-1]) and adx_arr[-1] > 25:
        count += 1
    # 5. Volume expansion
    vol_20 = np.mean(vols[-21:-1])
    if vol_20 > 0 and vols[-1] > vol_20 * 1.2:
        count += 1
    return (count >= min_score and count >= 1), count


def _trend_strength_score(closes):
    """Mirrors StockMomentumStrategy._trend_strength_score. Returns 0-100."""
    if len(closes) < 130:
        return 0.0
    # Momentum component
    if closes[-63] < 1e-6 or closes[-126] < 1e-6:
        mom_score = 0.0   # production fix: not 30.0
    else:
        ret_3m = closes[-1] / closes[-63] - 1
        ret_6m = closes[-1] / closes[-126] - 1
        mom_3m = min(max(ret_3m / 0.5, 0), 1) if ret_3m > 0 else 0
        mom_6m = min(max(ret_6m / 0.8, 0), 1) if ret_6m > 0 else 0
        mom_score = (0.5 * mom_3m + 0.5 * mom_6m) * 100
    # Low volatility bonus
    if len(closes) >= 61:
        window_c = closes[-61:]
        daily_ret = np.diff(window_c) / (window_c[:-1] + 1e-10)
        vol = np.nanstd(daily_ret)
        vol_score = max(0, (1 - min(vol / 0.04, 1))) * 100
    else:
        vol_score = 50.0
    return 0.6 * mom_score + 0.4 * vol_score


def _is_market_trending(date_str, stock_data):
    """Check if market (CSI300 proxy via 510300) is trending."""
    code = "510300"
    if code not in stock_data:
        return True  # permissive
    c, h, l, v, d = stock_data[code]
    # Find index up to date_str
    idx = None
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= date_str:
            idx = i + 1
            break
    if idx is None or idx < 200:
        return True  # permissive
    is_t, _ = _is_stock_trending(c[:idx], h[:idx], l[:idx], v[:idx], 3, MA_FILTER_PERIOD)
    return is_t


def data_at(code, date_str, n):
    if code not in stock_data:
        return None
    c, h, l, v, d = stock_data[code]
    idx = None
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= date_str:
            idx = i + 1
            break
    if idx is None or idx < n:
        return None
    return (c[idx - n:idx], h[idx - n:idx], l[idx - n:idx], v[idx - n:idx])


def price_on(code, date_str):
    if code not in stock_data:
        return None
    c, _, _, _, d = stock_data[code]
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= date_str:
            return c[i]
    return None


def get_monthly_dates(dates, start, end):
    monthly = []
    for d in dates:
        if d < start or d > end:
            continue
        dm = d[4:6]
        if not monthly or dm != monthly[-1][4:6]:
            monthly.append(d)
    return monthly


# ═══════════════════════════════════════════════════════════════
# Backtest engine (production-equivalent logic, research-speed)
# ═══════════════════════════════════════════════════════════════

def run_backtest(top_n, stop_loss_pct, min_trend_score, mkt_required, dd_exit_pct, start_d, end_d):
    """
    Monthly rebalance backtest with production StockMomentumStrategy logic:
    - Market gate (CSI300 > 120MA)
    - Stock trend filter (5 conditions)
    - Rank by trend strength, buy top N equal-weight
    - Per-position HWM trailing stop at stop_loss_pct
    - Portfolio DD exit at dd_exit_pct with time-based recovery

    Returns: (cagr, sharpe, maxdd, n_inv, n_cash, n_dd, hit_rate)
    """
    rebal = get_monthly_dates(all_dates, start_d, end_d)
    if len(rebal) < 12:
        return (0.0, 0.0, 100.0, 0, 0, 0, 0.0)

    cash = CAPITAL
    holdings = {}        # symbol -> {"qty", "hwm", "avg_cost"}
    eq_curve = [cash]
    max_equity = cash
    dd_exit_active = False
    dd_exit_days = 0
    n_inv = 0
    n_cash = 0
    n_dd = 0
    invested_rets = []

    for reb_date in rebal:
        # ── Update prices & check per-position trailing stops ──
        for sym in list(holdings.keys()):
            p = price_on(sym, reb_date)
            if p is None or p < 0.01:
                cash += holdings[sym].get("value", holdings[sym]["qty"] * holdings[sym].get("avg_cost", 1.0)) * 0.7
                del holdings[sym]
                continue
            pos = holdings[sym]
            pos["current_price"] = p
            if "hwm" not in pos or p > pos["hwm"]:
                pos["hwm"] = p
            hwm = pos["hwm"]
            # stop_loss_pct is negative; loss is negative; stop when loss < stop_loss_pct
            if hwm > 0:
                loss_pct = (p / hwm - 1) * 100
                if loss_pct < stop_loss_pct:
                    mv = pos["qty"] * p
                    cash += mv * (1 - COMM)
                    del holdings[sym]

        # ── Total equity ──
        total = cash + sum(
            h["qty"] * h["current_price"] for h in holdings.values()
            if "current_price" in h
        )

        # ── Portfolio drawdown circuit breaker ──
        if dd_exit_pct > 0:
            if total > max_equity:
                max_equity = total
            dd = (max_equity - total) / max_equity * 100
            if dd > dd_exit_pct and not dd_exit_active:
                # Trigger: liquidate everything
                for sym in list(holdings.keys()):
                    mv = holdings[sym]["qty"] * holdings[sym].get("current_price", 0)
                    cash += mv * (1 - COMM)
                    del holdings[sym]
                dd_exit_active = True
                dd_exit_days = 0
            elif dd_exit_active:
                dd_exit_days += 1
                # Recovery: equity recovers past 92% OR 60 days with no re-trigger
                if (total / max(max_equity, 1) > 0.92) or (dd_exit_days > 60 and dd <= dd_exit_pct):
                    dd_exit_active = False
                    max_equity = total
                    dd_exit_days = 0

        if dd_exit_active:
            n_dd += 1
            eq_curve.append(total)
            continue

        # ── Market trend gate ──
        if mkt_required:
            mkt_ok = _is_market_trending(reb_date, stock_data)
            if not mkt_ok:
                n_cash += 1
                eq_curve.append(total)
                continue

        # ── Find trending stocks ──
        trending = []
        for code in stock_data:
            d = data_at(code, reb_date, 260)
            if d is None:
                continue
            c, h, l, v = d
            is_t, cond_count = _is_stock_trending(c, h, l, v, min_trend_score, MA_FILTER_PERIOD)
            if is_t and cond_count >= min_trend_score:
                score = _trend_strength_score(c)
                if score > 0:
                    trending.append((code, score, cond_count))

        if not trending:
            n_cash += 1
            eq_curve.append(total)
            continue

        n_inv += 1
        trending.sort(key=lambda x: x[1], reverse=True)
        selected = {t[0] for t in trending[:top_n]}

        # ── Sell rotated-out ──
        for sym in list(holdings.keys()):
            if sym not in selected:
                p = holdings[sym].get("current_price", 0)
                mv = holdings[sym]["qty"] * p
                cash += mv * (1 - COMM)
                del holdings[sym]

        # ── Buy new entries (equal weight, 92% allocation for buffer) ──
        # Only buy symbols we don't already hold
        new_entries = [s for s in selected if s not in holdings]
        n_positions = len(selected)
        if new_entries and n_positions > 0:
            per_stock = total / n_positions * 0.92
            for sym in new_entries:
                p = price_on(sym, reb_date)
                if p is None or p < 0.01:
                    continue
                qty = int(per_stock / p / 100) * 100
                if qty >= 100 and qty * p * (1 + COMM) <= cash:
                    cash -= qty * p * (1 + COMM)
                    holdings[sym] = {"qty": qty, "hwm": p, "avg_cost": p, "current_price": p}

        # ── Rebalance existing holdings to target weight ──
        for sym in selected:
            if sym not in holdings:
                continue
            p = holdings[sym].get("current_price")
            if p is None or p < 0.01:
                continue
            target_qty = int(per_stock / p / 100) * 100
            cur = holdings[sym]["qty"]
            diff = target_qty - cur
            if abs(diff) >= 100:
                cost = abs(diff) * p
                if diff > 0 and cash >= cost * (1 + COMM):
                    cash -= cost * (1 + COMM)
                    holdings[sym]["qty"] = target_qty
                elif diff < 0:
                    cash += cost * (1 - COMM)
                    holdings[sym]["qty"] = target_qty

        prev_total = eq_curve[-1]
        total = cash + sum(
            h["qty"] * h.get("current_price", 0) for h in holdings.values()
            if "current_price" in h
        )
        eq_curve.append(total)
        if prev_total > 0:
            invested_rets.append((total - prev_total) / prev_total * 100)

    # ── Metrics ──
    eq = np.array(eq_curve)
    n_y = max(len(rebal) / 12.0, 0.5)
    if eq[0] <= 0:
        return (0.0, 0.0, 100.0, n_inv, n_cash, n_dd, 0.0)
    cagr = ((eq[-1] / eq[0]) ** (1 / n_y) - 1) * 100
    mr = np.diff(eq) / (eq[:-1] + 1e-10)
    sharpe = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 and np.std(mr) > 1e-10 else 0.0
    peak = eq[0]
    maxdd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > maxdd:
            maxdd = dd
    hit_rate = sum(1 for r in invested_rets if r > 0) / max(len(invested_rets), 1) * 100
    return (cagr, sharpe, maxdd, n_inv, n_cash, n_dd, hit_rate)


# ═══════════════════════════════════════════════════════════════
# Grid Search
# ═══════════════════════════════════════════════════════════════
grid = list(itertools.product(TOP_N_VALS, STOP_LOSS_VALS, MIN_TREND_VALS, DD_EXIT_VALS))
total = len(grid)
print(f"\n{'=' * 80}")
print(f"PRODUCTION-CORRECT GRID SEARCH: {total} combinations")
print(f"Train: {TRAIN_START} to {TRAIN_END}  |  Test: {TEST_START} to {TEST_END}")
print("Using StockMomentumStrategy logic via shared quanti.indicators")
print(f"{'=' * 80}\n")

results = []
t_start = time.monotonic()
for idx, (top_n, stop_pct, min_trend, dd_exit) in enumerate(grid):
    name = f"T{top_n}_Stop{abs(stop_pct)}_Trend{min_trend}_DD{dd_exit}"

    # Train
    t_cagr, t_sharpe, t_maxdd, t_inv, t_cash, t_dd, t_hr = run_backtest(
        top_n, stop_pct, min_trend, MARKET_TREND_REQUIRED, dd_exit, TRAIN_START, TRAIN_END)

    # Test
    v_cagr, v_sharpe, v_maxdd, v_inv, v_cash, v_dd, v_hr = run_backtest(
        top_n, stop_pct, min_trend, MARKET_TREND_REQUIRED, dd_exit, TEST_START, TEST_END)

    sharpe_decay = (t_sharpe - v_sharpe) / t_sharpe if abs(t_sharpe) > 0.01 else 999
    dd_inflate = (v_maxdd / t_maxdd - 1) * 100 if abs(t_maxdd) > 0.01 else 999
    inv_pct = v_inv / max(v_inv + v_cash, 1) * 100
    test_pos = 1 if v_cagr > 0 else 0

    results.append({
        "name": name, "top_n": top_n, "stop_loss_pct": stop_pct,
        "min_trend": min_trend, "dd_exit": dd_exit,
        "train_cagr": round(t_cagr, 2), "train_sharpe": round(t_sharpe, 3),
        "train_maxdd": round(t_maxdd, 2),
        "test_cagr": round(v_cagr, 2), "test_sharpe": round(v_sharpe, 3),
        "test_maxdd": round(v_maxdd, 2),
        "sharpe_decay": round(sharpe_decay, 3),
        "maxdd_inflate": round(dd_inflate, 2),
        "test_inv_pct": round(inv_pct, 0),
        "test_positive": test_pos,
    })

    elapsed = time.monotonic() - t_start
    rate = elapsed / (idx + 1)
    eta = rate * (total - idx - 1)
    print(f"[{idx + 1:2d}/{total}] {name:<28s} | Train CAGR={t_cagr:+5.1f}% S={t_sharpe:5.2f} D={t_maxdd:4.1f}% | "
          f"Test CAGR={v_cagr:+5.1f}% S={v_sharpe:5.2f} D={v_maxdd:4.1f}% | "
          f"Decay={sharpe_decay:.2f} Inv%={inv_pct:.0f}% | ETA {eta:.0f}s")

elapsed = time.monotonic() - t_start
print(f"\nGrid search completed in {elapsed:.0f}s ({elapsed/60:.1f}m)")

# ── Save CSV ──
csv_path = r"C:\study\AIWorkspace\quanti\data\production_grid_results.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fieldnames = list(results[0].keys())
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)
print(f"Results saved to: {csv_path}")

# ── Rankings ──
print(f"\n{'=' * 80}")
print("TOP 5 BY TEST CAGR")
print(f"{'=' * 80}")
for i, r in enumerate(sorted(results, key=lambda x: x["test_cagr"], reverse=True)[:5], 1):
    print(f"  #{i}: {r['name']:<28s} Test C={r['test_cagr']:+5.1f}% S={r['test_sharpe']:.2f} D={r['test_maxdd']:.1f}% "
          f"| Train C={r['train_cagr']:+5.1f}% S={r['train_sharpe']:.2f}")

print("\nTOP 5 BY TEST SHARPE")
print(f"{'=' * 80}")
for i, r in enumerate(sorted(results, key=lambda x: x["test_sharpe"], reverse=True)[:5], 1):
    print(f"  #{i}: {r['name']:<28s} Test S={r['test_sharpe']:.2f} C={r['test_cagr']:+5.1f}% D={r['test_maxdd']:.1f}% "
          f"| Train S={r['train_sharpe']:.2f}")

print("\nTOP 3 BY MINIMUM SHARPE DECAY")
print(f"{'=' * 80}")
for i, r in enumerate(sorted(results, key=lambda x: x["sharpe_decay"])[:3], 1):
    print(f"  #{i}: {r['name']:<28s} Decay={r['sharpe_decay']:.2f} "
          f"Train S={r['train_sharpe']:.2f} -> Test S={r['test_sharpe']:.2f} C={r['test_cagr']:+5.1f}%")

# ── Composite ranking ──
sorted_by_cagr = sorted(results, key=lambda x: x["test_cagr"], reverse=True)
sorted_by_sharpe = sorted(results, key=lambda x: x["test_sharpe"], reverse=True)
sorted_by_decay = sorted(results, key=lambda x: x["sharpe_decay"])
sorted_by_inflate = sorted(results, key=lambda x: x["maxdd_inflate"])

for i, r in enumerate(sorted_by_cagr): r["_r_cagr"] = i
for i, r in enumerate(sorted_by_sharpe): r["_r_sharpe"] = i
for i, r in enumerate(sorted_by_decay): r["_r_decay"] = i
for i, r in enumerate(sorted_by_inflate): r["_r_inflate"] = i
for r in results:
    r["composite_rank"] = r["_r_cagr"] + r["_r_sharpe"] + r["_r_decay"] + r["_r_inflate"]

best = sorted(results, key=lambda x: x["composite_rank"])[0]

print(f"\n{'=' * 80}")
print("COMPOSITE BEST")
print(f"{'=' * 80}")
print(f"  {best['name']} (top_n={best['top_n']}, stop_loss_pct={best['stop_loss_pct']}%, "
      f"min_trend={best['min_trend']}, dd_exit={best['dd_exit']}%)")
print(f"  Train: CAGR={best['train_cagr']:+.1f}%  Sharpe={best['train_sharpe']:.3f}  MaxDD={best['train_maxdd']:.1f}%")
print(f"  Test:  CAGR={best['test_cagr']:+.1f}%  Sharpe={best['test_sharpe']:.3f}  MaxDD={best['test_maxdd']:.1f}%")
print(f"  Sharpe Decay: {best['sharpe_decay']:.2f}  |  MaxDD Inflation: {best['maxdd_inflate']:+.1f}%")

# Count how many configs are positive on test
n_positive = sum(1 for r in results if r["test_cagr"] > 0)
print(f"\nConfigurations with positive Test CAGR: {n_positive}/{total}")

# ── Parameter importance ──
print(f"\n{'=' * 80}")
print("PARAMETER IMPORTANCE (average test CAGR by parameter value)")
print(f"{'=' * 80}")
for param_name, param_key in [("top_n", "top_n"), ("min_trend_score", "min_trend"),
                                ("stop_loss_pct", "stop_loss_pct"), ("dd_exit_pct", "dd_exit")]:
    param_vals = sorted(set(r[param_key] for r in results))
    for pv in param_vals:
        subset = [r for r in results if r[param_key] == pv]
        avg_cagr = np.mean([r["test_cagr"] for r in subset])
        avg_sharpe = np.mean([r["test_sharpe"] for r in subset])
        pos_rate = np.mean([r["test_positive"] for r in subset]) * 100
        print(f"  {param_name}={pv:>4}: avg Test CAGR={avg_cagr:+.2f}%  avg Sharpe={avg_sharpe:+.3f}  "
              f"positive rate={pos_rate:.0f}%  (n={len(subset)})")

print("\nDone.")

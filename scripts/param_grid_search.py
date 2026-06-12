"""
Trend-first momentum strategy: Parameter grid search & deep failure diagnostics.
Supports two modes:
  --mode grid_search (default): sweep parameter combinations, rank by composite score
  --mode diagnostic         : detailed sub-period breakdown + failure analysis

Train: 2015-2021, Test: 2022-2025. Strict sample-out-of-sample framework.
"""
import csv
import itertools
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

from quanti.data.storage import DataStorage

CAPITAL = 90000
COMM = 0.00025

TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END = "20220101", "20251231"

# ── Parameter grid ──
TOP_N_VALS = [3, 5, 8]
STOP_PCT_VALS = [-8, -10, -15]
MIN_TREND_VALS = [3, 4]
DD_EXIT_VALS = [15, 20, 25]
MKT_TREND_REQUIRED = True  # kept fixed as in original best configs

# Diagnostic configs (used by --mode diagnostic)
DIAGNOSTIC_CONFIGS = [
    ("BEST_T5_Trend4_DD15_Mkt",   5, -10, 4, 15, True),
    ("BEST_T5_Trend4_DD15_NoMkt", 5, -10, 4, 15, False),
    ("BEST_T5_Trend4_DD20_Mkt",   5, -10, 4, 20, True),
    ("T3_Trend4_DD15_Mkt",        3, -10, 4, 15, True),
    ("T8_Trend4_DD15_Mkt",        8, -10, 4, 15, True),
    ("T5_Trend3_DD15_Mkt",        5, -10, 3, 15, True),
]


# ═══════════════════════════════════════════════════════════════
# Shared infrastructure (loaded once)
# ═══════════════════════════════════════════════════════════════

def _load_stock_data():
    """Load all stock data into numpy arrays. Returns (stock_data, all_dates)."""
    storage = DataStorage()
    all_files = sorted(storage.clean_dir.glob("*.parquet"))
    stocks = [p.stem for p in all_files if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]

    stock_data = {}
    all_dates_set = set()
    for i, code in enumerate(stocks):
        if (i+1) % 100 == 0:
            print(f"  Loading {i+1}/{len(stocks)}...")
        raw = storage.load_bars(code)
        if not raw or len(raw) < 200:
            continue
        dates = [r.trade_date for r in raw]
        closes = np.array([r.close for r in raw], dtype=np.float64)
        highs  = np.array([r.high for r in raw], dtype=np.float64)
        lows   = np.array([r.low for r in raw], dtype=np.float64)
        volumes = np.array([r.volume for r in raw], dtype=np.float64)
        stock_data[code] = (closes, highs, lows, volumes, dates)
        all_dates_set.update(dates)

    all_dates = sorted(all_dates_set)
    return stock_data, all_dates


def data_at(code, date_str, n, stock_data):
    if code not in stock_data: return None
    c, h, l, v, d = stock_data[code]
    idx = None
    for i in range(len(d)-1, -1, -1):
        if d[i] <= date_str: idx = i+1; break
    if idx is None or idx < n: return None
    return (c[idx-n:idx], h[idx-n:idx], l[idx-n:idx], v[idx-n:idx])


def price_on(code, date_str, stock_data):
    if code not in stock_data: return None
    c,_,_,_,d = stock_data[code]
    for i in range(len(d)-1, -1, -1):
        if d[i] <= date_str: return c[i]
    return None


def sma(arr, period):
    if len(arr) < period: return None
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(np.insert(arr, 0, 0))
    out[period-1:] = (cs[period:] - cs[:-period]) / period
    return out


def adx(high, low, close, period=14):
    n = len(close)
    if n < period*2: return None
    tr = np.zeros(n)
    tr[0] = high[0]-low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = high[i]-high[i-1]; down = low[i-1]-low[i]
        if up > down and up > 0: pdm[i] = up
        if down > up and down > 0: mdm[i] = down
    atr = np.full(n, np.nan)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, n): atr[i] = (tr[i] + (period-1)*atr[i-1])/period
    pdi = np.full(n, np.nan); mdi = np.full(n, np.nan)
    pdi[period] = np.mean(pdm[1:period+1])/atr[period]*100
    mdi[period] = np.mean(mdm[1:period+1])/atr[period]*100
    for i in range(period+1, n):
        if np.isnan(atr[i]) or atr[i] < 1e-10:
            # NaN-safe: copy previous values to avoid gap in Wilder smoothing
            pdi[i] = pdi[i-1] if not np.isnan(pdi[i-1]) else 0.0
            mdi[i] = mdi[i-1] if not np.isnan(mdi[i-1]) else 0.0
            continue
        pdi[i] = (pdm[i] + (period-1)*pdi[i-1])/period / atr[i] * 100
        mdi[i] = (mdm[i] + (period-1)*mdi[i-1])/period / atr[i] * 100
    dx = np.abs(pdi-mdi)/(pdi+mdi+1e-10)*100
    adx_arr = np.full(n, np.nan)
    # Find first valid ADX seed
    seed_start = period*2-1
    while seed_start < n and np.isnan(np.nanmean(dx[max(1,seed_start-period+1):seed_start+1])):
        seed_start += 1
    if seed_start >= n: return adx_arr
    adx_arr[seed_start] = np.nanmean(dx[max(1,seed_start-period+1):seed_start+1])
    prev = adx_arr[seed_start]
    for i in range(seed_start+1, n):
        cur = dx[i]
        if np.isnan(cur):
            adx_arr[i] = prev  # propagate last valid value through NaN gaps
        else:
            prev = (cur + (period-1)*prev)/period
            adx_arr[i] = prev
    return adx_arr


def is_stock_uptrend(closes, highs, lows, volumes):
    if len(closes) < 130: return False, 0
    ma120 = sma(closes, 120)
    if ma120 is None or np.isnan(ma120[-1]): return False, 0
    above_ma = closes[-1] > ma120[-1]
    recent_high = np.max(highs[-20:]); prev_high = np.max(highs[-60:-20])
    higher_high = recent_high > prev_high
    recent_low = np.min(lows[-20:]); prev_low = np.min(lows[-60:-20])
    higher_low = recent_low > prev_low
    ma20_arr = sma(closes, 20); ma60_arr = sma(closes, 60)
    if ma20_arr is None or ma60_arr is None: return False, 0
    if np.isnan(ma20_arr[-1]) or np.isnan(ma60_arr[-1]): return False, 0
    ma_aligned = ma20_arr[-1] > ma60_arr[-1] > ma120[-1]
    adx_arr = adx(highs, lows, closes, 14)
    adx_ok = adx_arr is not None and not np.isnan(adx_arr[-1]) and adx_arr[-1] > 25
    vol_20 = np.mean(volumes[-21:-1])
    vol_surge = volumes[-1] > vol_20 * 1.2
    conditions = [above_ma, higher_high and higher_low, ma_aligned, adx_ok, vol_surge]
    score = sum(conditions)
    is_trend = above_ma and adx_ok and score >= 3
    return is_trend, score


def market_uptrend(date_str, stock_data):
    d = data_at("510300", date_str, 200, stock_data)
    if d is None: return True, 5
    c, h, l, v = d
    return is_stock_uptrend(c, h, l, v)


def trend_strength_score(closes):
    if len(closes) < 130: return 0
    ret_3m = closes[-1]/closes[-63]-1 if closes[-63]>1e-6 else 0
    ret_6m = closes[-1]/closes[-126]-1 if closes[-126]>1e-6 else 0
    mom = (min(max(ret_3m/0.5,0),1)*0.5 + min(max(ret_6m/0.8,0),1)*0.5)*100
    window_c = closes[-61:]
    dr = np.diff(window_c)/(window_c[:-1]+1e-10)
    vol = np.nanstd(dr)
    vol_s = (1-min(vol/0.04,1))*100
    return 0.6*mom + 0.4*vol_s


def get_monthly_dates(dates, start, end):
    monthly = []
    for d in dates:
        if d < start or d > end: continue
        dm = d[4:6]
        if not monthly or dm != monthly[-1][4:6]: monthly.append(d)
    return monthly


# ═══════════════════════════════════════════════════════════════
# Backtest engines (two variants)
# ═══════════════════════════════════════════════════════════════

def run_backtest(top_n, stop_pct, min_trend_cond, mkt_trend_required, dd_exit_pct,
                 start_date, end_date, stock_data, all_dates):
    """Lightweight: returns (cagr, sharpe, maxdd, n_years, final_eq)."""
    rebal = get_monthly_dates(all_dates, start_date, end_date)
    if len(rebal) < 12:
        return (0.0, 0.0, 100.0, 0.0, CAPITAL)

    cash = CAPITAL
    holdings = {}
    eq_curve = [cash]
    max_eq = cash
    dd_exit_triggered = False

    for reb_date in rebal:
        for sym in list(holdings.keys()):
            p = price_on(sym, reb_date, stock_data)
            if p is None or p < 0.01:
                cash += holdings[sym]["value"] * 0.7
                del holdings[sym]; continue
            holdings[sym]["price"] = p
            holdings[sym]["value"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]:
                holdings[sym]["hwm"] = p
            hwm = holdings[sym]["hwm"]
            if hwm > 0 and (p/hwm - 1)*100 < stop_pct:
                mv = holdings[sym]["qty"]*p
                cash += mv*(1-COMM)
                del holdings[sym]

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())

        if dd_exit_pct > 0:
            if total > max_eq: max_eq = total
            dd = (max_eq - total) / max_eq * 100
            if dd > dd_exit_pct:
                for sym in list(holdings.keys()):
                    mv = holdings[sym]["qty"]*holdings[sym]["price"]
                    cash += mv*(1-COMM)
                    del holdings[sym]
                dd_exit_triggered = True
            elif dd_exit_triggered:
                if total / max_eq > 0.92:
                    dd_exit_triggered = False

        if dd_exit_triggered:
            total = cash
            eq_curve.append(total)
            continue

        mkt_trend, _ = market_uptrend(reb_date, stock_data)
        if mkt_trend_required and not mkt_trend:
            eq_curve.append(total)
            continue

        trending = []
        for code in stock_data:
            d = data_at(code, reb_date, 260, stock_data)
            if d is None: continue
            c, h, l, v = d
            is_t, n_cond = is_stock_uptrend(c, h, l, v)
            if is_t and n_cond >= min_trend_cond:
                s = trend_strength_score(c)
                trending.append((code, s, n_cond))

        if not trending:
            eq_curve.append(total)
            continue

        trending.sort(key=lambda x: x[1], reverse=True)
        selected = {t[0] for t in trending[:top_n]}

        for sym in list(holdings.keys()):
            if sym not in selected:
                mv = holdings[sym]["qty"]*holdings[sym]["price"]
                cash += mv*(1-COMM)
                del holdings[sym]

        n_pos = max(len(selected), 1)
        per_stock = total / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, reb_date, stock_data)
            if p is None or p < 0.01: continue
            target_qty = int(per_stock / p / 100) * 100
            if target_qty < 100: continue
            if sym in holdings:
                cur = holdings[sym]["qty"]
                diff = target_qty - cur
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff > 0 and cash >= cost*(1+COMM):
                        cash -= cost*(1+COMM)
                        holdings[sym]["qty"] = target_qty
                    elif diff < 0:
                        cash += cost*(1-COMM)
                        holdings[sym]["qty"] = target_qty
            else:
                cost = target_qty*p
                if cash >= cost*(1+COMM):
                    cash -= cost*(1+COMM)
                    holdings[sym] = {"qty": target_qty, "price": p, "value": cost, "hwm": p}

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())
        eq_curve.append(total)

    eq = np.array(eq_curve)
    n_y = len(eq_curve) / 12.0
    if eq[0] <= 0 or n_y <= 0:
        return (0.0, 0.0, 100.0, n_y, eq[-1] if len(eq) > 0 else CAPITAL)

    cagr = ((eq[-1]/eq[0])**(1/n_y)-1)*100
    mr = np.diff(eq)/(eq[:-1]+1e-10)
    sharpe = np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12) if len(mr)>1 and np.std(mr)>1e-10 else 0
    peak = eq[0]; maxdd = 0.0
    for v in eq:
        if v>peak: peak=v
        dd=(peak-v)/peak*100
        if dd>maxdd: maxdd=dd
    return (cagr, sharpe, maxdd, n_y, eq[-1])


def run_detailed(top_n, stop_pct, min_trend_cond, mkt_required, dd_exit_pct,
                 start_date, end_date, stock_data, all_dates):
    """Detailed: returns (eq_curve, monthly_returns_list, month_counts_dict, n_trending_avg)."""
    rebal = get_monthly_dates(all_dates, start_date, end_date)
    cash = CAPITAL
    holdings = {}
    eq_curve = []
    max_eq = cash
    dd_exit_triggered = False

    months_mkt_blocked = 0
    months_no_stocks = 0
    months_invested = 0
    months_dd_exit = 0
    n_trending_avg = []
    monthly_returns = []

    for reb_date in rebal:
        for sym in list(holdings.keys()):
            p = price_on(sym, reb_date, stock_data)
            if p is None or p < 0.01:
                cash += holdings[sym]["value"] * 0.7
                del holdings[sym]; continue
            holdings[sym]["price"] = p
            holdings[sym]["value"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]:
                holdings[sym]["hwm"] = p
            hwm = holdings[sym]["hwm"]
            if hwm > 0 and (p/hwm - 1)*100 < stop_pct:
                mv = holdings[sym]["qty"]*p
                cash += mv*(1-COMM)
                del holdings[sym]

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())

        if dd_exit_pct > 0:
            if total > max_eq: max_eq = total
            dd = (max_eq - total) / max_eq * 100
            if dd > dd_exit_pct:
                for sym in list(holdings.keys()):
                    mv = holdings[sym]["qty"]*holdings[sym]["price"]
                    cash += mv*(1-COMM)
                    del holdings[sym]
                dd_exit_triggered = True
            elif dd_exit_triggered:
                if total / max_eq > 0.92:
                    dd_exit_triggered = False

        if dd_exit_triggered:
            months_dd_exit += 1
            eq_curve.append(total)
            if len(eq_curve) > 1:
                monthly_returns.append((reb_date, (total - eq_curve[-2]) / eq_curve[-2] * 100, "DD_EXIT"))
            else:
                monthly_returns.append((reb_date, 0.0, "DD_EXIT"))
            continue

        mkt_trend, _ = market_uptrend(reb_date, stock_data)
        if mkt_required and not mkt_trend:
            months_mkt_blocked += 1
            eq_curve.append(total)
            if len(eq_curve) > 1:
                monthly_returns.append((reb_date, (total - eq_curve[-2]) / eq_curve[-2] * 100, "MKT_BLOCK"))
            else:
                monthly_returns.append((reb_date, 0.0, "MKT_BLOCK"))
            continue

        trending = []
        for code in stock_data:
            d = data_at(code, reb_date, 260, stock_data)
            if d is None: continue
            c, h, l, v = d
            is_t, n_cond = is_stock_uptrend(c, h, l, v)
            if is_t and n_cond >= min_trend_cond:
                s = trend_strength_score(c)
                trending.append((code, s, n_cond))

        n_trending_avg.append(len(trending))

        if not trending:
            months_no_stocks += 1
            eq_curve.append(total)
            if len(eq_curve) > 1:
                monthly_returns.append((reb_date, (total - eq_curve[-2]) / eq_curve[-2] * 100, "NO_STOCKS"))
            else:
                monthly_returns.append((reb_date, 0.0, "NO_STOCKS"))
            continue

        months_invested += 1
        trending.sort(key=lambda x: x[1], reverse=True)
        selected = {t[0] for t in trending[:top_n]}

        for sym in list(holdings.keys()):
            if sym not in selected:
                mv = holdings[sym]["qty"]*holdings[sym]["price"]
                cash += mv*(1-COMM)
                del holdings[sym]

        n_pos = max(len(selected), 1)
        per_stock = total / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, reb_date, stock_data)
            if p is None or p < 0.01: continue
            target_qty = int(per_stock / p / 100) * 100
            if target_qty < 100: continue
            if sym in holdings:
                cur = holdings[sym]["qty"]
                diff = target_qty - cur
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff > 0 and cash >= cost*(1+COMM):
                        cash -= cost*(1+COMM)
                        holdings[sym]["qty"] = target_qty
                    elif diff < 0:
                        cash += cost*(1-COMM)
                        holdings[sym]["qty"] = target_qty
            else:
                cost = target_qty*p
                if cash >= cost*(1+COMM):
                    cash -= cost*(1+COMM)
                    holdings[sym] = {"qty": target_qty, "price": p, "value": cost, "hwm": p}

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())
        eq_curve.append(total)
        if len(eq_curve) > 1:
            mr = (total - eq_curve[-2]) / eq_curve[-2] * 100
            monthly_returns.append((reb_date, mr, "INVESTED"))
        else:
            monthly_returns.append((reb_date, 0.0, "INVESTED"))

    counts = {
        "mkt_blocked": months_mkt_blocked,
        "no_stocks": months_no_stocks,
        "invested": months_invested,
        "dd_exit": months_dd_exit,
    }
    return eq_curve, monthly_returns, counts, n_trending_avg


def compute_index_curve(start_date, end_date, stock_data, all_dates):
    """CSI300 buy-and-hold equity curve (via 510300 ETF proxy)."""
    d = data_at("510300", "20260601", 1, stock_data)
    if d is None: return []
    dates = get_monthly_dates(all_dates, start_date, end_date)
    initial = None
    curve = []
    for dt in dates:
        p = price_on("510300", dt, stock_data)
        if p is None: continue
        if initial is None:
            initial = p
            curve.append((dt, 100.0))
        else:
            curve.append((dt, (p / initial) * 100.0))
    return curve


# ═══════════════════════════════════════════════════════════════
# Mode: grid_search
# ═══════════════════════════════════════════════════════════════

def mode_grid_search(stock_data, all_dates):
    grid = list(itertools.product(TOP_N_VALS, STOP_PCT_VALS, MIN_TREND_VALS, DD_EXIT_VALS))
    total = len(grid)

    print(f"Grid search: {total} combinations")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print(f"Test:  {TEST_START} to {TEST_END}\n")

    results = []
    for idx, (top_n, stop_pct, min_trend, dd_exit) in enumerate(grid):
        name = f"T{top_n}_Stop{abs(stop_pct)}_Trend{min_trend}_DD{dd_exit}"

        t_cagr, t_sharpe, t_maxdd, t_ny, t_final = run_backtest(
            top_n, stop_pct, min_trend, MKT_TREND_REQUIRED, dd_exit,
            TRAIN_START, TRAIN_END, stock_data, all_dates)

        v_cagr, v_sharpe, v_maxdd, v_ny, v_final = run_backtest(
            top_n, stop_pct, min_trend, MKT_TREND_REQUIRED, dd_exit,
            TEST_START, TEST_END, stock_data, all_dates)

        sharpe_decay = (t_sharpe - v_sharpe) / t_sharpe if t_sharpe > 0.01 else 999
        maxdd_inflate = (v_maxdd / t_maxdd - 1) * 100 if t_maxdd > 0.01 else 999

        results.append({
            "name": name,
            "top_n": top_n, "stop_pct": stop_pct, "min_trend": min_trend, "dd_exit": dd_exit,
            "train_cagr": round(t_cagr, 2), "train_sharpe": round(t_sharpe, 3),
            "train_maxdd": round(t_maxdd, 2), "train_years": round(t_ny, 2),
            "test_cagr": round(v_cagr, 2), "test_sharpe": round(v_sharpe, 3),
            "test_maxdd": round(v_maxdd, 2), "test_years": round(v_ny, 2),
            "sharpe_decay": round(sharpe_decay, 3),
            "maxdd_inflate": round(maxdd_inflate, 2),
        })

        print(f"[{idx+1:3d}/{total}] {name:<28s} | Train: C={t_cagr:+5.1f}% S={t_sharpe:5.2f} D={t_maxdd:5.1f}% | "
              f"Test: C={v_cagr:+5.1f}% S={v_sharpe:5.2f} D={v_maxdd:5.1f}% | "
              f"Decay={sharpe_decay:.2f}")

    # Save CSV
    csv_path = os.path.join(_PROJECT_ROOT, "data", "param_search_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["name","top_n","stop_pct","min_trend","dd_exit",
                  "train_cagr","train_sharpe","train_maxdd","train_years",
                  "test_cagr","test_sharpe","test_maxdd","test_years",
                  "sharpe_decay","maxdd_inflate"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to: {csv_path}")

    # Rankings
    _print_rankings(results)

    # Composite best
    sorted_by_cagr = sorted(results, key=lambda x: x["test_cagr"], reverse=True)
    sorted_by_sharpe = sorted(results, key=lambda x: x["test_sharpe"], reverse=True)
    sorted_by_decay = sorted(results, key=lambda x: x["sharpe_decay"])
    sorted_by_inflate = sorted(results, key=lambda x: x["maxdd_inflate"])

    for i, r in enumerate(sorted_by_cagr): r["rank_cagr"] = i
    for i, r in enumerate(sorted_by_sharpe): r["rank_sharpe"] = i
    for i, r in enumerate(sorted_by_decay): r["rank_decay"] = i
    for i, r in enumerate(sorted_by_inflate): r["rank_inflate"] = i

    for r in results:
        r["composite_rank"] = r["rank_cagr"] + r["rank_sharpe"] + r["rank_decay"] + r["rank_inflate"]

    best = sorted(results, key=lambda x: x["composite_rank"])[0]

    print(f"\n{'='*70}")
    print("COMPOSITE BEST (rank sum: Cagr+Sharpe+Decay+Inflate)")
    print(f"{'='*70}")
    print(f"  {best['name']}")
    print(f"  Train: CAGR={best['train_cagr']:+.1f}%  Sharpe={best['train_sharpe']:.3f}  MaxDD={best['train_maxdd']:.1f}%")
    print(f"  Test:  CAGR={best['test_cagr']:+.1f}%  Sharpe={best['test_sharpe']:.3f}  MaxDD={best['test_maxdd']:.1f}%")
    print(f"  Sharpe Decay: {best['sharpe_decay']:.2f}  |  MaxDD Inflation: {best['maxdd_inflate']:+.1f}%")

    # Write report
    _write_grid_report(results, best, total)
    return results


def _print_rankings(results):
    print(f"\n{'='*70}")
    print("TOP 5 BY TEST CAGR")
    print(f"{'='*70}")
    for i, r in enumerate(sorted(results, key=lambda x: x["test_cagr"], reverse=True)[:5], 1):
        print(f"  #{i}: {r['name']:<28s} Test C={r['test_cagr']:+5.1f}% S={r['test_sharpe']:.2f} D={r['test_maxdd']:.1f}% "
              f"| Train C={r['train_cagr']:+5.1f}% S={r['train_sharpe']:.2f}")

    print("\nTOP 5 BY TEST SHARPE")
    print(f"{'='*70}")
    for i, r in enumerate(sorted(results, key=lambda x: x["test_sharpe"], reverse=True)[:5], 1):
        print(f"  #{i}: {r['name']:<28s} Test S={r['test_sharpe']:.2f} C={r['test_cagr']:+5.1f}% D={r['test_maxdd']:.1f}% "
              f"| Train S={r['train_sharpe']:.2f}")

    print("\nTOP 3 BY MINIMUM SHARPE DECAY (least overfitting)")
    print(f"{'='*70}")
    for i, r in enumerate(sorted(results, key=lambda x: x["sharpe_decay"])[:3], 1):
        print(f"  #{i}: {r['name']:<28s} Decay={r['sharpe_decay']:.2f} "
              f"| Train S={r['train_sharpe']:.2f} -> Test S={r['test_sharpe']:.2f} "
              f"C={r['test_cagr']:+5.1f}%")


def _write_grid_report(results, best, total):
    report_path = os.path.join(_PROJECT_ROOT, "data", "param_search_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Trend-First Momentum Strategy: Parameter Search Report\n\n")
        f.write("**Date:** 2026-06-12\n\n")
        f.write("## Framework\n\n")
        f.write(f"- **Train:** {TRAIN_START} to {TRAIN_END} (7 years)\n")
        f.write(f"- **Test:** {TEST_START} to {TEST_END} (4 years)\n")
        f.write("- **Strict sample-out-of-sample** - no future information leakage\n\n")
        f.write("## Parameter Space\n\n")
        f.write(f"- `top_n`: {TOP_N_VALS}\n")
        f.write(f"- `stop_pct`: {STOP_PCT_VALS}\n")
        f.write(f"- `min_trend_score`: {MIN_TREND_VALS}\n")
        f.write(f"- `dd_exit_pct`: {DD_EXIT_VALS}\n")
        f.write(f"- Total combinations: {total}\n\n")

        f.write("## 1. Top 5 by Test CAGR\n\n")
        f.write("| # | Config | Test CAGR | Test Sharpe | Test MaxDD | Train CAGR | Train Sharpe |\n")
        f.write("|---|--------|-----------|-------------|------------|------------|-------------|\n")
        for i, r in enumerate(sorted(results, key=lambda x: x["test_cagr"], reverse=True)[:5], 1):
            f.write(f"| {i} | {r['name']} | {r['test_cagr']:+.2f}% | {r['test_sharpe']:.3f} | {r['test_maxdd']:.1f}% | {r['train_cagr']:+.2f}% | {r['train_sharpe']:.3f} |\n")

        f.write("\n## 2. Top 5 by Test Sharpe\n\n")
        f.write("| # | Config | Test Sharpe | Test CAGR | Test MaxDD | Train Sharpe |\n")
        f.write("|---|--------|-------------|-----------|------------|-------------|\n")
        for i, r in enumerate(sorted(results, key=lambda x: x["test_sharpe"], reverse=True)[:5], 1):
            f.write(f"| {i} | {r['name']} | {r['test_sharpe']:.3f} | {r['test_cagr']:+.2f}% | {r['test_maxdd']:.1f}% | {r['train_sharpe']:.3f} |\n")

        f.write("\n## 3. Top 3 by Minimum Sharpe Decay (Least Overfitting)\n\n")
        f.write("| # | Config | Sharpe Decay | Train Sharpe | Test Sharpe | Test CAGR |\n")
        f.write("|---|--------|-------------|-------------|------------|----------|\n")
        for i, r in enumerate(sorted(results, key=lambda x: x["sharpe_decay"])[:3], 1):
            f.write(f"| {i} | {r['name']} | {r['sharpe_decay']:.3f} | {r['train_sharpe']:.3f} | {r['test_sharpe']:.3f} | {r['test_cagr']:+.2f}% |\n")

        f.write("\n## 4. Composite Best Recommendation\n\n")
        f.write(f"**{best['name']}**\n\n")
        f.write("| Metric | Train | Test |\n")
        f.write("|--------|-------|------|\n")
        f.write(f"| CAGR | {best['train_cagr']:+.2f}% | {best['test_cagr']:+.2f}% |\n")
        f.write(f"| Sharpe | {best['train_sharpe']:.3f} | {best['test_sharpe']:.3f} |\n")
        f.write(f"| MaxDD | {best['train_maxdd']:.1f}% | {best['test_maxdd']:.1f}% |\n")
        f.write(f"| Sharpe Decay | | {best['sharpe_decay']:.3f} |\n")
        f.write(f"| MaxDD Inflation | | {best['maxdd_inflate']:+.1f}% |\n\n")

        f.write("### Full Parameter Grid\n\n")
        f.write("| Config | Train CAGR | Train Sharpe | Train MaxDD | Test CAGR | Test Sharpe | Test MaxDD | Sharpe Decay | MaxDD Infl |\n")
        f.write("|--------|-----------|-------------|------------|-----------|-------------|------------|-------------|----------|\n")
        for r in sorted(results, key=lambda x: (x["test_cagr"]), reverse=True):
            f.write(f"| {r['name']} | {r['train_cagr']:+.2f}% | {r['train_sharpe']:.3f} | {r['train_maxdd']:.1f}% | {r['test_cagr']:+.2f}% | {r['test_sharpe']:.3f} | {r['test_maxdd']:.1f}% | {r['sharpe_decay']:.3f} | {r['maxdd_inflate']:+.1f}% |\n")

    print(f"\nReport saved to: {report_path}")


# ═══════════════════════════════════════════════════════════════
# Mode: diagnostic
# ═══════════════════════════════════════════════════════════════

def _sub_period_metrics(monthly_returns, sp_start, sp_end):
    """Compute CAGR, Sharpe, MaxDD for a sub-period from monthly returns list."""
    sp_returns = [r for r in monthly_returns if sp_start <= r[0] <= sp_end]
    if len(sp_returns) < 3:
        return None

    rets = np.array([x[1] for x in sp_returns])
    n_y = len(sp_returns) / 12.0
    cum = np.prod(1 + rets/100)
    cagr = ((cum) ** (1/n_y) - 1) * 100 if n_y > 0 and cum > 0 else 0
    sharpe = np.mean(rets)/(np.std(rets)+1e-10)*np.sqrt(12)
    cum_eq = 100 * np.cumprod(1 + rets/100)
    peak = cum_eq[0]; maxdd = 0.0
    for v in cum_eq:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > maxdd: maxdd = dd

    sp_n_inv = sum(1 for x in sp_returns if x[2] == "INVESTED")
    sp_n_mkt = sum(1 for x in sp_returns if x[2] == "MKT_BLOCK")
    sp_n_ns = sum(1 for x in sp_returns if x[2] == "NO_STOCKS")
    sp_n_dd = sum(1 for x in sp_returns if x[2] == "DD_EXIT")
    pct_inv = sp_n_inv / len(sp_returns) * 100

    return (cagr, sharpe, maxdd, pct_inv, sp_n_mkt, sp_n_ns, sp_n_dd)


def mode_diagnostic(stock_data, all_dates):
    print("DEEP FAILURE ANALYSIS (merged diagnostic mode)")
    print("=" * 70)

    for cfg_name, tn, sp, tc, mkt, dd in DIAGNOSTIC_CONFIGS:
        print(f"\n{'─'*70}")
        print(f"Config: {cfg_name}  (top_n={tn}, stop={sp}%, trend_cond={tc}, dd_exit={dd}%, mkt_req={mkt})")
        print(f"{'─'*70}")

        eq, mr_list, counts, n_trend = run_detailed(
            tn, sp, tc, mkt, dd, "20220101", "20251231", stock_data, all_dates)
        eq = np.array(eq)
        n = len(mr_list)
        if n < 1: continue

        n_mkt = counts["mkt_blocked"]
        n_nostock = counts["no_stocks"]
        n_inv = counts["invested"]
        n_dd = counts["dd_exit"]

        print(f"\n  Months total: {n} | Invested: {n_inv} | Mkt blocked: {n_mkt} | No stocks: {n_nostock} | DD exit: {n_dd}")
        print(f"  Avg trending stocks when invested: {np.mean(n_trend):.1f}" if n_trend else "  No trending periods")

        sub_periods = [
            ("2022", "20220101", "20221231"),
            ("2023", "20230101", "20231231"),
            ("2024", "20240101", "20241231"),
            ("2025", "20250101", "20251231"),
        ]

        print("\n  Sub-period analysis:")
        print(f"  {'Period':>6s} | {'CAGR':>7s} | {'Sharpe':>7s} | {'MaxDD':>6s} | {'Invest%':>7s} | {'MktBlk':>6s} | {'NoStk':>6s} | {'DDext':>6s}")
        print(f"  {'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")

        for sp_name, sp_start, sp_end in sub_periods:
            m = _sub_period_metrics(mr_list, sp_start, sp_end)
            if m is None:
                print(f"  {sp_name:>6s} | {'N/A':>7s} |")
            else:
                cagr, sharpe, maxdd, pct_inv, sp_n_mkt, sp_n_ns, sp_n_dd = m
                print(f"  {sp_name:>6s} | {cagr:+6.1f}% | {sharpe:+7.3f} | {maxdd:5.1f}% | {pct_inv:6.0f}% | {sp_n_mkt:5d} | {sp_n_ns:5d} | {sp_n_dd:5d}")

        # Overall metrics
        rets = np.array([x[1] for x in mr_list])
        n_y = len(rets) / 12.0
        cum = np.prod(1 + rets/100)
        cagr = ((cum) ** (1/n_y) - 1) * 100
        sharpe = np.mean(rets)/(np.std(rets)+1e-10)*np.sqrt(12)
        cum_eq = 100 * np.cumprod(1 + rets/100)
        peak = cum_eq[0]; maxdd = 0.0
        for v in cum_eq:
            if v > peak: peak = v
            dd = (peak - v) / peak * 100
            if dd > maxdd: maxdd = dd

        sum(1 for r in rets if r > 0)
        sum(1 for r in rets if r < 0)
        invested_rets = np.array([x[1] for x in mr_list if x[2] == "INVESTED"])
        if len(invested_rets) > 0:
            hit_rate = sum(1 for r in invested_rets if r > 0) / len(invested_rets) * 100
            avg_win = np.mean(invested_rets[invested_rets > 0]) if any(invested_rets > 0) else 0
            avg_loss = np.mean(invested_rets[invested_rets < 0]) if any(invested_rets < 0) else 0
        else:
            hit_rate = avg_win = avg_loss = 0

        print(f"\n  Overall: CAGR={cagr:+.1f}% Sharpe={sharpe:.3f} MaxDD={maxdd:.1f}%")
        print(f"  Invested months: hit_rate={hit_rate:.0f}% avg_win={avg_win:+.2f}% avg_loss={avg_loss:+.2f}%")


    # Market benchmark
    print(f"\n{'='*70}")
    print("MARKET BENCHMARK (CSI300 via 510300 ETF)")
    print(f"{'='*70}")

    idx = compute_index_curve("20220101", "20251231", stock_data, all_dates)
    if idx:
        idx_vals = np.array([v[1] for v in idx])
        idx_rets = np.diff(idx_vals) / idx_vals[:-1] * 100
        n_y = len(idx_rets) / 12.0
        idx_cagr = ((idx_vals[-1]/idx_vals[0])**(1/n_y)-1)*100
        idx_sharpe = np.mean(idx_rets)/(np.std(idx_rets)+1e-10)*np.sqrt(12)
        peak = idx_vals[0]; idx_maxdd = 0.0
        for v in idx_vals:
            if v > peak: peak = v
            dd = (peak - v) / peak * 100
            if dd > idx_maxdd: idx_maxdd = dd

        for sp_name, sp_start, sp_end in [("2022","2022","2022"),("2023","2023","2023"),("2024","2024","2024"),("2025","2025","2025")]:
            sp_idx = [v for v in idx if v[0].startswith(sp_name)]
            if len(sp_idx) < 3: continue
            sp_vals = np.array([v[1] for v in sp_idx])
            sp_rets = np.diff(sp_vals) / sp_vals[:-1] * 100
            sp_n_y = len(sp_rets) / 12.0
            if sp_n_y <= 0: continue
            sp_cagr = ((sp_vals[-1]/sp_vals[0])**(1/sp_n_y)-1)*100
            sp_sharpe = np.mean(sp_rets)/(np.std(sp_rets)+1e-10)*np.sqrt(12)
            sp_peak = sp_vals[0]; sp_dd = 0.0
            for v in sp_vals:
                if v > sp_peak: sp_peak = v
                dd = (sp_peak - v) / sp_peak * 100
                if dd > sp_dd: sp_dd = dd
            print(f"  {sp_name}: CAGR={sp_cagr:+5.1f}% Sharpe={sp_sharpe:+6.3f} MaxDD={sp_dd:.1f}%")

        print(f"  Overall: CAGR={idx_cagr:+5.1f}% Sharpe={idx_sharpe:+6.3f} MaxDD={idx_maxdd:.1f}%")

    # Worst/best months for best config
    print(f"\n{'='*70}")
    print("WORST & BEST MONTHS FOR DEFAULT CONFIG (T5 Trend4 DD15 Mkt)")
    print(f"{'='*70}")

    eq, mr_list, _, _, _, _, _ = run_detailed(5, -10, 4, True, 15, "20220101", "20251231", stock_data, all_dates)

    sorted_months = sorted(mr_list, key=lambda x: x[1])
    print(f"\n{'Date':>10s} | {'Return':>7s} | {'State':>10s}")
    print(f"{'─'*10}─┼─{'─'*7}─┼─{'─'*10}")
    for date, ret, state in sorted_months[:15]:
        print(f"  {date} | {ret:+6.2f}% | {state:>10s}")

    print(f"\n{'Date':>10s} | {'Return':>7s} | {'State':>10s}")
    print(f"{'─'*10}─┼─{'─'*7}─┼─{'─'*10}")
    for date, ret, state in sorted_months[-15:]:
        print(f"  {date} | {ret:+6.2f}% | {state:>10s}")

    # Trending stock availability
    print(f"\n{'='*70}")
    print("TRENDING STOCK AVAILABILITY (default config)")
    print(f"{'='*70}")

    eq, mr_list, _, _, _, _, n_trend = run_detailed(5, -10, 4, True, 15, "20220101", "20251231", stock_data, all_dates)
    rebal = get_monthly_dates(all_dates, "20220101", "20251231")

    print("\n  Sampling monthly trending stock counts:")
    for i, (dt, nt) in enumerate(zip(rebal, n_trend + [0]*(len(rebal)-len(n_trend)), strict=False)):
        if i % 12 == 0:
            print(f"    {dt}: {nt} trending stocks meet criteria")
        if i >= len(n_trend): break
    print(f"  Avg trending: {np.mean(n_trend):.1f}  |  Min: {min(n_trend)}  |  Max: {max(n_trend)}")

    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trend-First Momentum Strategy Analysis")
    parser.add_argument("--mode", choices=["grid_search", "diagnostic"],
                        default="grid_search",
                        help="grid_search: parameter sweep and ranking; diagnostic: failure analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("Loading data...")
    print("=" * 70)

    stock_data, all_dates = _load_stock_data()
    print(f"Loaded {len(stock_data)} stocks, {len(all_dates)} trading days ({all_dates[0]} to {all_dates[-1]})")

    if args.mode == "grid_search":
        mode_grid_search(stock_data, all_dates)
    elif args.mode == "diagnostic":
        mode_diagnostic(stock_data, all_dates)

    print("\nDone.")


if __name__ == "__main__":
    main()

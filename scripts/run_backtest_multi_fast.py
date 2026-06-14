"""Optimized multi-industry ETF rotation backtest.
Precomputes scores on monthly dates (not daily event loop).
Matches run_backtest.py architecture for fast 10-year backtests.
"""
import os, sys, time, json
_PROJECT_ROOT = r"C:\study\AIWorkspace\quanti"
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
from datetime import datetime, timedelta
from quanti.data.storage import DataStorage

CAPITAL = 90000
COMM = 0.00025
TOP_N = 3
MAX_PER_CAT = 2
STOP_PCT = -10
DD_EXIT_PCT = 15
START, END = "20150101", "20251231"

# Load audit data
with open(os.path.join(_PROJECT_ROOT, "data", "etf_listing_audit.json"), encoding="utf-8") as f:
    AUDIT = json.load(f)

CATEGORY_MAP = {}
for cat, syms in AUDIT["categories"].items():
    for s in syms:
        if AUDIT["etfs"].get(s, {}).get("status") == "ok":
            CATEGORY_MAP[s] = cat


def sma(arr, p):
    if len(arr) < p:
        return None
    o = np.full(len(arr), np.nan)
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p - 1:] = (cs[p:] - cs[:-p]) / p
    return o


def compute_adx(hi, lo, cl, period=14):
    n = len(cl)
    tr = np.zeros(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1]))
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    for i in range(1, n):
        up = hi[i] - hi[i - 1]
        dn = lo[i - 1] - lo[i]
        if up > dn and up > 0:
            pdm[i] = up
        if dn > up and dn > 0:
            mdm[i] = dn
    atr = float(np.mean(tr[1:period + 1])) if tr[1:period + 1].any() else 0.001
    ps = float(np.mean(pdm[1:period + 1]))
    ms = float(np.mean(mdm[1:period + 1]))
    for i in range(period + 1, n):
        atr = (tr[i] + (period - 1) * atr) / period
        ps = (pdm[i] + (period - 1) * ps) / period
        ms = (mdm[i] + (period - 1) * ms) / period
    denom = max(atr, 0.001)
    pdi = min(ps / denom * 100, 1000)
    mdi = min(ms / denom * 100, 1000)
    return float(np.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100)


def compute_score(cl, hi, lo):
    m120 = sma(cl, 120)
    trend = 1.0 if (m120 is not None and not np.isnan(m120[-1]) and cl[-1] > m120[-1]) else 0.0
    if len(cl) >= 28:
        adx_v = compute_adx(hi, lo, cl)
        adx_val = min(adx_v / 50.0, 1.0) if not np.isnan(adx_v) else 0.5
    else:
        adx_val = 0.5
    if cl[-21] > 1e-6:
        ret = (cl[-1] / cl[-21] - 1) * 100
        mom = min(max(ret / 15.0, 0), 1) if ret > 0 else 0
    else:
        mom = 0.5
    return 0.35 * trend + 0.40 * adx_val + 0.25 * mom


def ma_rising(cl):
    if len(cl) < 140:
        return False
    return np.mean(cl[-120:]) > np.mean(cl[-140:-20])


def monthly_dates(dates):
    m = []
    for d in dates:
        dm = d[4:6]
        if not m or dm != m[-1][4:6]:
            m.append(d)
    return m


def price_on(code, dt, sdata):
    c, d = sdata[code][0], sdata[code][4]
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= dt:
            return c[i]
    return None


def data_at(code, dt, n, sdata):
    if code not in sdata:
        return None
    c, h, l, _, d = sdata[code]
    idx = None
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= dt:
            idx = i + 1
            break
    if idx is None or idx < n:
        return None
    return (c[idx - n:idx], h[idx - n:idx], l[idx - n:idx])


def eligible_etfs(date_str):
    target = datetime.strptime(date_str, "%Y%m%d")
    threshold = target - timedelta(days=252)
    result = []
    for s in CATEGORY_MAP:
        fd_str = AUDIT["etfs"][s]["first_date"]
        if datetime.strptime(fd_str, "%Y-%m-%d") <= threshold:
            result.append(s)
    return result


def apply_category_cap(scored, max_cat=2, top_n=3):
    selected = []
    counts = {}
    for sym, score in scored:
        if len(selected) >= top_n:
            break
        cat = CATEGORY_MAP.get(sym, "?")
        n = counts.get(cat, 0)
        if n >= max_cat:
            continue
        counts[cat] = n + 1
        selected.append(sym)
    return selected


def calculate_metrics(eq_curve, n_rebal):
    eq = np.array(eq_curve)
    cagr = ((eq[-1] / eq[0]) ** (1.0 / (n_rebal / 12.0)) - 1) * 100 if eq[0] > 0 else 0
    mr = np.diff(eq) / (eq[:-1] + 1e-10)
    sharpe = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 else 0
    peak = eq[0]
    mdd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        d = (peak - v) / peak * 100
        if d > mdd:
            mdd = d
    calmar = cagr / mdd if mdd > 0 else 0
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": mdd, "calmar": calmar, "final": float(eq[-1])}


def run_backtest(progressive=True, symbols_override=None):
    storage = DataStorage()
    syms = symbols_override or list(CATEGORY_MAP.keys())

    stock_data = {}
    all_ds = set()
    for code in syms:
        rs = storage.load_bars(code)
        if not rs or len(rs) < 200:
            continue
        d = [r.trade_date for r in rs]
        stock_data[code] = (
            np.array([r.close for r in rs], dtype=np.float64),
            np.array([r.high for r in rs], dtype=np.float64),
            np.array([r.low for r in rs], dtype=np.float64),
            np.array([r.volume for r in rs], dtype=np.float64),
            d,
        )
        all_ds.update(d)

    all_dates = sorted(all_ds)
    rebal = [r for r in monthly_dates(all_dates) if START <= r <= END]

    cash = CAPITAL
    holdings = {}
    eq = [cash]
    max_eq = cash
    dd_active = False

    for ri, rd in enumerate(rebal):
        if ri % 12 == 0:
            print(f"  {rd[:4]}/{rd[4:6]} ...")

        elig = eligible_etfs(rd) if progressive else list(stock_data.keys())
        scored = []
        for code in elig:
            d2 = data_at(code, rd, 260, stock_data)
            if d2 is None:
                continue
            cl, hi, lo = d2
            if not ma_rising(cl):
                continue
            s = compute_score(cl, hi, lo)
            if s > 0:
                scored.append((code, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = set(apply_category_cap(scored, MAX_PER_CAT, TOP_N))

        # Update positions
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01:
                cash += holdings[sym]["val"] * 0.7
                del holdings[sym]
                continue
            holdings[sym]["price"] = p
            holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]:
                holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p / holdings[sym]["hwm"] - 1) * 100 < STOP_PCT:
                cash += holdings[sym]["val"] * (1 - COMM)
                del holdings[sym]

        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())
        if total > max_eq:
            max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0

        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1 - COMM)
                del holdings[sym]
            dd_active = True
        elif dd_active and total / max_eq > 0.92:
            dd_active = False

        if dd_active:
            eq.append(cash)
            continue

        # Sell rotated-out
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1 - COMM)
                del holdings[sym]

        # Buy new
        n_pos = max(len(selected), 1)
        per = total / n_pos * 0.92
        for sym in selected:
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01:
                continue
            tq = int(per / p / 100) * 100
            if tq < 100:
                continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1 + COMM):
                        cash -= cost * (1 + COMM)
                        holdings[sym]["qty"] = tq
                    elif diff < 0:
                        cash += cost * (1 - COMM)
                        holdings[sym]["qty"] = tq
            else:
                cost = tq * p
                if cash >= cost * (1 + COMM):
                    cash -= cost * (1 + COMM)
                    holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}

        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())
        eq.append(total)

    return {**calculate_metrics(eq, len(rebal)), "months": len(rebal), "eq": eq}


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("MULTI-INDUSTRY ETF ROTATION BACKTEST (2015-2025)")
    print("=" * 70)
    print(f"ETF Pool: {len(CATEGORY_MAP)} symbols, {len(set(CATEGORY_MAP.values()))} categories")
    print(f"Top N={TOP_N}, Max per category={MAX_PER_CAT}")
    print()

    print("Running PROGRESSIVE enrollment backtest ...")
    t1 = time.time()
    r_prog = run_backtest(progressive=True)
    dt1 = time.time() - t1
    print(f"  Done in {dt1:.1f}s")
    print(f"  CAGR: {r_prog['cagr']:+.2f}%  Sharpe: {r_prog['sharpe']:.3f}  MaxDD: {r_prog['maxdd']:.2f}%")

    print()
    print("Running STATIC all-in backtest (look-ahead bias test) ...")
    t2 = time.time()
    r_static = run_backtest(progressive=False)
    dt2 = time.time() - t2
    print(f"  Done in {dt2:.1f}s")
    print(f"  CAGR: {r_static['cagr']:+.2f}%  Sharpe: {r_static['sharpe']:.3f}  MaxDD: {r_static['maxdd']:.2f}%")

    print()
    print("Running 6-ETF BASELINE backtest ...")
    baseline_syms = ["510300", "510500", "159915", "510880", "518880", "511880"]
    # Temporarily override CATEGORY_MAP for baseline (all same category = no cap)
    backup_cat = dict(CATEGORY_MAP)
    CATEGORY_MAP.clear()
    for s in baseline_syms:
        CATEGORY_MAP[s] = "all"
    t3 = time.time()
    r_baseline = run_backtest(progressive=False, symbols_override=baseline_syms)
    dt3 = time.time() - t3
    # Restore
    CATEGORY_MAP.clear()
    CATEGORY_MAP.update(backup_cat)
    print(f"  Done in {dt3:.1f}s")
    print(f"  CAGR: {r_baseline['cagr']:+.2f}%  Sharpe: {r_baseline['sharpe']:.3f}  MaxDD: {r_baseline['maxdd']:.2f}%")

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Metric':<20s} {'Progressive(25ETF)':>18s} {'Static(25ETF)':>18s} {'Baseline(6ETF)':>18s}")
    print("-" * 75)
    for label, key, fmt in [
        ("CAGR %", "cagr", ".2f"),
        ("Sharpe", "sharpe", ".3f"),
        ("MaxDD %", "maxdd", ".2f"),
        ("Calmar", "calmar", ".3f"),
        ("Final Value CNY", "final", ".0f"),
        ("Rebal Months", "months", "d"),
    ]:
        if key == "months":
            print(f"{label:<20s} {r_prog[key]:>18d} {r_static[key]:>18d} {r_baseline[key]:>18d}")
        else:
            vp, vs, vb = r_prog[key], r_static[key], r_baseline[key]
            print(f"{label:<20s} {vp:>18{fmt}} {vs:>18{fmt}} {vb:>18{fmt}}")

    bias = r_static["cagr"] - r_prog["cagr"]
    delta = r_prog["cagr"] - r_baseline["cagr"]
    print()
    print(f"Look-ahead bias (static - progressive): {bias:+.2f}% CAGR")
    print(f"Multi-industry vs Baseline delta:      {delta:+.2f}% CAGR")
    print(f"Total runtime: {time.time() - t0:.1f}s")

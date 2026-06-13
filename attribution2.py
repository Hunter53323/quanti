"""
Proper attribution: separate stock selection from market timing.
Three comparators on the same test period:
  1. CSI300 B&H                 = market benchmark
  2. Always-invested (raw mom)   = truly always-invested, top 5 by raw 3m+6m momentum, NO trend filter
  3. Always-invested (trend-filt)= current strategy's stock pool (5-condition filter), always invested when stocks exist
  4. BOND_ROTATE entry-only      = trend-filtered stocks + 120MA/60MA market timing
  5. BOND_ROTATE + A43           = + decay

The delta between (2) and (1) is PURE stock selection alpha (picking stocks vs buying CSI300).
The delta between (3) and (2) is the stock quality filter effect.
The delta between (4) and (3) is PURE market timing alpha (state machine + bond rotation).
"""
import os, sys, time
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025
TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END = "20220101", "20251231"

def monthly_dates(dates, s, e):
    m = []
    for d in dates:
        if d < s or d > e: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

def sma(arr, p):
    if len(arr) < p: return None
    o = np.full(len(arr), np.nan); cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p; return o

def data_at(code, dt, n, sdata):
    if code not in sdata: return None
    c, h, l, v, d = sdata[code]; idx = None
    for i in range(len(d)-1, -1, -1):
        if d[i] <= dt: idx = i+1; break
    if idx is None or idx < n: return None
    return (c[idx-n:idx], h[idx-n:idx], l[idx-n:idx], v[idx-n:idx])

def price_on(code, dt, sdata):
    if code not in sdata: return None
    c, d = sdata[code][0], sdata[code][4]
    for i in range(len(d)-1, -1, -1):
        if d[i] <= dt: return c[i]
    return None

def raw_momentum_score(closes):
    """Pure momentum score without trend filter. Just 3m+6m returns."""
    if len(closes) < 130: return 0
    r3 = closes[-1] / closes[-63] - 1 if closes[-63] > 1e-6 else 0
    r6 = closes[-1] / closes[-126] - 1 if closes[-126] > 1e-6 else 0
    m3 = min(max(r3 / 0.5, 0), 1) if r3 > 0 else 0
    m6 = min(max(r6 / 0.8, 0), 1) if r6 > 0 else 0
    return (0.5 * m3 + 0.5 * m6) * 100

def adx_arr(h, l, c, p=14):
    n = len(c)
    if n < p*2: return None
    tr = np.zeros(n); tr[0] = h[0]-l[0]
    for i in range(1, n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        if up > dn and up > 0: pdm[i] = up
        if dn > up and dn > 0: mdm[i] = dn
    atr = np.full(n, np.nan); atr[p] = float(np.mean(tr[1:p+1]))
    for i in range(p+1, n): atr[i] = (tr[i] + (p-1)*atr[i-1]) / p
    ps = float(np.mean(pdm[1:p+1])); ms = float(np.mean(mdm[1:p+1]))
    pdi = np.full(n, np.nan); mdi = np.full(n, np.nan)
    pdi[p] = ps/max(atr[p], 0.001)*100; mdi[p] = ms/max(atr[p], 0.001)*100
    for i in range(p+1, n):
        ps = (pdm[i] + (p-1)*ps)/p; ms = (mdm[i] + (p-1)*ms)/p
        pdi[i] = min(ps/max(atr[i], 0.001)*100, 1000)
        mdi[i] = min(ms/max(atr[i], 0.001)*100, 1000)
    dx = np.abs(pdi-mdi)/(pdi+mdi+1e-10)*100
    ax = np.full(n, np.nan); seed = float(np.nanmean(dx[p:p*2]))
    ax[p*2-1] = 0.0 if np.isnan(seed) else seed; ds = ax[p*2-1]
    for i in range(p*2, n):
        vi = dx[i] if not np.isnan(dx[i]) else ds; ds = (vi + (p-1)*ds)/p; ax[i] = ds
    return ax

def is_stock_uptrend(cl, hi, lo, vol):
    if len(cl) < 200: return False, 0
    m120 = sma(cl, 120)
    if m120 is None or np.isnan(m120[-1]): return False, 0
    above = cl[-1] > m120[-1]
    rh = np.max(hi[-20:]); ph = np.max(hi[-60:-20])
    rl = np.min(lo[-20:]); pl = np.min(lo[-60:-20])
    m20 = sma(cl, 20); m60 = sma(cl, 60)
    if m20 is None or m60 is None: return False, 0
    if np.isnan(m20[-1]) or np.isnan(m60[-1]): return False, 0
    align = m20[-1] > m60[-1] > m120[-1]
    av = adx_arr(hi, lo, cl, 14)
    adx_ok = av is not None and not np.isnan(av[-1]) and av[-1] > 25
    v20 = np.mean(vol[-21:-1]); surge = vol[-1] > v20 * 1.2
    score = sum([above, rh > ph and rl > pl, align, adx_ok, surge])
    return above and adx_ok and score >= 3, score

def trend_score(cl):
    if len(cl) < 130: return 0
    r3 = cl[-1]/cl[-63]-1 if cl[-63] > 1e-6 else 0
    r6 = cl[-1]/cl[-126]-1 if cl[-126] > 1e-6 else 0
    m3 = min(max(r3/0.5, 0), 1) if r3 > 0 else 0; m6 = min(max(r6/0.8, 0), 1) if r6 > 0 else 0
    mom = (0.5*m3 + 0.5*m6)*100; w = cl[-61:]; dr = np.diff(w)/(w[:-1]+1e-10)
    vs = (1 - min(np.nanstd(dr)/0.04, 1))*100
    return 0.6*mom + 0.4*vs

def metrics(eq_curve):
    eq = np.array(eq_curve); ny = len(eq_curve) / 12.0
    if eq[0] <= 0 or ny <= 0: return {"cagr": 0, "sharpe": 0, "maxdd": 100}
    cagr = ((eq[-1] / eq[0]) ** (1 / ny) - 1) * 100
    mr = np.diff(eq) / (eq[:-1] + 1e-10)
    sh = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 else 0
    peak = eq[0]; mdd = 0.0
    for v in eq:
        if v > peak: peak = v
        d = (peak - v) / peak * 100
        if d > mdd: mdd = d
    return {"cagr": cagr, "sharpe": sh, "maxdd": mdd}

def build_state_map(csi_dates, csi_closes, csi_volumes, N, M):
    n = len(csi_closes); ma120 = sma(csi_closes, 120); ma60 = sma(csi_closes, 60)
    a120 = (csi_closes > ma120) & (~np.isnan(ma120))
    a60  = (csi_closes > ma60)  & (~np.isnan(ma60))
    st = np.full(n, 0, dtype=int); cd, cf, fd = -1, -1, -1
    for i in range(121, n):
        if i <= cd: st[i] = 3; continue
        if fd >= 0:
            if a120[i]: st[i] = 2; continue
            else: fd = -1
        if cf >= 0:
            if not a120[i]: cd = i+M-1; st[i] = 3; cf = -1; continue
            if i-cf+1 == N: st[i] = 2; fd = i; cf = -1
            else: st[i] = 1; continue
        if a120[i] and not a120[i-1]: cf = i; st[i] = 1
        elif a60[i]: st[i] = 4
        else: st[i] = 0
    return {str(csi_dates[j]): int(st[j]) for j in range(n)}

# ═══════════ STRATEGY RUNNERS ═══════════

def run_backtest_simple(stock_pool_fn, stock_data, all_dates, start, end, use_timing=False, decay_fn=None):
    """
    Generic monthly rebalance backtest.
    stock_pool_fn(rd, stock_data) -> [(symbol, score), ...] or None if no stocks to buy.
    use_timing: if True, CSI300 120MA/60MA state machine gates position size.
    decay_fn: if given, applied to months_in_cycle.
    """
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; eq = [cash]; max_eq = cash; dd_active = False
    months_in_cycle = 0; prev_mst = 0

    state_map = None
    if use_timing:
        s2 = DataStorage(); r2 = s2.load_bars("510300")
        csi_d2 = np.array([x.trade_date for x in r2])
        csi_c2 = np.array([x.close for x in r2], dtype=np.float64)
        csi_v2 = np.array([x.volume for x in r2], dtype=np.float64)
        state_map = build_state_map(csi_d2, csi_c2, csi_v2, 5, 40)

    for rd in rebal:
        # Determine position size
        if use_timing:
            mst = state_map.get(rd, 0)
            if mst in (2, 4):
                if prev_mst not in (2, 4): months_in_cycle = 1
                else: months_in_cycle += 1
            else: months_in_cycle = 0
            prev_mst = mst
            base = 1.0 if mst == 2 else (0.5 if mst == 4 else 0.0)
        else:
            base = 1.0; months_in_cycle = 1

        sm_val = base
        if decay_fn: sm_val *= decay_fn(months_in_cycle)

        # Stop-loss
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"] * 0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p/holdings[sym]["hwm"]-1)*100 < -10:
                cash += holdings[sym]["val"]*(1-COMM); del holdings[sym]

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())
        if total > max_eq: max_eq = total
        dd = (max_eq-total)/max_eq*100 if max_eq > 0 else 0
        if dd > 15:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            dd_active = True
        elif dd_active and total/max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue

        if sm_val == 0:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            eq.append(cash); continue

        pool = stock_pool_fn(rd, stock_data)
        if pool is None:
            eq.append(cash + sum(h["qty"]*h["price"] for h in holdings.values())); continue

        selected = {t[0] for t in pool[:5]}

        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]

        n_pos = max(len(selected), 1); per = total * sm_val / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: continue
            tq = int(per/p/100)*100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff > 0 and cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym]["qty"] = tq
                    elif diff < 0: cash += cost*(1-COMM); holdings[sym]["qty"] = tq
            else:
                cost = tq*p
                if cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}
        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())
        eq.append(total)
    return metrics(eq)


# ═══════════ STOCK POOLS ═══════════

def make_pool_raw_momentum(stock_data):
    """All stocks with >=200 bars, sorted by raw momentum (no trend filter)."""
    def pool_fn(rd, sdata):
        t = []
        for code in sdata:
            d2 = data_at(code, rd, 260, sdata)
            if d2 is None: continue
            cl = d2[0]
            if len(cl) >= 130:
                sc = raw_momentum_score(cl)
                if sc > 0: t.append((code, sc))
        if not t: return None
        t.sort(key=lambda x: x[1], reverse=True)
        return t
    return pool_fn

def make_pool_trend_filtered(stock_data):
    """Only stocks passing 5-condition trend check."""
    # Use precomputed trending cache (built below)
    pass  # We'll build a dict

# ═══════════ MAIN ═══════════

if __name__ == "__main__":
    t0 = time.time()
    print("Loading...")
    storage = DataStorage()
    raw = storage.load_bars("510300")
    csi_d = np.array([r.trade_date for r in raw])
    csi_c = np.array([r.close for r in raw], dtype=np.float64)
    csi_v = np.array([r.volume for r in raw], dtype=np.float64)

    all_f = sorted(storage.clean_dir.glob("*.parquet"))
    codes = [p.stem for p in all_f if len(p.stem) == 6 and not p.stem.startswith(("51", "58", "15", "56"))]
    stock_data = {}; all_ds = set()
    for i, code in enumerate(codes):
        if (i+1) % 200 == 0: print(f"  Loading {i+1}/{len(codes)}...")
        rs = storage.load_bars(code)
        if not rs or len(rs) < 200: continue
        d = [r.trade_date for r in rs]
        stock_data[code] = (np.array([r.close for r in rs], dtype=np.float64),
                            np.array([r.high for r in rs], dtype=np.float64),
                            np.array([r.low for r in rs], dtype=np.float64),
                            np.array([r.volume for r in rs], dtype=np.float64), d)
        all_ds.update(d)
    all_dates = sorted(all_ds)

    # Precompute both stock pools
    print("Precomputing stock pools...")
    rebal_dates = monthly_dates(all_dates, TEST_START, TEST_END)
    pool_raw = {}   # raw momentum, no filter
    pool_trend = {} # 5-condition trend filter + trend_score
    for i, rd in enumerate(rebal_dates):
        if i % 12 == 0: print(f"  {i}/{len(rebal_dates)}")
        raw_t = []; trend_t = []
        for code in stock_data:
            d2 = data_at(code, rd, 260, stock_data)
            if d2 is None: continue
            cl, hi, lo, vo = d2
            if len(cl) >= 130:
                rms = raw_momentum_score(cl)
                if rms > 0: raw_t.append((code, rms))
            is_t, nc = is_stock_uptrend(cl, hi, lo, vo)
            if is_t and nc >= 3:
                trend_t.append((code, trend_score(cl)))
        raw_t.sort(key=lambda x: x[1], reverse=True)
        trend_t.sort(key=lambda x: x[1], reverse=True)
        pool_raw[rd] = raw_t
        pool_trend[rd] = trend_t

    # How many months have 0 stocks in each pool?
    zero_raw = sum(1 for rd in rebal_dates if not pool_raw.get(rd))
    zero_trend = sum(1 for rd in rebal_dates if not pool_trend.get(rd))
    print(f"\nMonths with 0 stocks in raw mom pool:    {zero_raw}/{len(rebal_dates)}")
    print(f"Months with 0 stocks in trend-filt pool: {zero_trend}/{len(rebal_dates)}")

    # ═══════ Run all comparators ═══════
    raw_pool_fn   = lambda rd, sd: pool_raw.get(rd, None)
    trend_pool_fn = lambda rd, sd: pool_trend.get(rd, None)

    a43_fn = lambda m: 1.0 if m <= 4 else (0.75 if m <= 8 else 0.50)

    # 1. Raw momentum, no timing
    r1 = run_backtest_simple(raw_pool_fn, stock_data, all_dates, TEST_START, TEST_END, use_timing=False)
    # 2. Trend-filtered, no timing (always invested when stocks exist)
    r2 = run_backtest_simple(trend_pool_fn, stock_data, all_dates, TEST_START, TEST_END, use_timing=False)
    # 3. Trend-filtered + market timing
    r3 = run_backtest_simple(trend_pool_fn, stock_data, all_dates, TEST_START, TEST_END, use_timing=True)
    # 4. Trend-filtered + market timing + A43
    r4 = run_backtest_simple(trend_pool_fn, stock_data, all_dates, TEST_START, TEST_END, use_timing=True, decay_fn=a43_fn)

    # CSI300 B&H
    csi_start_idx = csi_end_idx = None
    for i, d in enumerate(raw):
        if d.trade_date >= TEST_START and csi_start_idx is None: csi_start_idx = i
        if d.trade_date <= TEST_END: csi_end_idx = i
    csi_ny = (int(TEST_END[:4]) - int(TEST_START[:4])) + 1
    csi_cagr = ((raw[csi_end_idx].close / raw[csi_start_idx].close) ** (1/max(csi_ny,0.5)) - 1) * 100

    print(f"\n{'='*85}")
    print("PROPER ATTRIBUTION: Stock Selection vs Market Timing")
    print(f"{'='*85}")
    print(f"\n{'Strategy':<45s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s}")
    print("-"*70)
    print(f"{'CSI300 B&H':<45s} {csi_cagr:+7.2f}% {'---':>7s} {'---':>7s}")
    print(f"{'[A] Raw momentum, always-invested':<45s} {r1['cagr']:+7.2f}% {r1['sharpe']:6.3f} {r1['maxdd']:5.1f}%")
    print(f"{'[B] Trend-filtered, always-invested':<45s} {r2['cagr']:+7.2f}% {r2['sharpe']:6.3f} {r2['maxdd']:5.1f}%")
    print(f"{'[C] Trend-filtered + market timing':<45s} {r3['cagr']:+7.2f}% {r3['sharpe']:6.3f} {r3['maxdd']:5.1f}%")
    print(f"{'[D] Trend-filtered + timing + A43':<45s} {r4['cagr']:+7.2f}% {r4['sharpe']:6.3f} {r4['maxdd']:5.1f}%")

    print(f"\n{'='*85}")
    print("DECOMPOSITION:")
    print(f"{'='*85}")
    stock_selection_raw = r1['cagr'] - csi_cagr
    quality_filter = r2['cagr'] - r1['cagr']
    timing_pure = r3['cagr'] - r2['cagr']
    timing_decay = r4['cagr'] - r2['cagr']
    print(f"  Stock selection (raw mom vs CSI300):         {stock_selection_raw:+.1f}%")
    print(f"  Stock quality filter (trend-filt vs raw mom): {quality_filter:+.1f}%")
    print(f"  Market timing (timing vs always-invested):   {timing_pure:+.1f}%")
    print(f"  Market timing + A43 decay:                   {timing_decay:+.1f}%")
    print(f"{'='*85}")

    print(f"\nDone in {time.time()-t0:.0f}s")

"""
Validate lightweight backtest loop: Buy-and-hold CSI300 vs loop vs BOND_ROTATE.
Also: isolate stock selection vs market timing contributions.
"""
import os, sys, time
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025; SLIPPAGE_BPS = 5
TEST_START, TEST_END = "20200101", "20251231"

def monthly_dates(dates, s, e):
    m = []
    for d in dates:
        if d < s or d > e: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

print("Loading...")
t0 = time.time()
storage = DataStorage()
raw = storage.load_bars("510300")
csi_dates = np.array([r.trade_date for r in raw])
csi_closes = np.array([r.close for r in raw], dtype=np.float64)

# ── Method 1: Manual buy-and-hold CSI300 ──
rebal = monthly_dates(csi_dates, TEST_START, TEST_END)
start_idx = end_idx = None
for i, d in enumerate(raw):
    if d.trade_date >= rebal[0] and start_idx is None: start_idx = i
    if d.trade_date <= rebal[-1]: end_idx = i

start_px = raw[start_idx].close; end_px = raw[end_idx].close
shares = int(CAPITAL * 0.99 / start_px / 100) * 100
manual_final = (CAPITAL - shares * start_px * (1 + COMM + SLIPPAGE_BPS/10000)
                + shares * end_px * (1 - COMM - SLIPPAGE_BPS/10000))
ny = (int(rebal[-1][:4]) - int(rebal[0][:4])) + (int(rebal[-1][4:6]) - int(rebal[0][4:6])) / 12.0
manual_cagr = ((manual_final / CAPITAL) ** (1/ny) - 1) * 100
peak = raw[start_idx].close; manual_mdd = 0.0
for i in range(start_idx, end_idx + 1):
    if raw[i].close > peak: peak = raw[i].close
    dd = (peak - raw[i].close) / peak * 100
    if dd > manual_mdd: manual_mdd = dd

print(f"Manual CSI300 B&H:  CAGR={manual_cagr:+.2f}%  MaxDD={manual_mdd:.1f}%")

# ── Method 2: Through loop ──
dates_l = monthly_dates(csi_dates, TEST_START, TEST_END)
cash = CAPITAL; hold = False; eq = [cash]
for rd in dates_l:
    px = None
    for b in raw:
        if b.trade_date == rd: px = b.close; break
    if px is None: continue
    if not hold:
        qty = int(cash * 0.99 / px / 100) * 100
        cost = qty * px
        slip = cost * SLIPPAGE_BPS / 10000
        if cost + slip + cost * COMM <= cash:
            cash -= (cost + slip + cost * COMM)
            hold = True; bpx = px; bq = qty
    total = cash + (bq * px if hold else 0)
    eq.append(total)
loop_cagr = ((eq[-1] / CAPITAL) ** (1/ny) - 1) * 100
print(f"Loop B&H:           CAGR={loop_cagr:+.2f}%  diff={manual_cagr-loop_cagr:+.3f}%")
print(f"Loop validated: {'PASS' if abs(manual_cagr-loop_cagr) < 1.0 else 'FAIL'}")

# ── Method 3: Stock universe always-invested (no timing, just stock selection) ──
all_f = sorted(storage.clean_dir.glob("*.parquet"))
codes = [p.stem for p in all_f if len(p.stem) == 6 and not p.stem.startswith(("51", "58", "15", "56"))]
stock_data = {}; all_ds = set()
for i, code in enumerate(codes):
    rs = storage.load_bars(code)
    if not rs or len(rs) < 200: continue
    d = [r.trade_date for r in rs]
    stock_data[code] = (np.array([r.close for r in rs], dtype=np.float64),
                        np.array([r.high for r in rs], dtype=np.float64),
                        np.array([r.low for r in rs], dtype=np.float64),
                        np.array([r.volume for r in rs], dtype=np.float64), d)
    all_ds.update(d)
all_dates = sorted(all_ds)

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

def sma(arr, p):
    if len(arr) < p: return None
    o = np.full(len(arr), np.nan); cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p; return o

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

# ── Always-invested stock rotation (NO MARKET TIMING) ──
def run_always_invested(tc, stock_data, all_dates, start, end, top_n=5):
    """Same stock selection as BOND_ROTATE, but always invested (no market gate)."""
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; eq = [cash]; max_eq = cash; dd_active = False
    for rd in rebal:
        # Always try to stay invested
        trending = tc.get(rd, [])
        if not trending: eq.append(cash + sum(h["qty"]*h["price"] for h in holdings.values())); continue
        selected = {t[0] for t in trending[:top_n]}

        # Stop-loss
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"] * 0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p/holdings[sym]["hwm"]-1)*100 < -10:
                cash += holdings[sym]["val"] * (1-COMM); del holdings[sym]

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())
        if total > max_eq: max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0
        if dd > 15:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            dd_active = True
        elif dd_active and total/max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue

        # Rotate out
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]

        # Buy new
        n_pos = max(len(selected), 1); per = total / n_pos * 0.90
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

# Precompute
d_l = monthly_dates(all_dates, TEST_START, TEST_END)
tc = {}
for i, d in enumerate(d_l):
    t = []
    for code in stock_data:
        d2 = data_at(code, d, 260, stock_data)
        if d2 is None: continue
        cl, hi, lo, vo = d2
        is_t, nc = is_stock_uptrend(cl, hi, lo, vo)
        if is_t and nc >= 3: t.append((code, trend_score(cl)))
    t.sort(key=lambda x: x[1], reverse=True)
    tc[d] = t

r_always = run_always_invested(tc, stock_data, all_dates, TEST_START, TEST_END)
print(f"\nAlways-invested (no timing): C={r_always['cagr']:+.2f}% S={r_always['sharpe']:.3f} D={r_always['maxdd']:.1f}%")

# ── BOND_ROTATE entry-only ──
def build_state_map(csi_dates, csi_closes, csi_volumes, N, M, vt=None, md=None):
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
            if i-cf+1 == N:
                st[i] = 2; fd = i; cf = -1
            else: st[i] = 1; continue
        if a120[i] and not a120[i-1]: cf = i; st[i] = 1
        elif a60[i]: st[i] = 4
        else: st[i] = 0
    return {str(csi_dates[j]): int(st[j]) for j in range(n)}

sm = build_state_map(csi_dates, csi_closes,
                     np.array([r.volume for r in raw], dtype=np.float64), 5, 40)

def run_bond_rotate(state_map, tc, stock_data, all_dates, start, end):
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; bu = 0; eq = [cash]; max_eq = cash; dd_active = False
    for rd in rebal:
        mst = state_map.get(rd, 0); sm_val = 1.0 if mst==2 else (0.5 if mst==4 else 0.0)
        if sm_val > 0 and bu > 0: cash += bu * 1.0 * (1-COMM); bu = 0
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"]*0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"]*p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p/holdings[sym]["hwm"]-1)*100 < -10:
                cash += holdings[sym]["val"]*(1-COMM); del holdings[sym]
        total = cash + sum(h["qty"]*h["price"] for h in holdings.values()) + bu
        if total > max_eq: max_eq = total
        dd = (max_eq-total)/max_eq*100 if max_eq > 0 else 0
        if dd > 15:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            if bu > 0: cash += bu*1.0*(1-COMM); bu = 0; dd_active = True
        elif dd_active and total/max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue
        if sm_val == 0:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            if cash > 1000: bu = int(cash*0.99); cash -= bu
            eq.append(cash + bu); continue
        trending = tc.get(rd, [])
        if not trending: eq.append(cash+sum(h["qty"]*h["price"] for h in holdings.values())+bu); continue
        selected = {t[0] for t in trending[:5]}
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
        n_pos = max(len(selected), 1); per = total*sm_val/n_pos*0.90
        for sym in selected:
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: continue
            tq = int(per/p/100)*100
            if tq < 100: continue
            if sym in holdings:
                diff = tq-holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff > 0 and cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym]["qty"] = tq
                    elif diff < 0: cash += cost*(1-COMM); holdings[sym]["qty"] = tq
            else:
                cost = tq*p
                if cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}
        total = cash + sum(h["qty"]*h["price"] for h in holdings.values()) + bu
        eq.append(total)
    return metrics(eq)

r_br = run_bond_rotate(sm, tc, stock_data, all_dates, TEST_START, TEST_END)
print(f"BOND_ROTATE entry-only:  C={r_br['cagr']:+.2f}% S={r_br['sharpe']:.3f} D={r_br['maxdd']:.1f}%")

# ── Summary ──
print(f"\n{'='*70}")
print("ATTRIBUTION:")
print(f"  CSI300 B&H:                C={manual_cagr:+.2f}%")
print(f"  Always-invested stocks:    C={r_always['cagr']:+.2f}%  (stock selection alpha)")
print(f"  BOND_ROTATE (timing+stock): C={r_br['cagr']:+.2f}%  (selection + market timing)")
stock_alpha = r_always['cagr'] - manual_cagr
timing_alpha = r_br['cagr'] - r_always['cagr']
print(f"  Stock selection alpha:     {stock_alpha:+.2f}%  (always-invested vs CSI300 B&H)")
print(f"  Market timing alpha:       {timing_alpha:+.2f}%  (BOND_ROTATE vs always-invested)")
print(f"{'='*70}")

print(f"\nDone in {time.time()-t0:.0f}s")

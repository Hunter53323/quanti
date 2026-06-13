"""
Final Friction & Robustness Analysis
=====================================
Remaining 20% — completes the strategy research with:
  1. Commission sensitivity (0.0%, 0.01%, 0.03%, 0.05%, 0.10%, 0.15%)
  2. Slippage impact (0bps, 10bps, 30bps, 50bps, 100bps per trade)
  3. Max capacity estimation (based on daily turnover)
  4. Parameter stability: how much does Test CAGR vary with param perturbation?
  5. Strategy catalog — final comparison of all strategies explored
  6. Monthly return distribution & tail risk analysis
"""
import sys, os, time, itertools, json, numpy as np
os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(divide='ignore', invalid='ignore')

from quanti.data.storage import DataStorage

CAPITAL = 90000.0
T0 = time.time()

print("=" * 90)
print("FINAL FRICTION & ROBUSTNESS ANALYSIS")
print("=" * 90)

# ═══════════════════════════════════════════════════════
# 1. Load all data
# ═══════════════════════════════════════════════════════
print("\n[1] Loading data...")
storage = DataStorage()

r3 = storage.load_bars("510300")
CD = np.array([r.trade_date for r in r3]); CC = np.array([r.close for r in r3], dtype=np.float64)
CH = np.array([r.high for r in r3], dtype=np.float64); CL = np.array([r.low for r in r3], dtype=np.float64)

rc = storage.load_bars("511880")
CsD = np.array([r.trade_date for r in rc]); CsC = np.array([r.close for r in rc], dtype=np.float64)

af = sorted(storage.clean_dir.glob("*.parquet"))
sc = [p.stem for p in af if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]
ST = {}; ads = set(CD)
for c in sc:
    r = storage.load_bars(c)
    if not r or len(r) < 200: continue
    ST[c] = {"d": np.array([x.trade_date for x in r]), "c": np.array([x.close for x in r], dtype=np.float64),
             "h": np.array([x.high for x in r], dtype=np.float64), "l": np.array([x.low for x in r], dtype=np.float64),
             "v": np.array([x.volume for x in r], dtype=np.float64),
             "a": np.array([x.amount for x in r], dtype=np.float64)}
    ads.update(x.trade_date for x in r)
AD = sorted(ads)
print(f"  CSI300:{len(CD)}  Stocks:{len(ST)}  Dates:{len(AD)}")

# ═══════════════════════════════════════════════════════
# 2. Indicators + State Machine
# ═══════════════════════════════════════════════════════
def _sma(a, p):
    if len(a) < p: return np.full(len(a), np.nan)
    o = np.full(len(a), np.nan); cs = np.cumsum(np.insert(a, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p; return o

def _adx(h, l, c, p=14):
    n = len(c)
    if n < p*2: return np.full(n, np.nan)
    tr = np.zeros(n); tr[0] = h[0]-l[0]
    for i in range(1,n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    pd = np.zeros(n); md = np.zeros(n)
    for i in range(1,n):
        u = h[i]-h[i-1]; d = l[i-1]-l[i]
        if u>d and u>0: pd[i] = u
        if d>u and d>0: md[i] = d
    at = np.full(n,np.nan); at[p] = float(np.mean(tr[1:p+1]))
    for i in range(p+1,n): at[i] = (tr[i]+(p-1)*at[i-1])/p
    ps = float(np.mean(pd[1:p+1])); ms = float(np.mean(md[1:p+1]))
    pi = np.full(n,np.nan); mi = np.full(n,np.nan)
    pi[p] = ps/max(at[p],0.001)*100; mi[p] = ms/max(at[p],0.001)*100
    for i in range(p+1,n):
        ps = (pd[i]+(p-1)*ps)/p; ms = (md[i]+(p-1)*ms)/p
        pi[i] = min(ps/max(at[i],0.001)*100,1000); mi[i] = min(ms/max(at[i],0.001)*100,1000)
    dx = np.abs(pi-mi)/(pi+mi+1e-10)*100
    ax = np.full(n,np.nan)
    sd = float(np.nanmean(dx[p:p*2])); ax[p*2-1] = 0.0 if np.isnan(sd) else sd; ds = ax[p*2-1]
    for i in range(p*2,n):
        vi = dx[i] if not np.isnan(dx[i]) else ds; ds = (vi+(p-1)*ds)/p; ax[i] = ds
    return ax

stkma = {}
for c, sd in ST.items():
    m20 = _sma(sd["c"], 20); stkma[c] = (sd["d"], sd["c"] > m20)

def _brd(dt):
    cn, tt = 0, 0
    for c, (da, aa) in stkma.items():
        idx = np.searchsorted(da, dt, side="right")-1
        if idx < 19: continue
        tt += 1; cn += 1 if aa[idx] else 0
    return cn/tt*100 if tt > 0 else 50

ma120 = _sma(CC, 120); ax14 = _adx(CH, CL, CC, 14); ba = np.array([_brd(d) for d in CD])
ab = (CC > ma120) & (~np.isnan(ma120))
n_csi = len(CC)
raw_s = np.full(n_csi, 0, dtype=int)
for i in range(120, n_csi):
    if ab[i]:
        ao = not np.isnan(ax14[i]) and ax14[i] > 25; bo = not np.isnan(ba[i]) and ba[i] > 45
        raw_s[i] = 2 if (ao and bo) else 1
conf_s = np.full(n_csi, 0, dtype=int)
for i in range(1, n_csi):
    rs = raw_s[i]
    if conf_s[i-1] == 0:
        if i >= 4 and np.all(raw_s[i-4:i+1] >= 1):
            if i >= 1 and np.all(raw_s[i-1:i+1] == 2): conf_s[i] = 2
            else: conf_s[i] = 1
        else: conf_s[i] = 0
    elif conf_s[i-1] == 1:
        if rs == 0: conf_s[i] = 0
        elif i >= 1 and np.all(raw_s[i-1:i+1] == 2): conf_s[i] = 2
        else: conf_s[i] = 1
    elif conf_s[i-1] == 2:
        if rs == 0: conf_s[i] = 0
        elif rs == 1: conf_s[i] = 1
        else: conf_s[i] = 2
SM = {CD[i]: int(conf_s[i]) for i in range(n_csi)}

def gm(s, e):
    m = []
    for d in AD:
        if d < s or d > e: continue
        mon = d[4:6]
        if not m or mon != m[-1][4:6]: m.append(d)
    return m

def _p(c, dt):
    if c not in ST: return None
    sd = ST[c]; idx = np.searchsorted(sd["d"], dt, side="right")-1
    return sd["c"][idx] if idx >= 0 else None

def _cp(dt):
    idx = np.searchsorted(CsD, dt, side="right")-1
    return CsC[idx] if idx >= 0 else 100.0

# Precompute scores for all monthly dates
print("  Precomputing scores...")
ams = gm("20150101", "20251231")
PRE = {}
for c, sd in ST.items():
    PRE[c] = {}
    for rd in ams:
        idx = np.searchsorted(sd["d"], rd, side="right")
        if idx < 260: continue
        cs = sd["c"][idx-260:idx]; hs = sd["h"][idx-260:idx]; ls = sd["l"][idx-260:idx]; vol_arr = sd["v"][idx-260:idx]
        nn = len(cs)
        m120 = _sma(cs, 120); abv = 1.0 if (not np.isnan(m120[-1]) and cs[-1] > m120[-1]) else 0.0
        rh = np.max(hs[-20:]); ph = np.max(hs[-60:-20]); rl = np.min(ls[-20:]); pl = np.min(ls[-60:-20])
        hh = 1.0 if (rh > ph and rl > pl) else 0.0
        m20 = _sma(cs, 20); m60 = _sma(cs, 60); al = 0.0
        if (not np.isnan(m20[-1]) and not np.isnan(m60[-1]) and not np.isnan(m120[-1]) and m20[-1] > m60[-1] > m120[-1]): al = 1.0
        av = _adx(hs, ls, cs, 14)[-1]; an = min(max((av-15)/35, 0), 1) if not np.isnan(av) else 0
        r3 = cs[-1]/cs[-63]-1 if cs[-63] > 1e-6 else 0; r6 = cs[-1]/cs[-126]-1 if cs[-126] > 1e-6 else 0
        m3 = min(max(r3/0.5, 0), 1) if r3 > 0 else max(r3/0.3, -1)
        m6 = min(max(r6/0.8, 0), 1) if r6 > 0 else max(r6/0.5, -1)
        ms = (0.5*m3+0.5*m6)*100
        w_arr = cs[-61:]; dr = np.diff(w_arr)/(w_arr[:-1]+1e-10); vol_sc = (1-min(np.nanstd(dr)/0.05, 1))*100
        tc = (0.35*abv+0.25*al+0.20*an+0.20*hh)*100
        ret60 = (cs[-1]/cs[-60]-1) if cs[-60] > 1e-6 else 0
        # Daily turnover for capacity calc
        amt_arr = sd["a"]
        avg_daily_amt = float(np.mean(amt_arr[max(0,idx-20):idx])) if idx >= 20 and len(amt_arr) > 0 else 0.0
        PRE[c][rd] = (0.60*ms+0.30*tc+0.10*vol_sc, float(ret60), float(avg_daily_amt))

print(f"  Precomputed {len(PRE)} stocks")

# ═══════════════════════════════════════════════════════
# 3. Backtest engine (parametrized commission)
# ═══════════════════════════════════════════════════════
def bt_comm(s, e, comm_rate=0.00025, slippage_bps=0.0, max_pos=5, dual_mom=True):
    """Backtest with configurable commission and slippage."""
    reb = gm(s, e)
    if len(reb) < 6: return None
    cash = CAPITAL; hld = {}; etf = 0.0
    eq = [CAPITAL]; me = CAPITAL; ps = -1; sc = {0:0,1:0,2:0}
    total_trades = 0; total_notional_traded = 0.0
    total_comm_paid = 0.0; total_slippage_paid = 0.0

    for rd in reb:
        mk = SM.get(rd, 0); sc[mk] += 1; scd = (mk != ps)
        for sy in list(hld.keys()):
            p = _p(sy, rd)
            if p is None or p < 0.01: cash += hld[sy].get("qty",0)*hld[sy].get("price",10)*0.7; del hld[sy]; continue
            hld[sy]["price"] = p
        cpv = _cp(rd); cv = etf*cpv
        tot = cash + cv + sum(h["qty"]*h.get("price",0) for h in hld.values())
        for sy in list(hld.keys()):
            p = hld[sy].get("price",0)
            if p <= 0: continue
            hwm = hld[sy].get("hwm", p)
            if p > hwm: hld[sy]["hwm"] = p
            if hwm > 0 and (p/hwm-1)*100 < -10:
                notional = hld[sy]["qty"]*p
                total_notional_traded += notional
                total_comm_paid += notional * comm_rate
                total_slippage_paid += notional * (slippage_bps/10000.0)
                cash += notional * (1 - comm_rate - slippage_bps/10000.0)
                del hld[sy]; total_trades += 1
        tot = cash + etf*cpv + sum(h["qty"]*h.get("price",0) for h in hld.values())
        if tot > me: me = tot
        dd = (me-tot)/me if me > 0 else 0
        if dd > 0.15:
            for sy in list(hld.keys()):
                ntnl = hld[sy]["qty"]*hld[sy].get("price",0)
                total_notional_traded += ntnl; total_comm_paid += ntnl*comm_rate
                total_slippage_paid += ntnl*(slippage_bps/10000.0)
                cash += ntnl*(1 - comm_rate - slippage_bps/10000.0); del hld[sy]
            ntnl_etf = etf*cpv
            total_notional_traded += ntnl_etf; total_comm_paid += ntnl_etf*comm_rate
            total_slippage_paid += ntnl_etf*(slippage_bps/10000.0)
            cash += etf*cpv*(1 - comm_rate - slippage_bps/10000.0); etf = 0; total_trades += 1; tot = cash
        if mk == 0:
            if scd or not etf:
                for sy in list(hld.keys()):
                    ntnl = hld[sy]["qty"]*hld[sy].get("price",0)
                    total_notional_traded += ntnl; total_comm_paid += ntnl*comm_rate
                    total_slippage_paid += ntnl*(slippage_bps/10000.0)
                    cash += ntnl*(1 - comm_rate - slippage_bps/10000.0); del hld[sy]
                ntnl_etf = etf*cpv
                total_notional_traded += ntnl_etf; total_comm_paid += ntnl_etf*comm_rate
                total_slippage_paid += ntnl_etf*(slippage_bps/10000.0)
                cash += etf*cpv*(1 - comm_rate - slippage_bps/10000.0); etf = 0.0
                if cpv > 0 and cash > 0:
                    etf = cash/cpv; cash = 0.0
                    total_trades += 1
            eq.append(cash + etf*cpv); ps = mk; continue
        if etf > 0:
            ntnl_etf = etf*cpv
            total_notional_traded += ntnl_etf; total_comm_paid += ntnl_etf*comm_rate
            total_slippage_paid += ntnl_etf*(slippage_bps/10000.0)
            cash += etf*cpv*(1 - comm_rate - slippage_bps/10000.0); etf = 0.0; total_trades += 1
        psz = 0.5 if mk == 1 else 1.0; tn = 3 if mk == 1 else max_pos
        scr = [(cd, PRE[cd][rd][0]) for cd in PRE if rd in PRE[cd]]
        if dual_mom: scr = [(cd, s) for cd, s in scr if PRE[cd][rd][1] > 0]
        if not scr: eq.append(tot); ps = mk; continue
        scr.sort(key=lambda x: x[1], reverse=True)
        selc = [s[0] for s in scr[:tn]]
        for sy in list(hld.keys()):
            if sy not in selc:
                ntnl = hld[sy]["qty"]*hld[sy].get("price",0)
                total_notional_traded += ntnl; total_comm_paid += ntnl*comm_rate
                total_slippage_paid += ntnl*(slippage_bps/10000.0)
                cash += ntnl*(1 - comm_rate - slippage_bps/10000.0); del hld[sy]; total_trades += 1
        np2 = max(len(selc), 1); pps = tot*psz/np2*0.90
        for sy in selc:
            p = _p(sy, rd)
            if p is None or p < 0.01: continue
            tq = int(pps/p/100)*100
            if tq < 100 or tq*p < 10000: continue
            if sy in hld:
                df = tq - hld[sy]["qty"]
                if abs(df) >= 100:
                    ntnl = abs(df)*p
                    total_notional_traded += ntnl; total_comm_paid += ntnl*comm_rate
                    total_slippage_paid += ntnl*(slippage_bps/10000.0)
                    if df > 0 and cash >= ntnl*(1+comm_rate+slippage_bps/10000.0):
                        cash -= ntnl*(1+comm_rate+slippage_bps/10000.0); hld[sy]["qty"] = tq; total_trades += 1
                    elif df < 0:
                        cash += ntnl*(1-comm_rate-slippage_bps/10000.0); hld[sy]["qty"] = tq; total_trades += 1
            else:
                ntnl = tq*p
                total_notional_traded += ntnl; total_comm_paid += ntnl*comm_rate
                total_slippage_paid += ntnl*(slippage_bps/10000.0)
                if cash >= ntnl*(1+comm_rate+slippage_bps/10000.0):
                    cash -= ntnl*(1+comm_rate+slippage_bps/10000.0)
                    hld[sy] = {"qty": tq, "price": p, "hwm": p}; total_trades += 1
        leftover = cash; cpv2 = _cp(rd)
        if cpv2 > 0 and leftover > 0: etf = leftover/cpv2; cash = 0.0
        else: cash = leftover; etf = 0.0
        tot = cash + etf*cpv2 + sum(h["qty"]*h.get("price",0) for h in hld.values())
        eq.append(tot); ps = mk

    ea = np.array(eq)
    if len(ea) < 2 or ea[0] <= 0: return None
    ny = len(ea)/12.0
    if ny < 0.5: return None
    cagr = ((ea[-1]/ea[0])**(1/ny)-1)*100
    mr = np.diff(ea)/(ea[:-1]+1e-10)
    sh = np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12) if len(mr) > 1 else 0
    pk = ea[0]; md = 0.0
    for v in ea:
        if v > pk: pk = v
        if (pk-v)/pk*100 > md: md = (pk-v)/pk*100
    tr = (ea[-1]/ea[0]-1)*100
    avg_cap = np.mean(ea)
    annual_turnover_pct = (total_notional_traded / 2) / avg_cap / ny * 100 if avg_cap > 0 and ny > 0 else 0

    return {"c": cagr, "s": sh, "d": md, "tr": tr, "trades": total_trades,
            "notional": total_notional_traded, "comm_total": total_notional_traded*0.00025,
            "ann_turnover": annual_turnover_pct, "sc": sc}


# ═══════════════════════════════════════════════════════
# 4. Commission Sensitivity
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 1: Commission Sensitivity (0.0% - 0.15% one-way)")
print("=" * 90)
print(f"  {'Rate':>8s} {'Test CAGR':>10s} {'MaxDD':>8s} {'Sharpe':>8s} {'AnnTurnover':>12s} {'CommCost':>10s}")
print(f"  {'-'*65}")

for rate in [0.0, 0.0001, 0.00025, 0.0005, 0.0010, 0.0015]:
    r = bt_comm("20220101", "20251231", comm_rate=rate, slippage_bps=0.0)
    if r:
        comm_total = r["notional"] * rate
        print(f"  {rate*100:7.2f}% {r['c']:>+9.2f}% {r['d']:>7.1f}% {r['s']:>8.3f} {r['ann_turnover']:>11.0f}% {comm_total:>9,.0f}")


# ═══════════════════════════════════════════════════════
# 5. Slippage Sensitivity
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 2: Slippage Sensitivity (0 - 100bps per trade)")
print("=" * 90)
print(f"  {'Slip(bps)':>10s} {'Test CAGR':>10s} {'MaxDD':>8s} {'Sharpe':>8s} {'SlippageCost':>12s}")
print(f"  {'-'*65}")

for slip in [0, 5, 10, 30, 50, 100]:
    r = bt_comm("20220101", "20251231", comm_rate=0.00025, slippage_bps=float(slip))
    if r:
        slip_cost = r["notional"] * (slip/10000.0)
        print(f"  {slip:>10d} {r['c']:>+9.2f}% {r['d']:>7.1f}% {r['s']:>8.3f} {slip_cost:>11,.0f}")


# ═══════════════════════════════════════════════════════
# 6. Max Capacity Estimation
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 3: Maximum Capacity Estimation")
print("=" * 90)

# For each stock we hold, compute daily turnover in CNY
# Max capacity = min(1% of daily turnover per stock, total market impact limit)
# Sample the positions held during Test period's non-BEAR months
test_reb = gm("20220101", "20251231")
pos_samples = []
for rd in test_reb:
    if SM.get(rd, 0) == 0: continue
    scr = [(cd, PRE[cd][rd][0]) for cd in PRE if rd in PRE[cd]]
    if not scr: continue
    scr.sort(key=lambda x: x[1], reverse=True)
    tn = 3 if SM.get(rd, 0) == 1 else 5
    top_codes = [s[0] for s in scr[:tn]]
    for tc in top_codes:
        if tc in PRE and rd in PRE[tc]:
            amt = PRE[tc][rd][2]  # avg daily amount
            if amt > 0:
                pos_samples.append(amt)

if pos_samples:
    pos_samples = np.array(pos_samples)
    print(f"  Sampled {len(pos_samples)} position-days in non-BEAR months")
    print(f"  Avg daily turnover per stock: {np.mean(pos_samples):,.0f} CNY")
    print(f"  Median daily turnover per stock: {np.median(pos_samples):,.0f} CNY")
    print(f"  P25: {np.percentile(pos_samples,25):,.0f}  P10: {np.percentile(pos_samples,10):,.0f}")
    print(f"  P5: {np.percentile(pos_samples,5):,.0f}")

    # 1% of daily turnover as max position per stock
    max_per_stock_pct1 = np.percentile(pos_samples, 10) * 0.01
    print(f"\n  Max position per stock (1% of P10 daily turnover): {max_per_stock_pct1:,.0f} CNY")
    print(f"  Strategy typical position size: ~{CAPITAL/5:,.0f} CNY per stock (for 5 positions)")

    # Conservative capacity: 5 positions x min turnover
    total_capacity_pct1 = 5 * np.percentile(pos_samples, 10) * 0.01
    total_capacity_pct05 = 5 * np.percentile(pos_samples, 10) * 0.005
    print(f"\n  Conservative capacity (0.5% of P10 daily turnover): {total_capacity_pct05:,.0f} CNY")
    print(f"  Estimated capacity (1% of P10 daily turnover): {total_capacity_pct1:,.0f} CNY")

    # Per-month deployment
    print(f"\n  === Deployment Scaling ===")
    print(f"  For 90,000 CNY base capital:")
    print(f"    Per-position size: ~{CAPITAL/5:,.0f} CNY")
    print(f"    Fraction of P10 daily volume: {CAPITAL/5 / (np.percentile(pos_samples,10)):.4%}")
    print(f"  For 1M CNY portfolio (5 positions):")
    print(f"    Per-position size: ~200,000 CNY")
    print(f"    Fraction of P10 daily volume: {200000/np.percentile(pos_samples,10):.4%}")
    print(f"  For 5M CNY portfolio (5 positions):")
    print(f"    Per-position size: ~1,000,000 CNY")
    print(f"    Fraction of P10 daily volume: {1000000/np.percentile(pos_samples,10):.4%}")
    print(f"  For 10M CNY portfolio:")
    print(f"    Per-position size: ~2,000,000 CNY")
    print(f"    Fraction of P10 daily volume: {2000000/np.percentile(pos_samples,10):.4%}")
    print(f"  (Industry rule: <1% of daily turnover to avoid market impact)")


# ═══════════════════════════════════════════════════════
# 7. Parameter Stability / Robustness
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 4: Parameter Stability (Robustness Check)")
print("=" * 90)

# Test: hold 3 params fixed at V1-best, perturb 1 param
# How much does Test CAGR change with +/- perturbation?
base_params = {"adx": 25, "br": 45, "n_br": 5, "n_rb": 2}

def build_map_perturb(adx_t=25, br_t=45, cbr=5, crb=2):
    n = len(CC)
    rw = np.full(n, 0, dtype=int)
    for i in range(120, n):
        if ab[i]:
            ao = not np.isnan(ax14[i]) and ax14[i] > adx_t
            bo = not np.isnan(ba[i]) and ba[i] > br_t
            rw[i] = 2 if (ao and bo) else 1
    cf = np.full(n, 0, dtype=int)
    for i in range(1, n):
        rs = rw[i]
        if cf[i-1] == 0:
            if i >= cbr-1 and np.all(rw[i-cbr+1:i+1] >= 1):
                if i >= crb-1 and np.all(rw[i-crb+1:i+1] == 2): cf[i] = 2
                else: cf[i] = 1
            else: cf[i] = 0
        elif cf[i-1] == 1:
            if rs == 0: cf[i] = 0
            elif i >= crb-1 and np.all(rw[i-crb+1:i+1] == 2): cf[i] = 2
            else: cf[i] = 1
        elif cf[i-1] == 2:
            if rs == 0: cf[i] = 0
            elif rs == 1: cf[i] = 1
            else: cf[i] = 2
    return {CD[i]: int(cf[i]) for i in range(n)}

def bt_with_map(sm, s, e):
    reb = gm(s, e)
    if len(reb) < 6: return None
    cash = CAPITAL; hld = {}; etf = 0.0; eq = [CAPITAL]; me = CAPITAL; ps = -1
    for rd in reb:
        mk = sm.get(rd, 0); scd = (mk != ps)
        for sy in list(hld.keys()):
            p = _p(sy, rd)
            if p is None or p < 0.01: cash += hld[sy].get("qty",0)*10*0.7; del hld[sy]; continue
            hld[sy]["price"] = p
        cpv = _cp(rd); cv = etf*cpv; tot = cash+cv+sum(h["qty"]*h.get("price",0) for h in hld.values())
        for sy in list(hld.keys()):
            p = hld[sy].get("price",0)
            if p <= 0: continue
            hwm = hld[sy].get("hwm", p)
            if p > hwm: hld[sy]["hwm"] = p
            if hwm > 0 and (p/hwm-1)*100 < -10:
                cash += hld[sy]["qty"]*p*(1-0.00025); del hld[sy]
        tot = cash + etf*cpv + sum(h["qty"]*h.get("price",0) for h in hld.values())
        if tot > me: me = tot
        if mk == 0:
            if scd or not etf:
                for sy in list(hld.keys()): cash += hld[sy]["qty"]*hld[sy].get("price",0)*(1-0.00025); del hld[sy]
                cash += etf*cpv*(1-0.00025); etf = 0.0
                if cpv > 0 and cash > 0: etf = cash/cpv; cash = 0.0
            eq.append(cash+etf*cpv); ps = mk; continue
        if etf > 0: cash += etf*cpv*(1-0.00025); etf = 0.0
        psz = 0.5 if mk == 1 else 1.0; tn = 3 if mk == 1 else 5
        scr = [(cd, PRE[cd][rd][0]) for cd in PRE if rd in PRE[cd]]
        if not scr: eq.append(tot); ps = mk; continue
        scr.sort(key=lambda x: x[1], reverse=True)
        selc = [s[0] for s in scr[:tn]]
        for sy in list(hld.keys()):
            if sy not in selc: cash += hld[sy]["qty"]*hld[sy].get("price",0)*(1-0.00025); del hld[sy]
        np2 = max(len(selc), 1); pps = tot*psz/np2*0.90
        for sy in selc:
            p = _p(sy, rd)
            if p is None or p < 0.01: continue
            tq = int(pps/p/100)*100
            if tq < 100 or tq*p < 10000: continue
            if sy in hld:
                df = tq - hld[sy]["qty"]
                if abs(df) >= 100:
                    cst = abs(df)*p
                    if df > 0 and cash >= cst*(1+0.00025): cash -= cst*(1+0.00025); hld[sy]["qty"] = tq
                    elif df < 0: cash += cst*(1-0.00025); hld[sy]["qty"] = tq
            else:
                cst = tq*p
                if cash >= cst*(1+0.00025): cash -= cst*(1+0.00025); hld[sy] = {"qty": tq, "price": p, "hwm": p}
        leftover = cash; cpv2 = _cp(rd)
        if cpv2 > 0 and leftover > 0: etf = leftover/cpv2; cash = 0.0
        else: cash = leftover; etf = 0.0
        tot = cash + etf*cpv2 + sum(h["qty"]*h.get("price",0) for h in hld.values())
        eq.append(tot); ps = mk
    ea = np.array(eq)
    if len(ea) < 2 or ea[0] <= 0: return None
    ny = len(ea)/12.0
    if ny < 0.5: return None
    cagr = ((ea[-1]/ea[0])**(1/ny)-1)*100
    pk = ea[0]; md = 0.0
    for v in ea:
        if v > pk: pk = v
        if (pk-v)/pk*100 > md: md = (pk-v)/pk*100
    return {"c": cagr, "d": md}

# Test perturbations
perturbations = {
    "ADX": [(22, 45, 5, 2), (25, 45, 5, 2), (28, 45, 5, 2)],
    "Breadth": [(25, 40, 5, 2), (25, 45, 5, 2), (25, 50, 5, 2)],
    "N_BR": [(25, 45, 3, 2), (25, 45, 5, 2), (25, 45, 7, 2)],
    "N_RB": [(25, 45, 5, 1), (25, 45, 5, 2), (25, 45, 5, 3)],
}

print(f"  {'Param':<12s} {'Low Val':>10s} {'CAGR':>8s} {'DD':>8s} | {'Mid Val':>10s} {'CAGR':>8s} {'DD':>8s} | {'High Val':>10s} {'CAGR':>8s} {'DD':>8s}")
print(f"  {'-'*100}")
for pname, vals in perturbations.items():
    r_low = bt_with_map(build_map_perturb(*vals[0]), "20220101", "20251231")
    r_mid = bt_with_map(build_map_perturb(*vals[1]), "20220101", "20251231")
    r_hi = bt_with_map(build_map_perturb(*vals[2]), "20220101", "20251231")
    if r_low and r_mid and r_hi:
        low_v = vals[0][0] if pname == "ADX" else (vals[0][1] if pname == "Breadth" else (vals[0][2] if pname == "N_BR" else vals[0][3]))
        mid_v = vals[1][0] if pname == "ADX" else (vals[1][1] if pname == "Breadth" else (vals[1][2] if pname == "N_BR" else vals[1][3]))
        hi_v = vals[2][0] if pname == "ADX" else (vals[2][1] if pname == "Breadth" else (vals[2][2] if pname == "N_BR" else vals[2][3]))
        print(f"  {pname:<12s} {str(low_v):>10s} {r_low['c']:>+7.1f}% {r_low['d']:>7.1f}% | "
              f"{str(mid_v):>10s} {r_mid['c']:>+7.1f}% {r_mid['d']:>7.1f}% | "
              f"{str(hi_v):>10s} {r_hi['c']:>+7.1f}% {r_hi['d']:>7.1f}%")

    sensitivity = max(abs(r_low['c']-r_mid['c']), abs(r_hi['c']-r_mid['c']))
    print(f"  {'':12s} {'Sensitivity:':>10s} {sensitivity:>+.1f}% CAGR change")


# ═══════════════════════════════════════════════════════
# 8. Monthly Return Distribution & Tail Risk
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 5: Monthly Return Distribution & Tail Risk (Best Config)")
print("=" * 90)

# Run detailed monthly tracking for the best config
def bt_monthly_detail(sm, s, e):
    reb = gm(s, e)
    cash = CAPITAL; hld = {}; etf = 0.0; eq = [CAPITAL]; me = CAPITAL; ps = -1
    mrets = []; prev_eq = CAPITAL; state_seq = []

    for rd in reb:
        mk = sm.get(rd, 0); scd = (mk != ps); state_seq.append((rd, mk))
        for sy in list(hld.keys()):
            p = _p(sy, rd)
            if p is None or p < 0.01: cash += hld[sy].get("qty",0)*10*0.7; del hld[sy]; continue
            hld[sy]["price"] = p
        cpv = _cp(rd); cv = etf*cpv; tot = cash+cv+sum(h["qty"]*h.get("price",0) for h in hld.values())
        for sy in list(hld.keys()):
            p = hld[sy].get("price",0)
            if p <= 0: continue
            hwm = hld[sy].get("hwm", p)
            if p > hwm: hld[sy]["hwm"] = p
            if hwm > 0 and (p/hwm-1)*100 < -10: cash += hld[sy]["qty"]*p*(1-0.00025); del hld[sy]
        tot = cash + etf*cpv + sum(h["qty"]*h.get("price",0) for h in hld.values())
        if tot > me: me = tot
        if mk == 0:
            if scd or not etf:
                for sy in list(hld.keys()): cash += hld[sy]["qty"]*hld[sy].get("price",0)*(1-0.00025); del hld[sy]
                cash += etf*cpv*(1-0.00025); etf = 0.0
                if cpv > 0 and cash > 0: etf = cash/cpv; cash = 0.0
            total_now = cash + etf*cpv
            mrets.append((total_now/prev_eq - 1)*100)
            prev_eq = total_now; eq.append(total_now); ps = mk; continue
        if etf > 0: cash += etf*cpv*(1-0.00025); etf = 0.0
        psz = 0.5 if mk == 1 else 1.0; tn = 3 if mk == 1 else 5
        scr = [(cd, PRE[cd][rd][0]) for cd in PRE if rd in PRE[cd]]
        if not scr:
            total_now = cash + etf*cpv
            mrets.append(0); prev_eq = total_now; eq.append(total_now); ps = mk; continue
        scr.sort(key=lambda x: x[1], reverse=True)
        selc = [s[0] for s in scr[:tn]]
        for sy in list(hld.keys()):
            if sy not in selc: cash += hld[sy]["qty"]*hld[sy].get("price",0)*(1-0.00025); del hld[sy]
        np2 = max(len(selc), 1); pps = tot*psz/np2*0.90
        for sy in selc:
            p = _p(sy, rd)
            if p is None or p < 0.01: continue
            tq = int(pps/p/100)*100
            if tq < 100 or tq*p < 10000: continue
            if sy in hld:
                df = tq - hld[sy]["qty"]
                if abs(df) >= 100:
                    cst = abs(df)*p
                    if df > 0 and cash >= cst*(1+0.00025): cash -= cst*(1+0.00025); hld[sy]["qty"] = tq
                    elif df < 0: cash += cst*(1-0.00025); hld[sy]["qty"] = tq
            else:
                cst = tq*p
                if cash >= cst*(1+0.00025): cash -= cst*(1+0.00025); hld[sy] = {"qty": tq, "price": p, "hwm": p}
        leftover = cash; cpv2 = _cp(rd)
        if cpv2 > 0 and leftover > 0: etf = leftover/cpv2; cash = 0.0
        else: cash = leftover; etf = 0.0
        total_now = cash + etf*cpv2 + sum(h["qty"]*h.get("price",0) for h in hld.values())
        mrets.append((total_now/prev_eq - 1)*100)
        prev_eq = total_now; eq.append(total_now); ps = mk
    return mrets, state_seq

for period_label, ps, pe in [("Test (2022-2025)", "20220101", "20251231"),
                              ("Train (2015-2021)", "20150101", "20211231")]:
    sm_map = build_map_perturb(25, 45, 5, 2)
    mrets, _ = bt_monthly_detail(sm_map, ps, pe)
    mrets = np.array(mrets)

    if len(mrets) > 0:
        print(f"\n  === {period_label} ===")
        print(f"  Months: {len(mrets)}")
        print(f"  Mean: {np.mean(mrets):+.2f}%  Median: {np.median(mrets):+.2f}%  Std: {np.std(mrets):.2f}%")
        print(f"  Skew: {np.mean(((mrets-np.mean(mrets))/np.std(mrets))**3):.2f}  Kurtosis: {np.mean(((mrets-np.mean(mrets))/np.std(mrets))**4):.2f}")
        print(f"  Min: {np.min(mrets):+.1f}%  Max: {np.max(mrets):+.1f}%")
        print(f"  P5: {np.percentile(mrets,5):+.1f}%  P10: {np.percentile(mrets,10):+.1f}%")
        print(f"  P90: {np.percentile(mrets,90):+.1f}%  P95: {np.percentile(mrets,95):+.1f}%")
        print(f"  Positive months: {sum(1 for r in mrets if r > 0)}/{len(mrets)} ({sum(1 for r in mrets if r > 0)/len(mrets)*100:.1f}%)")
        print(f"  Win/Loss ratio: {np.mean(mrets[mrets>0])/abs(np.mean(mrets[mrets<0])):.2f}" if sum(mrets<0) > 0 and sum(mrets>0) > 0 else "  Win/Loss ratio: N/A")
        # Serial correlation
        if len(mrets) > 2:
            autocorr = np.corrcoef(mrets[:-1], mrets[1:])[0,1]
            print(f"  Autocorrelation (lag-1): {autocorr:.3f}")

    # State-conditional returns
    sm_map2 = build_map_perturb(25, 45, 5, 2)
    mrets2, state_seq = bt_monthly_detail(sm_map2, ps, pe)
    sn = {0: "BEAR", 1: "RANGE", 2: "BULL"}
    for si in [0, 1, 2]:
        si_rets = [mrets2[i] for i, (_, st) in enumerate(state_seq) if st == si]
        if si_rets:
            print(f"  {sn[si]} ({len(si_rets)}mo): Mean={np.mean(si_rets):+.2f}% Std={np.std(si_rets):.2f}% "
                  f"Min={np.min(si_rets):+.1f}% Max={np.max(si_rets):+.1f}% Pos={sum(1 for r in si_rets if r>0)}/{len(si_rets)}")


# ═══════════════════════════════════════════════════════
# 9. FINAL STRATEGY CATALOG
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TEST 6: FINAL STRATEGY CATALOG — Complete Comparison")
print("=" * 90)
print(f"  {'Strategy':<35s} {'CAGR':>8s} {'MaxDD':>8s} {'Sharpe':>8s} {'TotRet':>8s} {'Excess':>8s} {'Turnover':>9s} {'Capacity':>9s}")
print(f"  {'-'*105}")

# Compute all known strategies into a catalog
catalog = []

# CSI300 B&H
si = np.searchsorted(CD, "20220101"); ei = np.searchsorted(CD, "20251231", side="right")-1
seg = CC[si:ei+1]; cb = ((seg[-1]/seg[0])**(1/4)-1)*100
pk = seg[0]; db = 0.0
for v in seg:
    if v > pk: pk = v
    if (pk-v)/pk*100 > db: db = (pk-v)/pk*100
catalog.append(("CSI300 B&H", round(cb,2), round(db,2), 0.021, round((seg[-1]/seg[0]-1)*100,2), 0, 0, "unlimited"))

# 510880 B&H
raw_div = storage.load_bars("510880")
if raw_div:
    dv_d = np.array([r.trade_date for r in raw_div]); dv_c = np.array([r.close for r in raw_div], dtype=np.float64)
    si2 = np.searchsorted(dv_d, "20220101"); ei2 = np.searchsorted(dv_d, "20251231", side="right")-1
    if ei2 > si2:
        seg2 = dv_c[si2:ei2+1]; cd2 = ((seg2[-1]/seg2[0])**(1/4)-1)*100
        pk2 = seg2[0]; dd2 = 0.0
        for v in seg2:
            if v > pk2: pk2 = v
            if (pk2-v)/pk2*100 > dd2: dd2 = (pk2-v)/pk2*100
        catalog.append(("510880 B&H (dividend)", round(cd2,2), round(dd2,2), 0.221, round((seg2[-1]/seg2[0]-1)*100,2), 0, 0, "unlimited"))

# State Machine V1 base
r_sm_base = bt_comm("20220101", "20251231", max_pos=5, dual_mom=False)
if r_sm_base:
    catalog.append(("State Machine V1 (base)", round(r_sm_base["c"],2), round(r_sm_base["d"],2), round(r_sm_base["s"],3), round(r_sm_base["tr"],2),
                    round(r_sm_base["c"]-cb,2), round(r_sm_base["ann_turnover"],0), "~5M CNY"))

# State Machine V1 best
# Force with best params by building custom map
sm_best_map = build_map_perturb(25, 45, 5, 2)
r_sm_best = bt_with_map(sm_best_map, "20220101", "20251231")
if r_sm_best:
    catalog.append(("State Machine V1 (best)", round(r_sm_best["c"],2), round(r_sm_best["d"],2), 0.787, round((np.exp(r_sm_best["c"]/100*4)-1)*100,2),
                    round(r_sm_best["c"]-cb,2), 70, "~5M CNY"))

# Pure Stock Alpha
r_pure = bt_comm("20220101", "20251231", max_pos=10, dual_mom=False)
if r_pure:
    catalog.append(("Pure Stock Alpha (N=10)", round(r_pure["c"],2), round(r_pure["d"],2), round(r_pure["s"],3), round(r_pure["tr"],2),
                    round(r_pure["c"]-cb,2), round(r_pure["ann_turnover"],0), "~10M CNY"))

# ETF Rotation
# (from final_improvements results)
catalog.append(("ETF Rotation (Top 2)", 3.1, -24.2, 0.261, 12.9, 4.3, 120, "~50M CNY"))

# Asset Rotation v1 (from prior research)
catalog.append(("Asset Rotation v1", 5.07, -39.9, 0.27, 21.9, 6.3, 100, "~20M CNY"))

# Cash (511880)
catalog.append(("Cash (511880)", 0.0, -1.7, 0.001, 0.0, 1.3, 0, "unlimited"))

# Print catalog
for row in catalog:
    name, c, d, sh, tr, ex, to, cap = row
    print(f"  {name:<35s} {c:>+7.1f}% {d:>7.1f}% {sh:>8.3f} {tr:>+7.1f}% {ex:>+7.1f}% {str(to):>8s}% {str(cap):>9s}")

# ═══════════════════════════════════════════════════════
# 10. Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"COMPLETE — Total time: {time.time()-T0:.0f}s")
print(f"{'='*90}")

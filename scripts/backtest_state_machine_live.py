"""
Production-Ready State Machine Backtest — MONTHLY Rebalance + ATR Stop
=======================================================================
Monthly rebalancing with ATR(14) trailing stop and round-lot sizing.
Tests the deployment pipeline against the research V1 baseline.

Key deployment changes vs V1 research:
  - ATR(14) trailing stop at 3x ATR (wider than research -10%, accommodates A-share vol)
  - Max 8 positions (reduces concentration vs Top-5)
  - Dual-momentum filter: only buy stocks with ret_60d > 0
  - Round-lot (100 shares), min notional 10,000 CNY
  - 511880 as cash vehicle

Train: 2015-2021  |  Test: 2022-2025
"""
import sys, os, time, itertools
from datetime import datetime

os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from quanti.data.storage import DataStorage

CAPITAL = 90000.0; COMM = 0.00025
T0 = time.time()

print("=" * 90)
print("PRODUCTION STATE MACHINE BACKTEST (Monthly + ATR + Round Lot)")
print("=" * 90)

# ═══════════════════ 1. Load ═══════════════════
print("\n[1] Loading data...")
storage = DataStorage()

raw300 = storage.load_bars("510300")
CSI_D = np.array([r.trade_date for r in raw300]); CSI_C = np.array([r.close for r in raw300], dtype=np.float64)
CSI_H = np.array([r.high for r in raw300], dtype=np.float64); CSI_L = np.array([r.low for r in raw300], dtype=np.float64)

raw_cash = storage.load_bars("511880")
CASH_D = np.array([r.trade_date for r in raw_cash]); CASH_C = np.array([r.close for r in raw_cash], dtype=np.float64)

all_files = sorted(storage.clean_dir.glob("*.parquet"))
stock_codes = [p.stem for p in all_files if len(p.stem) == 6 and not p.stem.startswith(("51","58","15","56"))]
STOCK = {}; all_ds = set(CSI_D)
for code in stock_codes:
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200: continue
    STOCK[code] = {"d": np.array([r.trade_date for r in raw]), "c": np.array([r.close for r in raw], dtype=np.float64),
                   "h": np.array([r.high for r in raw], dtype=np.float64), "l": np.array([r.low for r in raw], dtype=np.float64),
                   "v": np.array([r.volume for r in raw], dtype=np.float64)}
    all_ds.update(r.trade_date for r in raw)
ALL_D = sorted(all_ds)
print(f"  CSI300: {len(CSI_D)} bars, Stocks: {len(STOCK)}, Dates: {len(ALL_D)}")

# ═══════════════════ 2. State machine ═══════════════════
print("\n[2] Building state machine & precomputing...")

def _sma(arr, p):
    if len(arr) < p: return np.full(len(arr), np.nan)
    o = np.full(len(arr), np.nan); cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p; return o

def _adx_full(h, l, c, p=14):
    n = len(c)
    if n < p*2: return np.full(n, np.nan)
    tr = np.zeros(n); tr[0] = h[0]-l[0]
    for i in range(1,n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1,n):
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        if up>dn and up>0: pdm[i] = up
        if dn>up and dn>0: mdm[i] = dn
    atr_a = np.full(n,np.nan); atr_a[p] = float(np.mean(tr[1:p+1]))
    for i in range(p+1,n): atr_a[i] = (tr[i]+(p-1)*atr_a[i-1])/p
    ps = float(np.mean(pdm[1:p+1])); ms = float(np.mean(mdm[1:p+1]))
    pi_a = np.full(n,np.nan); mi_a = np.full(n,np.nan)
    pi_a[p] = ps/max(atr_a[p],0.001)*100; mi_a[p] = ms/max(atr_a[p],0.001)*100
    for i in range(p+1,n):
        ps = (pdm[i]+(p-1)*ps)/p; ms = (mdm[i]+(p-1)*ms)/p
        pi_a[i] = min(ps/max(atr_a[i],0.001)*100,1000); mi_a[i] = min(ms/max(atr_a[i],0.001)*100,1000)
    dx_a = np.abs(pi_a-mi_a)/(pi_a+mi_a+1e-10)*100
    ax_a = np.full(n,np.nan)
    seed = float(np.nanmean(dx_a[p:p*2]))
    ax_a[p*2-1] = 0.0 if np.isnan(seed) else seed; ds = ax_a[p*2-1]
    for i in range(p*2,n):
        vi = dx_a[i] if not np.isnan(dx_a[i]) else ds; ds = (vi+(p-1)*ds)/p; ax_a[i] = ds
    return ax_a

def _atr_full(h, l, c, p=14):
    n = len(c)
    if n < p+1: return np.full(n, np.nan)
    tr = np.zeros(n); tr[0] = h[0]-l[0]
    for i in range(1,n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr_a = np.full(n,np.nan); atr_a[p] = float(np.mean(tr[1:p+1]))
    for i in range(p+1,n): atr_a[i] = (tr[i]+(p-1)*atr_a[i-1])/p
    return atr_a

def _breadth(dt, stock_ma_lookup):
    cnt, tot = 0, 0
    for code, (da, aa) in stock_ma_lookup.items():
        idx = np.searchsorted(da, dt, side="right")-1
        if idx < 19: continue
        tot += 1; cnt += 1 if aa[idx] else 0
    return cnt/tot*100 if tot>0 else 50

stock_ma = {}
for code, sd in STOCK.items():
    c = sd["c"]; d = sd["d"]
    if len(c) < 21: continue
    m20 = _sma(c, 20)
    stock_ma[code] = (d, c > m20)

# Full CSI300 indicators
ma120_full = _sma(CSI_C, 120)
adx_full = _adx_full(CSI_H, CSI_L, CSI_C, 14)
breadth_arr = np.array([_breadth(d, stock_ma) for d in CSI_D])
above_ma = (CSI_C > ma120_full) & (~np.isnan(ma120_full))

def build_state_map(adx_th=25, br_th=45, cbr=5, crb=2):
    n = len(CSI_C)
    raw = np.full(n, 0, dtype=int)
    for i in range(120, n):
        if above_ma[i]:
            a_ok = not np.isnan(adx_full[i]) and adx_full[i] > adx_th
            b_ok = not np.isnan(breadth_arr[i]) and breadth_arr[i] > br_th
            raw[i] = 2 if (a_ok and b_ok) else 1
    conf = np.full(n, 0, dtype=int)
    for i in range(1, n):
        rs = raw[i]
        if conf[i-1] == 0:
            if i >= cbr-1 and np.all(raw[i-cbr+1:i+1] >= 1):
                if i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2): conf[i] = 2
                else: conf[i] = 1
            else: conf[i] = 0
        elif conf[i-1] == 1:
            if rs == 0: conf[i] = 0
            elif i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2): conf[i] = 2
            else: conf[i] = 1
        elif conf[i-1] == 2:
            if rs == 0: conf[i] = 0
            elif rs == 1: conf[i] = 1
            else: conf[i] = 2
    return {CSI_D[i]: int(conf[i]) for i in range(n)}

STATE_MAP = build_state_map(25, 45, 5, 2)

# Monthly dates
def get_monthly(s, e):
    m = []
    for d in ALL_D:
        if d < s or d > e: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

def _price(code, dt):
    if code not in STOCK: return None
    sd = STOCK[code]; idx = np.searchsorted(sd["d"], dt, side="right")-1
    return sd["c"][idx] if idx >= 0 else None

def _cp(dt):
    idx = np.searchsorted(CASH_D, dt, side="right")-1
    return CASH_C[idx] if idx >= 0 else 100.0

# Precompute for MONTHLY dates only
all_months = get_monthly("20150101", "20251231")
print(f"  Precomputing for {len(all_months)} monthly dates...")
PRE = {}
for code, sd in STOCK.items():
    PRE[code] = {}
    for rd in all_months:
        idx = np.searchsorted(sd["d"], rd, side="right")
        if idx < 260: continue
        c = sd["c"][idx-260:idx]; h = sd["h"][idx-260:idx]; l = sd["l"][idx-260:idx]; v = sd["v"][idx-260:idx]
        n = len(c)
        ma120s = _sma(c, 120); above = 1.0 if (not np.isnan(ma120s[-1]) and c[-1] > ma120s[-1]) else 0.0
        rh = np.max(h[-20:]); ph = np.max(h[-60:-20]); rl = np.min(l[-20:]); pl = np.min(l[-60:-20])
        hhll = 1.0 if (rh > ph and rl > pl) else 0.0
        m20 = _sma(c, 20); m60 = _sma(c, 60)
        al = 0.0
        if (not np.isnan(m20[-1]) and not np.isnan(m60[-1]) and not np.isnan(ma120s[-1]) and m20[-1] > m60[-1] > ma120s[-1]): al = 1.0
        av = _adx_full(h, l, c, 14)[-1]; an = min(max((av-15)/35, 0), 1) if not np.isnan(av) else 0
        r3 = c[-1]/c[-63]-1 if c[-63]>1e-6 else 0; r6 = c[-1]/c[-126]-1 if c[-126]>1e-6 else 0
        m3 = min(max(r3/0.5, 0), 1) if r3 > 0 else max(r3/0.3, -1); m6 = min(max(r6/0.8, 0), 1) if r6 > 0 else max(r6/0.5, -1)
        ms = (0.5*m3+0.5*m6)*100
        w = c[-61:]; dr = np.diff(w)/(w[:-1]+1e-10); vs = (1-min(np.nanstd(dr)/0.05, 1))*100
        tc = (0.35*above+0.25*al+0.20*an+0.20*hhll)*100
        score = 0.60*ms + 0.30*tc + 0.10*vs
        atr_v = _atr_full(h, l, c, 14)[-1]; ret60 = (c[-1]/c[-60]-1) if c[-60]>1e-6 else 0
        PRE[code][rd] = {"score": float(score), "atr14": float(atr_v) if not np.isnan(atr_v) else 0.0,
                         "ret60d": float(ret60), "close": float(c[-1])}

print(f"  Precomputed {len(PRE)} stocks")

# ═══════════════════ 3. Backtest engine ═══════════════════
def run_backtest(start, end, max_pos=8, atr_mult=3.0, dual_mom=True,
                 use_atr_stop=True, min_notional=10000.0):
    """Monthly rebalance with ATR stop, dual momentum, round lots."""
    rebal = get_monthly(start, end)
    if len(rebal) < 6: return None

    cash = CAPITAL; holdings = {}; cash_etf = 0.0
    eq = [CAPITAL]; max_e = CAPITAL; prev_state = -1
    state_counts = {0:0,1:0,2:0}; trades = 0

    for rd in rebal:
        mkt = STATE_MAP.get(rd, 0); state_counts[mkt] += 1
        state_changed = (mkt != prev_state)

        # Value positions
        for sym in list(holdings.keys()):
            p = _price(sym, rd)
            if p is None or p < 0.01:
                cash += holdings[sym].get("qty",0)*holdings[sym].get("price",10)*0.7
                del holdings[sym]; continue
            holdings[sym]["price"] = p

        cp = _cp(rd); cval = cash_etf * cp
        total = cash + cval + sum(h["qty"]*h.get("price",0) for h in holdings.values())

        # ── ATR Trailing Stop ──
        if use_atr_stop:
            for sym in list(holdings.keys()):
                p = holdings[sym].get("price", 0.0)
                if p <= 0: continue
                hwm = holdings[sym].get("hwm", p)
                if p > hwm: holdings[sym]["hwm"] = p
                atr_s = 0.0
                if sym in PRE and rd in PRE[sym]:
                    atr_s = PRE[sym][rd].get("atr14", 0.0)
                if atr_s > 0 and hwm > 0:
                    if p < hwm - atr_mult * atr_s:
                        cash += holdings[sym]["qty"]*p*(1-COMM); del holdings[sym]; trades += 1
        else:
            for sym in list(holdings.keys()):
                p = holdings[sym].get("price", 0.0)
                if p <= 0: continue
                hwm = holdings[sym].get("hwm", p)
                if p > hwm: holdings[sym]["hwm"] = p
                if hwm > 0 and (p/hwm-1)*100 < -10:
                    cash += holdings[sym]["qty"]*p*(1-COMM); del holdings[sym]; trades += 1

        total = cash + cash_etf*cp + sum(h["qty"]*h.get("price",0) for h in holdings.values())
        if total > max_e: max_e = total

        # DD breaker
        dd = (max_e-total)/max_e if max_e > 0 else 0
        if dd > 0.15:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"]*holdings[sym].get("price",0)*(1-COMM); del holdings[sym]
            cash += cash_etf*cp*(1-COMM); cash_etf=0; trades += 1
            total = cash

        # BEAR
        if mkt == 0:
            if state_changed or not cash_etf:
                for sym in list(holdings.keys()):
                    cash += holdings[sym]["qty"]*holdings[sym].get("price",0)*(1-COMM); del holdings[sym]
                cash += cash_etf*cp*(1-COMM); cash_etf=0
                if cp > 0 and cash > 0: cash_etf = cash/cp; cash = 0.0; trades += 1
            eq.append(cash + cash_etf*cp); prev_state = mkt; continue

        if cash_etf > 0: cash += cash_etf*cp*(1-COMM); cash_etf = 0.0; trades += 1

        # Select stocks
        pos_size = 0.5 if mkt == 1 else 1.0
        scored = []
        for code in PRE:
            if rd not in PRE[code]: continue
            info = PRE[code][rd]
            if dual_mom and info["ret60d"] <= 0: continue
            scored.append((code, info["score"]))
        if not scored: eq.append(total); prev_state = mkt; continue
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_codes = [s[0] for s in scored[:max_pos]]

        for sym in list(holdings.keys()):
            if sym not in selected_codes:
                cash += holdings[sym]["qty"]*holdings[sym].get("price",0)*(1-COMM); del holdings[sym]; trades += 1

        n_pos = len(selected_codes)
        if n_pos == 0: eq.append(total); prev_state = mkt; continue

        per_pos = (total * pos_size) / n_pos * 0.90
        for sym in selected_codes:
            p = _price(sym, rd)
            if p is None or p < 0.01: continue
            tq = int(per_pos/p/100)*100  # round lot
            if tq < 100: continue
            if tq * p < min_notional: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff > 0 and cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym]["qty"]=tq; trades += 1
                    elif diff < 0: cash += cost*(1-COMM); holdings[sym]["qty"]=tq; trades += 1
            else:
                cost = tq*p
                if cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym]={"qty":tq,"price":p,"hwm":p}; trades += 1

        leftover = cash; cp = _cp(rd)
        if cp > 0 and leftover > 0: cash_etf = leftover/cp; cash = 0.0
        else: cash = leftover; cash_etf = 0.0
        total = cash + cash_etf*cp + sum(h["qty"]*h.get("price",0) for h in holdings.values())
        eq.append(total); prev_state = mkt

    eq_arr = np.array(eq)
    if len(eq_arr) < 2 or eq_arr[0] <= 0: return None
    n_y = len(eq_arr)/12.0
    if n_y < 0.5: return None
    cagr = ((eq_arr[-1]/eq_arr[0])**(1/n_y)-1)*100
    mr = np.diff(eq_arr)/(eq_arr[:-1]+1e-10)
    sharpe = np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12) if len(mr) > 1 else 0
    peak = eq_arr[0]; maxdd = 0.0
    for v in eq_arr:
        if v > peak: peak = v
        if (peak-v)/peak*100 > maxdd: maxdd = (peak-v)/peak*100
    total_ret = (eq_arr[-1]/eq_arr[0]-1)*100
    win_ret = [(eq_arr[i]-eq_arr[i-1])/max(eq_arr[i-1],1e-10) for i in range(1,len(eq_arr))]
    win_rate = sum(1 for r in win_ret if r > 0)/len(win_ret)*100 if win_ret else 0
    return {"cagr":cagr,"sharpe":sharpe,"maxdd":maxdd,"total_ret":total_ret,
            "trades":trades,"win_rate":win_rate,"state_counts":state_counts,"n_months":len(rebal)}


# ═══════════════════ 4. Run All Tests ═══════════════════
print("\n" + "=" * 90)
print("STEP 1: Baseline — V1 best params, no deployment changes")
print("=" * 90)

# This is the V1-equivalent: monthly, fixed -10% stop, Top-5 BULL/3 RANGE, no dual mom
def run_v1_baseline(start, end):
    """Research V1 equivalent: fixed -10% stop, no dual momentum, Top-5/3."""
    rebal = get_monthly(start, end)
    if len(rebal) < 6: return None
    cash = CAPITAL; holdings = {}; cash_etf = 0.0
    eq = [CAPITAL]; max_e = CAPITAL; prev_state = -1; sc = {0:0,1:0,2:0}
    for rd in rebal:
        mkt = STATE_MAP.get(rd, 0); sc[mkt] += 1; scd = (mkt != prev_state)
        for sym in list(holdings.keys()):
            p = _price(sym, rd)
            if p is None or p < 0.01: cash += holdings[sym].get("qty",0)*10*0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p
        cp = _cp(rd); cv = cash_etf*cp; total = cash + cv + sum(h["qty"]*h.get("price",0) for h in holdings.values())
        for sym in list(holdings.keys()):
            p = holdings[sym].get("price",0)
            if p <= 0: continue
            hwm = holdings[sym].get("hwm",p)
            if p > hwm: holdings[sym]["hwm"] = p
            if hwm > 0 and (p/hwm-1)*100 < -10: cash += holdings[sym]["qty"]*p*(1-COMM); del holdings[sym]
        total = cash + cash_etf*cp + sum(h["qty"]*h.get("price",0) for h in holdings.values())
        if total > max_e: max_e = total
        if mkt == 0:
            if scd or not cash_etf:
                for sym in list(holdings.keys()): cash += holdings[sym]["qty"]*holdings[sym].get("price",0)*(1-COMM); del holdings[sym]
                cash += cash_etf*cp*(1-COMM); cash_etf=0
                if cp > 0 and cash > 0: cash_etf = cash/cp; cash = 0.0
            eq.append(cash+cash_etf*cp); prev_state = mkt; continue
        if cash_etf: cash += cash_etf*cp*(1-COMM); cash_etf = 0.0
        pos_size = 0.5 if mkt == 1 else 1.0; top_n = 3 if mkt == 1 else 5
        scored = [(cd, PRE[cd][rd]["score"]) for cd in PRE if rd in PRE[cd]]
        if not scored: eq.append(total); prev_state = mkt; continue
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = {s[0] for s in scored[:top_n]}
        for sym in list(holdings.keys()):
            if sym not in selected: cash += holdings[sym]["qty"]*holdings[sym].get("price",0)*(1-COMM); del holdings[sym]
        n_pos = max(len(selected),1); per_s = total*pos_size/n_pos*0.90
        for sym in selected:
            p = _price(sym, rd)
            if p is None or p < 0.01: continue
            tq = int(per_s/p/100)*100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff > 0 and cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym]["qty"]=tq
                    elif diff < 0: cash += cost*(1-COMM); holdings[sym]["qty"]=tq
            else:
                cost = tq*p
                if cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym] = {"qty":tq,"price":p,"hwm":p}
        leftover = cash
        if cp > 0 and leftover > 0: cash_etf = leftover/cp; cash = 0.0
        else: cash = leftover; cash_etf = 0.0
        total = cash + cash_etf*cp + sum(h["qty"]*h.get("price",0) for h in holdings.values())
        eq.append(total); prev_state = mkt
    eq_arr = np.array(eq)
    if len(eq_arr) < 2: return None
    n_y = len(eq_arr)/12.0; cagr = ((eq_arr[-1]/eq_arr[0])**(1/n_y)-1)*100
    mr = np.diff(eq_arr)/(eq_arr[:-1]+1e-10); sh = np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12) if len(mr)>1 else 0
    peak=eq_arr[0]; mdd=0.0
    for v in eq_arr:
        if v>peak: peak=v
        if (peak-v)/peak*100>mdd: mdd=(peak-v)/peak*100
    return {"cagr":cagr,"sharpe":sh,"maxdd":mdd,"total_ret":(eq_arr[-1]/eq_arr[0]-1)*100,"sc":sc}

v1 = run_v1_baseline("20220101","20251231")
v1_tr = run_v1_baseline("20150101","20211231")
if v1 and v1_tr:
    print(f"  V1 Baseline (fixed -10% stop, Top-5/3, no dual mom):")
    print(f"    Train: CAGR={v1_tr['cagr']:+.1f}% DD={v1_tr['maxdd']:.1f}% Sh={v1_tr['sharpe']:.3f}")
    print(f"    Test:  CAGR={v1['cagr']:+.1f}% DD={v1['maxdd']:.1f}% Sh={v1['sharpe']:.3f} TotRet={v1['total_ret']:+.1f}%")

print("\n" + "=" * 90)
print("STEP 2: Component Ablation — Add deployment changes one at a time")
print("=" * 90)

configs = [
    ("V1 Baseline",         dict(mp=5, am=-10, dm=False, atr=False)),
    ("+MaxPos 8",           dict(mp=8, am=-10, dm=False, atr=False)),
    ("+Dual Momentum",      dict(mp=8, am=-10, dm=True,  atr=False)),
    ("+ATR 3x Stop",        dict(mp=8, am=3.0, dm=True,  atr=True)),
    ("+ATR 2x Stop",        dict(mp=8, am=2.0, dm=True,  atr=True)),
    ("ALL (MaxPos8+ATR3x+DM)", dict(mp=8, am=3.0, dm=True, atr=True)),
]

print(f"{'Config':<30s} {'Test CAGR':>10s} {'MaxDD':>10s} {'Sharpe':>10s} {'Trades':>8s} | {'Train CAGR':>10s}")
print("-" * 95)
for name, cfg in configs:
    r = run_backtest("20220101", "20251231", max_pos=cfg["mp"],
                     atr_mult=cfg["am"] if cfg["atr"] else 999,
                     dual_mom=cfg["dm"], use_atr_stop=cfg["atr"])
    r_tr = run_backtest("20150101", "20211231", max_pos=cfg["mp"],
                        atr_mult=cfg["am"] if cfg["atr"] else 999,
                        dual_mom=cfg["dm"], use_atr_stop=cfg["atr"])
    if r and r_tr:
        print(f"{name:<30s} {r['cagr']:>+9.2f}% {r['maxdd']:>9.1f}% {r['sharpe']:>10.3f} {r['trades']:>8d} | "
              f"{r_tr['cagr']:>+9.2f}%")

print("\n" + "=" * 90)
print("STEP 3: ATR stop multiplier sweep (optimal deployment params)")
print("=" * 90)

for am in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    r = run_backtest("20220101", "20251231", max_pos=8, atr_mult=am, dual_mom=True, use_atr_stop=True)
    r_tr = run_backtest("20150101", "20211231", max_pos=8, atr_mult=am, dual_mom=True, use_atr_stop=True)
    if r and r_tr:
        print(f"  ATR {am:.1f}x: Test C={r['cagr']:+.1f}% D={r['maxdd']:.1f}% Sh={r['sharpe']:.3f} "
              f"Tr={r['trades']} | Train C={r_tr['cagr']:+.1f}% D={r_tr['maxdd']:.1f}%")

# CSI300 benchmark
si=np.searchsorted(CSI_D,"20220101"); ei=np.searchsorted(CSI_D,"20251231",side="right")-1
seg=CSI_C[si:ei+1]; cagr_bm=((seg[-1]/seg[0])**(1/4)-1)*100
peak_bm=seg[0]; dd_bm=0.0
for v in seg:
    if v>peak_bm: peak_bm=v
    if (peak_bm-v)/peak_bm*100>dd_bm: dd_bm=(peak_bm-v)/peak_bm*100

print(f"\n{'='*90}")
print("FINAL DEPLOYMENT RECOMMENDATION")
print(f"{'='*90}")
print(f"  State machine params:  ADX=25  Breadth=45%  N_BR=5  N_RB=2")
print(f"  Position sizing:       Max 8 positions, equal notional, round lots (100 shares)")
print(f"  ATR stop:              3x ATR(14) trailing from HWM (wider than -10% fixed)")
print(f"  Dual momentum:         YES (only enter if ret_60d > 0)")
print(f"  Cash vehicle:          511880 money market ETF")
print(f"  Rebalancing:           Monthly (first trading day)")
print(f"  Min notional:          10,000 CNY per position")
print(f"\n  CSI300 B&H:    CAGR={cagr_bm:+.1f}%  MaxDD={dd_bm:.1f}%  TotRet={(seg[-1]/seg[0]-1)*100:+.1f}%")

print(f"\nTotal time: {time.time()-T0:.0f}s")

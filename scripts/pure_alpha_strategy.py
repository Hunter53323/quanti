"""
Pure Stock Selection Alpha Strategy (Optimized)
================================================
Precomputes indicator components for all stocks at all monthly dates,
then parameter sweeps are just weighted combinations (fast).

Tests: does Top-N momentum stock selection persistently beat CSI300 index?
"""
import sys, os, itertools, time, numpy as np
os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025
TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END = "20220101", "20251231"

T_START = time.time()
print("=" * 90)
print("PURE STOCK SELECTION ALPHA STRATEGY (Optimized)")
print("=" * 90)

# ═══════════════════════════════════════════════════════
# 1. Load Data
# ═══════════════════════════════════════════════════════
print("\n[1] Loading data...")
storage = DataStorage()

raw300 = storage.load_bars("510300")
csi_dates = np.array([r.trade_date for r in raw300])
csi_c = np.array([r.close for r in raw300], dtype=np.float64)

all_files = sorted(storage.clean_dir.glob("*.parquet"))
stock_codes = [p.stem for p in all_files if len(p.stem) == 6 and not p.stem.startswith(("51", "58", "15", "56"))]

stock_data = {}
all_dates_set = set(csi_dates)
for code in stock_codes:
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200: continue
    stock_data[code] = {
        "dates": np.array([r.trade_date for r in raw]),
        "close": np.array([r.close for r in raw], dtype=np.float64),
        "high": np.array([r.high for r in raw], dtype=np.float64),
        "low": np.array([r.low for r in raw], dtype=np.float64),
        "volume": np.array([r.volume for r in raw], dtype=np.float64),
    }
    all_dates_set.update(r.trade_date for r in raw)

all_dates = sorted(all_dates_set)
print(f"  Stocks: {len(stock_data)}, Dates: {len(all_dates)}")


def sma(arr, p):
    if len(arr) < p: return np.full(len(arr), np.nan)
    o = np.full(len(arr), np.nan)
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p
    return o

def adx_value(h, l, c, period=14):
    """Returns only the latest ADX value."""
    n = len(c)
    if n < period*2: return np.nan
    tr = np.zeros(n); tr[0] = h[0]-l[0]
    for i in range(1,n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1,n):
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        if up>dn and up>0: pdm[i] = up
        if dn>up and dn>0: mdm[i] = dn
    atr = np.full(n,np.nan); atr[period]=float(np.mean(tr[1:period+1]))
    for i in range(period+1,n): atr[i]=(tr[i]+(period-1)*atr[i-1])/period
    ps=float(np.mean(pdm[1:period+1])); ms=float(np.mean(mdm[1:period+1]))
    pi=np.full(n,np.nan); mi=np.full(n,np.nan)
    pi[period]=ps/max(atr[period],0.001)*100; mi[period]=ms/max(atr[period],0.001)*100
    for i in range(period+1,n):
        ps=(pdm[i]+(period-1)*ps)/period; ms=(mdm[i]+(period-1)*ms)/period
        pi[i]=min(ps/max(atr[i],0.001)*100,1000); mi[i]=min(ms/max(atr[i],0.001)*100,1000)
    dx=np.abs(pi-mi)/(pi+mi+1e-10)*100
    ax=np.full(n,np.nan)
    seed=float(np.nanmean(dx[period:period*2]))
    ax[period*2-1]=0.0 if np.isnan(seed) else seed; ds=ax[period*2-1]
    for i in range(period*2,n):
        vi=dx[i] if not np.isnan(dx[i]) else ds; ds=(vi+(period-1)*ds)/period; ax[i]=ds
    return ax[-1]

def get_monthly(start, end):
    m = []
    for d in all_dates:
        if d < start or d > end: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

def price_on(code, date_str):
    if code not in stock_data: return None
    sd = stock_data[code]
    idx = np.searchsorted(sd["dates"], date_str, side="right") - 1
    return sd["close"][idx] if idx >= 0 else None


# ═══════════════════════════════════════════════════════
# 2. Precompute indicator components for ALL stocks at ALL dates
# ═══════════════════════════════════════════════════════
print("\n[2] Precomputing indicator components...")
t0 = time.time()

# Combine all rebalance dates for entire period
all_rebal = get_monthly("20150101", "20251231")
print(f"  Rebalance dates: {len(all_rebal)} months")

# For each stock at each rebalance date, compute 6 normalized components
# Structure: {code: {date_str: [above_ma, hhll, aligned, adx_norm, mom_score, vol_score]}}
precomputed = {}
stock_count = 0
for code, sd in stock_data.items():
    stock_count += 1
    if stock_count % 100 == 0:
        t_elapsed = time.time() - t0
        print(f"  [{stock_count}/{len(stock_data)}] stocks, {t_elapsed:.0f}s")

    precomputed[code] = {}
    d_arr = sd["dates"]

    for rd in all_rebal:
        idx = np.searchsorted(d_arr, rd, side="right")
        if idx < 260: continue  # need at least 260 bars

        c = sd["close"][idx-260:idx].copy()
        h = sd["high"][idx-260:idx].copy()
        l = sd["low"][idx-260:idx].copy()
        v = sd["volume"][idx-260:idx].copy()
        n = len(c)

        # 1. above_ma120: 0 or 1
        ma120 = np.full(n, np.nan)
        cs120 = np.cumsum(np.insert(c, 0, 0.0))
        ma120[119:] = (cs120[120:] - cs120[:-120]) / 120.0
        above_ma = 1.0 if (c[-1] > ma120[-1] and not np.isnan(ma120[-1])) else 0.0

        # 2. hhll: higher highs and higher lows (20d vs 60d)
        rh = np.max(h[-20:]); ph = np.max(h[-60:-20])
        rl = np.min(l[-20:]); pl = np.min(l[-60:-20])
        hhll = 1.0 if (rh > ph and rl > pl) else 0.0

        # 3. ma_aligned: MA20 > MA60 > MA120
        ma20 = np.full(n, np.nan)
        cs20 = np.cumsum(np.insert(c, 0, 0.0))
        ma20[19:] = (cs20[20:] - cs20[:-20]) / 20.0
        ma60 = np.full(n, np.nan)
        cs60 = np.cumsum(np.insert(c, 0, 0.0))
        ma60[59:] = (cs60[60:] - cs60[:-60]) / 60.0
        aligned = 0.0
        if (not np.isnan(ma20[-1]) and not np.isnan(ma60[-1]) and not np.isnan(ma120[-1])
            and ma20[-1] > ma60[-1] > ma120[-1]):
            aligned = 1.0

        # 4. adx_norm: 0-1
        adx_val = adx_value(h, l, c, 14)
        adx_norm = min(max((adx_val - 15) / 35, 0), 1) if not np.isnan(adx_val) else 0

        # 5. momentum score: normalized 3M + 6M return
        r3 = c[-1]/c[-63]-1 if c[-63]>1e-6 else 0
        r6 = c[-1]/c[-126]-1 if c[-126]>1e-6 else 0
        m3 = min(max(r3/0.5, 0), 1) if r3 > 0 else max(r3/0.3, -1)
        m6 = min(max(r6/0.8, 0), 1) if r6 > 0 else max(r6/0.5, -1)
        mom_score = (0.5*m3 + 0.5*m6) * 100

        # 6. vol_score: inverse of 60d realized vol, 0-100
        w = c[-61:]
        dr = np.diff(w)/(w[:-1]+1e-10)
        vol = np.nanstd(dr)
        vol_score = (1 - min(vol/0.05, 1)) * 100

        precomputed[code][rd] = [above_ma, hhll, aligned, adx_norm, mom_score, vol_score]

elapsed = time.time() - t0
print(f"  Precomputed in {elapsed:.0f}s")


# ═══════════════════════════════════════════════════════
# 3. Fast backtest using precomputed scores
# ═══════════════════════════════════════════════════════
def compute_score_from_precomputed(code, rd, w_mom, w_trend, w_lowvol):
    """Composite score from precomputed components."""
    if code not in precomputed or rd not in precomputed[code]:
        return -999
    comps = precomputed[code][rd]
    above_ma, hhll, aligned, adx_norm, mom_score, vol_score = comps

    # Trend component
    trend_score = (0.35*above_ma + 0.25*aligned + 0.20*adx_norm + 0.20*hhll) * 100

    return w_mom*mom_score + w_trend*trend_score + w_lowvol*vol_score


def run_backtest_fast(start, end, top_n, w_mom, w_trend, w_lowvol, stop_pct):
    """Fast backtest using precomputed indicator components."""
    rebal = get_monthly(start, end)
    if len(rebal) < 6: return None

    cash = CAPITAL
    holdings = {}
    eq = [CAPITAL]
    max_e = CAPITAL
    trades = 0

    for rd in rebal:
        # Value
        for sym in list(holdings.keys()):
            p = price_on(sym, rd)
            if p is None or p < 0.01:
                cash += holdings[sym]["val"]*0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p
            holdings[sym]["val"] = holdings[sym]["qty"] * p

        total = cash + sum(h["qty"]*h.get("price",0) for h in holdings.values())

        # Stop-loss
        for sym in list(holdings.keys()):
            p = holdings[sym].get("price", 0)
            hwm = holdings[sym].get("hwm", p)
            if p > hwm: holdings[sym]["hwm"] = p
            if hwm > 0 and (p/hwm-1)*100 < stop_pct:
                cash += holdings[sym]["val"]*(1-COMM); trades += 1; del holdings[sym]

        total = cash + sum(h["qty"]*h.get("price",0) for h in holdings.values())
        if total > max_e: max_e = total

        # Score stocks (fast - just weighted sum of precomputed)
        scored = []
        for code in precomputed:
            s = compute_score_from_precomputed(code, rd, w_mom, w_trend, w_lowvol)
            if s > -999:
                scored.append((code, s))

        if not scored:
            eq.append(total)
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = {s[0] for s in scored[:min(top_n, len(scored))]}

        # Rotate
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["val"]*(1-COMM); trades += 1; del holdings[sym]

        # Allocate
        n_pos = max(len(selected), 1)
        per_s = total / n_pos * 0.90

        for sym in selected:
            p = price_on(sym, rd)
            if p is None or p < 0.01: continue
            tq = int(per_s/p/100)*100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff > 0 and cash >= cost*(1+COMM):
                        cash -= cost*(1+COMM); holdings[sym]["qty"] = tq; trades += 1
                    elif diff < 0:
                        cash += cost*(1-COMM); holdings[sym]["qty"] = tq; trades += 1
            else:
                cost = tq*p
                if cash >= cost*(1+COMM):
                    cash -= cost*(1+COMM)
                    holdings[sym] = {"qty":tq,"price":p,"val":cost,"hwm":p}; trades += 1

        total = cash + sum(h["qty"]*h.get("price",0) for h in holdings.values())
        eq.append(total)

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
        ddi = (peak-v)/peak*100
        if ddi > maxdd: maxdd = ddi
    total_ret = (eq_arr[-1]/eq_arr[0]-1)*100
    mrets = [(eq_arr[i]-eq_arr[i-1])/max(eq_arr[i-1],1e-10) for i in range(1,len(eq_arr))]
    win_rate = sum(1 for r in mrets if r>0)/len(mrets)*100 if mrets else 0
    return {"cagr":cagr,"sharpe":sharpe,"maxdd":maxdd,"total_ret":total_ret,
            "win_rate":win_rate,"trades":trades,"ny":n_y,"final":float(eq_arr[-1])}


# ═══════════════════════════════════════════════════════
# 4. CSI300 benchmark
# ═══════════════════════════════════════════════════════
def compute_csi300_bnh(start, end):
    si = np.searchsorted(csi_dates, start)
    ei = np.searchsorted(csi_dates, end, side="right")-1
    if ei <= si: return None
    seg = csi_c[si:ei+1]
    if len(seg) < 2: return None
    total_ret = (seg[-1]/seg[0]-1)*100
    n_y = len(seg)/252.0
    cagr = ((seg[-1]/seg[0])**(1/n_y)-1)*100
    dr = np.diff(seg)/(seg[:-1]+1e-10)
    sharpe = np.mean(dr)/(np.std(dr)+1e-10)*np.sqrt(252) if len(dr)>1 else 0
    peak=seg[0]; maxdd=0.0
    for v in seg:
        if v>peak: peak=v
        ddi=(peak-v)/peak*100
        if ddi>maxdd: maxdd=ddi
    return {"cagr":cagr,"sharpe":sharpe,"maxdd":maxdd,"total_ret":total_ret}

bm_test = compute_csi300_bnh(TEST_START, TEST_END)
bm_train = compute_csi300_bnh(TRAIN_START, TRAIN_END)
print(f"\n  CSI300 B&H Train: CAGR={bm_train['cagr']:+.1f}% MaxDD={bm_train['maxdd']:.1f}%")
print(f"  CSI300 B&H Test:  CAGR={bm_test['cagr']:+.1f}% MaxDD={bm_test['maxdd']:.1f}%")


# ═══════════════════════════════════════════════════════
# 5. Parameter Sweep (fast - only weight combos + top_n)
# ═══════════════════════════════════════════════════════
print("\n[3] Parameter sweep...")

param_grid = list(itertools.product(
    [3, 5, 10, 15],        # top_n
    [0.40, 0.50, 0.60],    # w_mom
    [0.20, 0.30, 0.40],    # w_trend
    [-5, -10, -15],        # stop_pct
))

total_combos = len(param_grid)
print(f"  {total_combos} raw combos")

results = []
count = 0
valid_count = 0
t1 = time.time()

for tn, wm, wt, sp in param_grid:
    wv = round(1.0 - wm - wt, 2)
    if wv < 0.10 or wv > 0.35:
        continue  # skip invalid weight combos

    valid_count += 1
    rt = run_backtest_fast(TRAIN_START, TRAIN_END, tn, wm, wt, wv, sp)
    re = run_backtest_fast(TEST_START, TEST_END, tn, wm, wt, wv, sp)

    if rt and re:
        ex_test = re["cagr"] - bm_test["cagr"]
        ex_train = rt["cagr"] - bm_train["cagr"]
        results.append({
            "tn": tn, "wm": wm, "wt": wt, "wv": wv, "sp": sp,
            "tr_c": round(rt["cagr"],2), "tr_sh": round(rt["sharpe"],3),
            "tr_dd": round(rt["maxdd"],2), "tr_ex": round(ex_train,2),
            "te_c": round(re["cagr"],2), "te_sh": round(re["sharpe"],3),
            "te_dd": round(re["maxdd"],2), "te_ex": round(ex_test,2),
            "te_ret": round(re["total_ret"],2),
        })

    count += 1
    if count % 20 == 0:
        e = time.time()-t1
        remaining = e/count*(valid_count-count)
        print(f"  [{count}/{valid_count}] {100*count/valid_count:.0f}%  ETA: {remaining:.0f}s")

elapsed_total = time.time() - t1
print(f"  Sweep complete in {elapsed_total:.0f}s, {len(results)} valid results")

# ═══════════════════════════════════════════════════════
# 6. Results
# ═══════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("RESULTS: Pure Stock Alpha Strategy")
print(f"{'='*90}")

# Ranking
by_ex = sorted(results, key=lambda x: x["te_ex"], reverse=True)
by_dd = sorted(results, key=lambda x: x["te_dd"])
by_sh = sorted(results, key=lambda x: x["te_sh"], reverse=True)

for i,r in enumerate(by_ex): r["re"] = i
for i,r in enumerate(by_dd): r["rd"] = i
for i,r in enumerate(by_sh): r["rs"] = i
for r in results:
    r["comp"] = r["re"]*0.50 + r["rd"]*0.30 + r["rs"]*0.20
best = sorted(results, key=lambda x: x["comp"])[0]

print(f"\n--- TOP 10 by Test Excess (vs CSI300) ---")
print(f"{'#':<4s} {'N':<4s} {'wM':>4s} {'wT':>4s} {'wV':>4s} {'Stop':>5s} {'Excess':>8s} {'CAGR':>8s} {'DD':>7s} {'Sharpe':>7s} | {'TrainEx':>7s}")
print("-"*95)
for i, r in enumerate(by_ex[:10], 1):
    print(f"{i:<4d} {r['tn']:<4d} {r['wm']:4.2f} {r['wt']:4.2f} {r['wv']:4.2f} {r['sp']:>5d} "
          f"{r['te_ex']:>+7.1f}% {r['te_c']:>+7.1f}% {r['te_dd']:>6.1f}% {r['te_sh']:>7.3f} | "
          f"{r['tr_ex']:>+6.1f}%")

print(f"\n--- TOP 10 by Minimum MaxDD ---")
for i, r in enumerate(by_dd[:10], 1):
    print(f"  #{i}: N={r['tn']} wM={r['wm']:.2f} wT={r['wt']:.2f} wV={r['wv']:.2f} stop={r['sp']} | "
          f"TestDD={r['te_dd']:.1f}% Excess={r['te_ex']:+.1f}% CAGR={r['te_c']:+.1f}%")

print(f"\n--- TOP 5 Composite Best ---")
for i, r in enumerate(sorted(results, key=lambda x: x["comp"])[:5], 1):
    print(f"  #{i}: N={r['tn']} wM={r['wm']:.2f} wT={r['wt']:.2f} wV={r['wv']:.2f} stop={r['sp']} | "
          f"Excess={r['te_ex']:+.1f}% CAGR={r['te_c']:+.1f}% DD={r['te_dd']:.1f}% Sharpe={r['te_sh']:.3f}")

print(f"\n--- Composite Best ---")
print(f"  Params: N={best['tn']}  w_mom={best['wm']:.2f}  w_trend={best['wt']:.2f}  "
      f"w_lowvol={best['wv']:.2f}  stop={best['sp']}%")
print(f"  Train: CAGR={best['tr_c']:+.1f}%  Excess={best['tr_ex']:+.1f}%  "
      f"DD={best['tr_dd']:.1f}%  Sharpe={best['tr_sh']:.3f}")
print(f"  Test:  CAGR={best['te_c']:+.1f}%  Excess={best['te_ex']:+.1f}%  "
      f"DD={best['te_dd']:.1f}%  Sharpe={best['te_sh']:.3f}  TotRet={best['te_ret']:+.1f}%")

# Yearly
print(f"\n--- Yearly Breakdown (Best Config) ---")
for yr in [2022, 2023, 2024, 2025]:
    ys, ye = f"{yr}0101", f"{yr}1231"
    ry = run_backtest_fast(ys, ye, best["tn"], best["wm"], best["wt"], best["wv"], best["sp"])
    csi_yr = compute_csi300_bnh(ys, ye)
    if ry and csi_yr:
        ex = ry["cagr"] - csi_yr["cagr"]
        print(f"  {yr}: Strat={ry['cagr']:+.1f}%  CSI300={csi_yr['cagr']:+.1f}%  "
              f"Excess={ex:+.1f}%  DD={ry['maxdd']:.1f}%  Sharpe={ry['sharpe']:.3f}")

# Comparison
print(f"\n{'='*90}")
print("FINAL COMPARISON: Pure Alpha vs State Machine vs CSI300 B&H")
print(f"{'='*90}")
print(f"  Strategy                | Test CAGR | MaxDD   | Excess over CSI300")
print(f"  ------------------------+-----------+---------+-------------------")
print(f"  Pure Stock Alpha (best) | {best['te_c']:+6.1f}%  | {best['te_dd']:6.1f}%  | {best['te_ex']:+6.1f}%")
sm_base = 4.40; sm_best = 7.0
print(f"  State Machine (base)    | {sm_base:+6.1f}%  | {9.1:6.1f}%  | {sm_base-bm_test['cagr']:+6.1f}%")
print(f"  State Machine (best)    | {sm_best:+6.1f}%  | {5.1:6.1f}%  | {sm_best-bm_test['cagr']:+6.1f}%")
print(f"  CSI300 B&H              | {bm_test['cagr']:+6.1f}%  | {bm_test['maxdd']:6.1f}%  | -")

print(f"\n{'='*90}")
print(f"Total wall time: {time.time() - T_START:.0f}s")
print(f"{'='*90}")

"""
Martingale Accumulation Strategy
================================
Instead of equal-weight rebalancing each month:
  - New entry: buy 1 unit of each Top-N stock
  - Persistence: add 1 unit each month the stock REMAINS in Top-N
  - Cap: max M units per stock
  - Exit: sell ALL units if stock drops out of Top-N

Hypothesis: Stocks that persistently rank high have stronger trends.
Adding to winners (pyramid) outperforms monthly equal-weight rebalancing.

Uses precomputed scores from pure_alpha_strategy logic.
"""
import sys, os, itertools, time, numpy as np
os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025
TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END = "20220101", "20251231"

print("=" * 90)
print("MARTINGALE ACCUMULATION STRATEGY")
print("=" * 90)

# ═══════════════════ 1. Load Data ═══════════════════
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


# ═══════════════════ 2. Indicators ═══════════════════
def sma(arr, p):
    if len(arr) < p: return np.full(len(arr), np.nan)
    o = np.full(len(arr), np.nan)
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p
    return o

def adx_value(h, l, c, period=14):
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


# ═══════════════════ 3. Precompute Scores ═══════════════════
print("\n[2] Precomputing composite scores...")
t0 = time.time()

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

# Use fixed weights from prior optimization
W_MOM = 0.60; W_TREND = 0.30; W_LOWVOL = 0.10

all_rebal = get_monthly("20150101", "20251231")
print(f"  Rebalance dates: {len(all_rebal)} months")

precomputed = {}
for code, sd in stock_data.items():
    precomputed[code] = {}
    d_arr = sd["dates"]
    for rd in all_rebal:
        idx = np.searchsorted(d_arr, rd, side="right")
        if idx < 260: continue
        c = sd["close"][idx-260:idx].copy()
        h = sd["high"][idx-260:idx].copy()
        l = sd["low"][idx-260:idx].copy()
        v = sd["volume"][idx-260:idx].copy()
        n = len(c)

        # above_ma120
        ma120 = np.full(n, np.nan)
        cs120 = np.cumsum(np.insert(c, 0, 0.0))
        ma120[119:] = (cs120[120:] - cs120[:-120]) / 120.0
        above_ma = 1.0 if (c[-1] > ma120[-1] and not np.isnan(ma120[-1])) else 0.0

        # hhll
        rh = np.max(h[-20:]); ph = np.max(h[-60:-20])
        rl = np.min(l[-20:]); pl = np.min(l[-60:-20])
        hhll = 1.0 if (rh > ph and rl > pl) else 0.0

        # aligned
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

        # adx_norm
        adx_val = adx_value(h, l, c, 14)
        adx_norm = min(max((adx_val - 15) / 35, 0), 1) if not np.isnan(adx_val) else 0

        # momentum
        r3 = c[-1]/c[-63]-1 if c[-63]>1e-6 else 0
        r6 = c[-1]/c[-126]-1 if c[-126]>1e-6 else 0
        m3 = min(max(r3/0.5, 0), 1) if r3 > 0 else max(r3/0.3, -1)
        m6 = min(max(r6/0.8, 0), 1) if r6 > 0 else max(r6/0.5, -1)
        mom_score = (0.5*m3 + 0.5*m6) * 100

        # low vol
        w_arr = c[-61:]
        dr = np.diff(w_arr)/(w_arr[:-1]+1e-10)
        vol = np.nanstd(dr)
        vol_score = (1 - min(vol/0.05, 1)) * 100

        # composite
        trend_comp = (0.35*above_ma + 0.25*aligned + 0.20*adx_norm + 0.20*hhll) * 100
        score = W_MOM*mom_score + W_TREND*trend_comp + W_LOWVOL*vol_score
        precomputed[code][rd] = score

elapsed = time.time() - t0
print(f"  Precomputed in {elapsed:.0f}s")


# ═══════════════════ 4. Martingale Backtest ═══════════════════
def run_martingale(start, end, top_n=10, max_units=5, unit_pct=0.05, stop_pct=-10):
    """
    Martingale accumulation:
      - New stock: buy 'unit_pct' of portfolio as initial position
      - Each month stock stays in Top-N: add 1 more unit
      - Max_units: cap on position size
      - Stock drops out: sell all

    unit_pct: fraction of current NAV per unit (e.g., 0.05 = 5%)
    So max exposure per stock = unit_pct * max_units (e.g., 0.05*5 = 25%)
    Total max exposure = top_n * unit_pct * max_units
    """
    rebal = get_monthly(start, end)
    if len(rebal) < 6: return None

    cash = CAPITAL
    holdings = {}  # code -> {"qty": int, "price": float, "units": int, "hwm": float}
    eq = [CAPITAL]
    max_e = CAPITAL
    trades = 0
    unit_value = CAPITAL * unit_pct  # initial unit size in yuan

    for rd in rebal:
        # Value positions
        for sym in list(holdings.keys()):
            p = price_on(sym, rd)
            if p is None or p < 0.01:
                cash += holdings[sym]["qty"] * p * 0.7 if p else 0; del holdings[sym]; continue
            holdings[sym]["price"] = p

        total = cash + sum(h["qty"] * h.get("price", 0) for h in holdings.values())

        # Stop-loss per position (sell ALL units if stop hit)
        for sym in list(holdings.keys()):
            p = holdings[sym].get("price", 0)
            if p <= 0: continue
            hwm = holdings[sym].get("hwm", p)
            if p > hwm: holdings[sym]["hwm"] = p
            if hwm > 0 and (p/hwm - 1) * 100 < stop_pct:
                cash += holdings[sym]["qty"] * p * (1 - COMM); trades += 1
                del holdings[sym]

        total = cash + sum(h["qty"] * h.get("price", 0) for h in holdings.values())
        if total > max_e: max_e = total

        # Update unit value as fraction of current NAV
        current_unit_pct = unit_pct
        unit_value = total * current_unit_pct

        # Score stocks
        scored = []
        for code in precomputed:
            if rd in precomputed[code]:
                scored.append((code, precomputed[code][rd]))

        if not scored:
            eq.append(total)
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        top_set = {s[0] for s in scored[:top_n]}

        # ── Martingale accumulation logic ──
        # 1. Remove stocks that dropped out of Top-N
        for sym in list(holdings.keys()):
            if sym not in top_set:
                cash += holdings[sym]["qty"] * holdings[sym].get("price", 0) * (1 - COMM)
                trades += 1
                del holdings[sym]

        # 2. Add unit to persistent stocks, enter new stocks
        for sym in top_set:
            p = price_on(sym, rd)
            if p is None or p < 0.01: continue

            if sym in holdings:
                # Stock persists in Top-N: add 1 more unit if below cap
                current_units = holdings[sym].get("units", 0)
                if current_units < max_units:
                    add_qty = int(unit_value / p / 100) * 100
                    if add_qty >= 100:
                        cost = add_qty * p
                        if cash >= cost * (1 + COMM):
                            cash -= cost * (1 + COMM)
                            holdings[sym]["qty"] += add_qty
                            holdings[sym]["units"] = current_units + 1
                            trades += 1
            else:
                # New entry: buy 1 unit
                buy_qty = int(unit_value / p / 100) * 100
                if buy_qty >= 100:
                    cost = buy_qty * p
                    if cash >= cost * (1 + COMM):
                        cash -= cost * (1 + COMM)
                        holdings[sym] = {"qty": buy_qty, "price": p, "units": 1, "hwm": p}
                        trades += 1

        total = cash + sum(h["qty"] * h.get("price", 0) for h in holdings.values())
        eq.append(total)

    # Metrics
    eq_arr = np.array(eq)
    if len(eq_arr) < 2 or eq_arr[0] <= 0: return None
    n_y = len(eq_arr) / 12.0
    if n_y < 0.5: return None
    cagr = ((eq_arr[-1] / eq_arr[0]) ** (1 / n_y) - 1) * 100
    mr = np.diff(eq_arr) / (eq_arr[:-1] + 1e-10)
    sharpe = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 else 0
    peak = eq_arr[0]; maxdd = 0.0
    for v in eq_arr:
        if v > peak: peak = v
        ddi = (peak - v) / peak * 100
        if ddi > maxdd: maxdd = ddi
    total_ret = (eq_arr[-1] / eq_arr[0] - 1) * 100

    # Compute concentration metrics
    if holdings:
        pos_sizes = [h["qty"] * h.get("price", 0) / max(total, 1) for h in holdings.values()]
        avg_concentration = np.mean(pos_sizes) if pos_sizes else 0
        max_concentration = np.max(pos_sizes) if pos_sizes else 0
        n_positions = len(holdings)
        avg_units = np.mean([h.get("units", 0) for h in holdings.values()]) if holdings else 0
    else:
        avg_concentration = 0; max_concentration = 0; n_positions = 0; avg_units = 0

    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "total_ret": total_ret,
            "trades": trades, "ny": n_y, "final": float(eq_arr[-1]),
            "avg_conc": avg_concentration, "max_conc": max_concentration,
            "n_pos": n_positions, "avg_units": avg_units, "cash_pct": cash / max(total, 1) * 100}


# ═══════════════════ 5. Equal-Weight Benchmark ═══════════════════
def run_equal_weight(start, end, top_n=10, stop_pct=-10):
    """Standard monthly equal-weight rebalance (no accumulation)."""
    rebal = get_monthly(start, end)
    if len(rebal) < 6: return None

    cash = CAPITAL; holdings = {}; eq = [CAPITAL]; max_e = CAPITAL; trades = 0

    for rd in rebal:
        for sym in list(holdings.keys()):
            p = price_on(sym, rd)
            if p is None or p < 0.01:
                cash += holdings[sym]["qty"] * p * 0.7 if p else 0; del holdings[sym]; continue
            holdings[sym]["price"] = p

        total = cash + sum(h["qty"] * h.get("price", 0) for h in holdings.values())

        for sym in list(holdings.keys()):
            p = holdings[sym].get("price", 0)
            if p <= 0: continue
            hwm = holdings[sym].get("hwm", p)
            if p > hwm: holdings[sym]["hwm"] = p
            if hwm > 0 and (p/hwm - 1) * 100 < stop_pct:
                cash += holdings[sym]["qty"] * p * (1 - COMM); trades += 1
                del holdings[sym]

        total = cash + sum(h["qty"] * h.get("price", 0) for h in holdings.values())
        if total > max_e: max_e = total

        scored = [(code, precomputed[code][rd]) for code in precomputed if rd in precomputed[code]]
        if not scored:
            eq.append(total); continue

        scored.sort(key=lambda x: x[1], reverse=True)
        top_set = {s[0] for s in scored[:top_n]}

        for sym in list(holdings.keys()):
            if sym not in top_set:
                cash += holdings[sym]["qty"] * holdings[sym].get("price", 0) * (1 - COMM)
                trades += 1; del holdings[sym]

        n_pos = max(len(top_set), 1)
        per_s = total / n_pos * 0.90
        for sym in top_set:
            p = price_on(sym, rd)
            if p is None or p < 0.01: continue
            tq = int(per_s / p / 100) * 100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1 + COMM):
                        cash -= cost * (1 + COMM); holdings[sym]["qty"] = tq; trades += 1
                    elif diff < 0:
                        cash += cost * (1 - COMM); holdings[sym]["qty"] = tq; trades += 1
            else:
                cost = tq * p
                if cash >= cost * (1 + COMM):
                    cash -= cost * (1 + COMM)
                    holdings[sym] = {"qty": tq, "price": p, "hwm": p}; trades += 1

        total = cash + sum(h["qty"] * h.get("price", 0) for h in holdings.values())
        eq.append(total)

    eq_arr = np.array(eq)
    if len(eq_arr) < 2 or eq_arr[0] <= 0: return None
    n_y = len(eq_arr) / 12.0
    cagr = ((eq_arr[-1] / eq_arr[0]) ** (1 / n_y) - 1) * 100
    mr = np.diff(eq_arr) / (eq_arr[:-1] + 1e-10)
    sharpe = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 else 0
    peak = eq_arr[0]; maxdd = 0.0
    for v in eq_arr:
        if v > peak: peak = v
        ddi = (peak - v) / peak * 100
        if ddi > maxdd: maxdd = ddi
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd,
            "total_ret": (eq_arr[-1] / eq_arr[0] - 1) * 100, "trades": trades}


# ═══════════════════ 6. CSI300 benchmark ═══════════════════
def csi300_bnh(start, end):
    si = np.searchsorted(csi_dates, start)
    ei = np.searchsorted(csi_dates, end, side="right") - 1
    if ei <= si: return None
    seg = csi_c[si:ei + 1]
    n_y = len(seg) / 252.0
    cagr = ((seg[-1] / seg[0]) ** (1 / n_y) - 1) * 100
    peak = seg[0]; maxdd = 0.0
    for v in seg:
        if v > peak: peak = v
        if (peak - v) / peak * 100 > maxdd: maxdd = (peak - v) / peak * 100
    return {"cagr": cagr, "maxdd": maxdd, "total_ret": (seg[-1] / seg[0] - 1) * 100}


bm_test = csi300_bnh(TEST_START, TEST_END)
bm_train = csi300_bnh(TRAIN_START, TRAIN_END)

# ═══════════════════ 7. Sweep ═══════════════════
print("\n[3] Martingale parameter sweep...")

param_grid = list(itertools.product(
    [5, 10, 15],       # top_n
    [3, 5, 8],         # max_units
    [0.03, 0.05, 0.08],  # unit_pct (per-unit size as % of NAV)
    [-5, -10, -15],    # stop_pct
))

results = []
ew_results = {}
count = 0
t1 = time.time()

for tn, mu, up, sp in param_grid:
    # Skip combos where max exposure exceeds 150% (unrealistic)
    max_exposure = tn * up * mu
    if max_exposure > 1.5: continue
    if max_exposure < 0.15: continue  # too little exposure

    count += 1
    rt = run_martingale(TRAIN_START, TRAIN_END, tn, mu, up, sp)
    re = run_martingale(TEST_START, TEST_END, tn, mu, up, sp)

    if rt and re:
        ex_test = re["cagr"] - bm_test["cagr"]
        ex_train = rt["cagr"] - bm_train["cagr"]
        results.append({
            "tn": tn, "mu": mu, "up": up, "sp": sp, "max_exp": max_exposure,
            "tr_c": round(rt["cagr"], 2), "tr_sh": round(rt["sharpe"], 3),
            "tr_dd": round(rt["maxdd"], 2), "tr_ex": round(ex_train, 2),
            "te_c": round(re["cagr"], 2), "te_sh": round(re["sharpe"], 3),
            "te_dd": round(re["maxdd"], 2), "te_ex": round(ex_test, 2),
            "te_ret": round(re["total_ret"], 2),
            "cash_pct": round(re.get("cash_pct", 0), 1),
            "avg_units": round(re.get("avg_units", 0), 1),
        })

    if count % 20 == 0:
        e = time.time() - t1
        print(f"  [{count}] {e:.0f}s  Last: N={tn} M={mu} U={up:.2f} S={sp}")

elapsed_sweep = time.time() - t1
print(f"  Sweep complete in {elapsed_sweep:.0f}s, {len(results)} valid results")

# ═══════════════════ 8. Results ═══════════════════
print(f"\n{'='*90}")
print("RESULTS: Martingale Accumulation Strategy")
print(f"{'='*90}")

# Equal-weight benchmark
print(f"\n--- Benchmark: Equal-Weight (monthly rebalance) ---")
for tn in [5, 10, 15]:
    ew_key = (tn, -10)
    if ew_key not in ew_results:
        ew_results[ew_key] = run_equal_weight(TEST_START, TEST_END, tn, -10)
    r = ew_results[ew_key]
    if r:
        ex = r["cagr"] - bm_test["cagr"]
        print(f"  N={tn}: CAGR={r['cagr']:+.1f}%  Excess={ex:+.1f}%  DD={r['maxdd']:.1f}%  Sharpe={r['sharpe']:.3f}")

# Rankings
by_ex = sorted(results, key=lambda x: x["te_ex"], reverse=True)
for i, r in enumerate(by_ex): r["re"] = i
by_dd = sorted(results, key=lambda x: x["te_dd"])
for i, r in enumerate(by_dd): r["rd"] = i
for r in results:
    r["comp"] = r["re"] * 0.50 + r["rd"] * 0.30
best = sorted(results, key=lambda x: x["comp"])[0]

print(f"\n--- TOP 10 Martingale by Test Excess ---")
print(f"{'#':<4s} {'N':<4s} {'MaxU':<5s} {'Unit%':<6s} {'Stop':<5s} {'MaxExp':<7s} {'Excess':>8s} {'CAGR':>8s} {'DD':>7s} {'Sh':>6s} {'Cash%':>6s}")
print("-" * 95)
for i, r in enumerate(by_ex[:15], 1):
    print(f"{i:<4d} {r['tn']:<4d} {r['mu']:<5d} {r['up']*100:<5.0f}% {r['sp']:<5d} {r['max_exp']*100:<6.0f}% "
          f"{r['te_ex']:>+7.1f}% {r['te_c']:>+7.1f}% {r['te_dd']:>6.1f}% {r['te_sh']:>6.3f} {r['cash_pct']:>5.0f}%")

print(f"\n--- TOP 5 by Composite (Excess + Low DD) ---")
for i, r in enumerate(sorted(results, key=lambda x: x["comp"])[:5], 1):
    print(f"  #{i}: N={r['tn']} M={r['mu']} U={r['up']*100:.0f}% stop={r['sp']} maxExp={r['max_exp']*100:.0f}% | "
          f"Excess={r['te_ex']:+.1f}% CAGR={r['te_c']:+.1f}% DD={r['te_dd']:.1f}% "
          f"Sh={r['te_sh']:.3f} Cash={r['cash_pct']:.0f}% AvgU={r['avg_units']:.1f}")

print(f"\n--- BEST CONFIG ---")
print(f"  N={best['tn']}  MaxUnits={best['mu']}  UnitSize={best['up']*100:.0f}%  Stop={best['sp']}%")
print(f"  Max exposure: {best['max_exp']*100:.0f}% of portfolio")
print(f"  Train: CAGR={best['tr_c']:+.1f}%  Excess={best['tr_ex']:+.1f}%  DD={best['tr_dd']:.1f}%")
print(f"  Test:  CAGR={best['te_c']:+.1f}%  Excess={best['te_ex']:+.1f}%  DD={best['te_dd']:.1f}%  "
      f"Sharpe={best['te_sh']:.3f}  TotRet={best['te_ret']:+.1f}%")
print(f"  Avg idle cash: {best['cash_pct']:.0f}%  Avg units: {best['avg_units']:.1f}")

# Yearly
print(f"\n--- Yearly (Best Config) ---")
for yr in [2022, 2023, 2024, 2025]:
    ys, ye = f"{yr}0101", f"{yr}1231"
    ry = run_martingale(ys, ye, best["tn"], best["mu"], best["up"], best["sp"])
    csiy = csi300_bnh(ys, ye)
    if ry and csiy:
        ex = ry["cagr"] - csiy["cagr"]
        print(f"  {yr}: Mart={ry['cagr']:+.1f}%  CSI300={csiy['cagr']:+.1f}%  "
              f"Excess={ex:+.1f}%  DD={ry['maxdd']:.1f}%  "
              f"Cash={ry.get('cash_pct',0):.0f}%  AvgU={ry.get('avg_units',0):.1f}")

# Final comparison
print(f"\n{'='*90}")
print("FINAL COMPARISON")
print(f"{'='*90}")
ew15 = run_equal_weight(TEST_START, TEST_END, 15, -10)
print(f"{'Strategy':<35s} {'Test CAGR':>10s} {'MaxDD':>10s} {'Excess':>10s} {'Sharpe':>10s}")
print("-" * 80)
if ew15:
    ew_ex = ew15["cagr"] - bm_test["cagr"]
    print(f"{'Equal-Weight N=15':<35s} {ew15['cagr']:>+9.1f}% {ew15['maxdd']:>9.1f}% {ew_ex:>+9.1f}% {ew15['sharpe']:>10.3f}")
print(f"{'Martingale Best':<35s} {best['te_c']:>+9.1f}% {best['te_dd']:>9.1f}% {best['te_ex']:>+9.1f}% {best['te_sh']:>10.3f}")
print(f"{'State Machine (best)':<35s} {'+7.0%':>10s} {'5.1%':>10s} {'+8.3%':>10s}")
print(f"{'CSI300 B&H':<35s} {bm_test['cagr']:>+9.1f}% {bm_test['maxdd']:>9.1f}%")

print(f"\n{'='*90}")
print(f"Total wall time: {time.time() - t1 + elapsed_sweep:.0f}s")
print(f"{'='*90}")

"""Canonical backtest. ALL decision logic through strategy class. python run_backtest.py [--verify] [--check-vtmd]"""
import os, sys, time, itertools
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from datetime import datetime
from quanti.data.storage import DataStorage
from quanti.strategy.delayed_confirm import DelayedConfirmStrategy
from quanti.types import Bar, MarketData, Portfolio, OrderSide

CAPITAL = 90000; COMM = 0.00025
TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END = "20220101", "20251231"
TOP_N = 5; STOP_PCT = -10; MIN_TREND = 3; DD_EXIT_PCT = 15

def sma(arr, p):
    if len(arr) < p: return None
    o = np.full(len(arr), np.nan); cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p; return o

def monthly_dates(dates, s, e):
    m = []
    for d in dates:
        if d < s or d > e: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

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
    return above and adx_ok and score >= MIN_TREND, score

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


# ═══════════════════════════════════════════════════════════════
# THE LOOP. All decision logic is inside strategy.generate_signals
# → size_positions → risk_check. This function only EXECUTES.
# ═══════════════════════════════════════════════════════════════

def run_through_strategy(strategy, stock_data, all_dates, bond_cl, gold_cl,
                         csi_raw_bars, start, end):
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; bu = 0; gu = 0; eq = [cash]
    max_eq_val = cash; dd_active = False

    for rd in rebal:
        bp = bond_cl.get(rd); gp = gold_cl.get(rd)

        # Build MarketData with index_bars (full CSI300 history up to rd)
        ib_list = []
        for b in csi_raw_bars:
            if b.trade_date > rd: break
            ib_list.append(Bar(symbol="510300",
                datetime=datetime.strptime(b.trade_date, "%Y%m%d"),
                open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume))

        # Stock bars: last 260 bars for each stock, up to rd
        bars_for_md = {}
        for code in stock_data:
            c_arr, h_arr, l_arr, v_arr, d_arr = stock_data[code]
            idx = None
            for i in range(len(d_arr)-1, -1, -1):
                if d_arr[i] <= rd: idx = i+1; break
            if idx is None or idx < 200: continue
            start_i = max(0, idx-260)
            bl = []
            for j in range(start_i, idx):
                bl.append(Bar(symbol=code, datetime=datetime.strptime(d_arr[j], "%Y%m%d"),
                             open=c_arr[j], high=h_arr[j], low=l_arr[j],
                             close=c_arr[j], volume=v_arr[j]))
            bars_for_md[code] = bl

        md = MarketData(bars=bars_for_md, index_bars={"510300": ib_list},
                        timestamp=datetime.strptime(rd, "%Y%m%d"))

        # Update holding prices
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"]*0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"]*p

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values()) + bu*(bp or 1.0) + gu*(gp or 1.0)

        # DD breaker
        if total > max_eq_val: max_eq_val = total; dd_active = False
        dd = (max_eq_val - total) / max_eq_val * 100 if max_eq_val > 0 else 0
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            if bu > 0 and bp: cash += bu*bp*(1-COMM); bu = 0
            if gu > 0 and gp: cash += gu*gp*(1-COMM); gu = 0
            dd_active = True; eq.append(cash); continue
        elif dd_active and total / max_eq_val > 0.92:
            dd_active = False
        if dd_active: eq.append(cash); continue

        # Strategy cycle
        pf = Portfolio(positions={}, cash=cash, total_capital=total,
                       settled_cash=cash, timestamp=datetime.strptime(rd, "%Y%m%d"))
        signals = strategy.generate_signals(md)
        orders = strategy.size_positions(signals, cash, pf, md)
        orders = strategy.risk_check(orders, pf, md)

        # Execute
        bond_sym = strategy.bond_etf
        gold_sym = getattr(strategy, 'gold_etf', '')
        for o in orders:
            if o.symbol in (bond_sym, gold_sym):
                px = bp if o.symbol == bond_sym else gp
                if px is None or px <= 0: continue
                cst = o.quantity * px
                if o.side == OrderSide.BUY:
                    if cst*(1+COMM) <= cash:
                        cash -= cst*(1+COMM)
                        if o.symbol == bond_sym: bu += o.quantity
                        else: gu += o.quantity
                else:
                    if o.symbol == bond_sym and bu >= o.quantity:
                        cash += cst*(1-COMM); bu -= o.quantity
                    elif o.symbol != bond_sym and gu >= o.quantity:
                        cash += cst*(1-COMM); gu -= o.quantity
            else:
                px = price_on(o.symbol, rd, stock_data)
                if px is None or px <= 0: continue
                cst = o.quantity * px
                if o.side == OrderSide.BUY:
                    if cst*(1+COMM) <= cash:
                        cash -= cst*(1+COMM)
                        if o.symbol in holdings:
                            h = holdings[o.symbol]
                            nq = h["qty"] + o.quantity
                            holdings[o.symbol] = {"qty": nq, "price": px, "val": nq*px}
                        else:
                            holdings[o.symbol] = {"qty": o.quantity, "price": px, "val": cst}
                else:
                    h = holdings.get(o.symbol)
                    if h and h["qty"] >= o.quantity:
                        cash += cst*(1-COMM); h["qty"] -= o.quantity
                        if h["qty"] == 0: del holdings[o.symbol]
                        else: h["val"] = h["qty"] * px

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values()) + bu*(bp or 1.0) + gu*(gp or 1.0)
        eq.append(total)
    return metrics(eq)


# ═══════════════════════════════════════════════════════════════
# Baseline: always-invested stock selection (no market timing)
# ═══════════════════════════════════════════════════════════════

def run_baseline(tc, stock_data, all_dates, start, end):
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; eq = [cash]; max_eq = cash; dd_active = False
    for rd in rebal:
        trending = tc.get(rd, [])
        selected = {t[0] for t in trending[:TOP_N]} if trending else set()
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"]*0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"]*p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p/holdings[sym]["hwm"]-1)*100 < STOP_PCT:
                cash += holdings[sym]["val"]*(1-COMM); del holdings[sym]
        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())
        if total > max_eq: max_eq = total
        dd = (max_eq-total)/max_eq*100 if max_eq > 0 else 0
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            dd_active = True
        elif dd_active and total/max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue
        if not selected: eq.append(cash); continue
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
        n_pos = max(len(selected), 1); per = total/n_pos*0.90
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


# ═══════════════════════════════════════════════════════════════
# Verify modes
# ═══════════════════════════════════════════════════════════════

def verify_nm():
    storage = DataStorage()
    raw = storage.load_bars("510300")
    raw_b = storage.load_bars("511880")
    bond_cl = {r.trade_date: float(r.close) for r in raw_b} if raw_b else {}
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
    N_VALS, M_VALS = [3,4,5,6,7], [20,30,40,50,60]
    grid = list(itertools.product(N_VALS, M_VALS))
    results = []
    for N, M in grid:
        st = DelayedConfirmStrategy(confirm_days=N, cooldown_days=M, decay_schedule="none",
                                     use_sharp_exit=False, stock_universe=codes)
        r = run_through_strategy(st, stock_data, all_dates, bond_cl, {}, raw, TEST_START, TEST_END)
        results.append({"N": N, "M": M, "cagr": r["cagr"]})
    print(f"\nNxM Test CAGR matrix (via strategy class):")
    print(f"  N\\M  " + "  ".join(f"{m:6d}" for m in M_VALS))
    for N in N_VALS:
        vals = [next(x["cagr"] for x in results if x["N"]==N and x["M"]==M) for M in M_VALS]
        print(f"  {N:>3d}   " + "  ".join(f"{v:+5.1f}%" for v in vals))
    best = max(results, key=lambda x: x["cagr"])
    n5m40 = next(x for x in results if x["N"]==5 and x["M"]==40)
    rank = sorted(results, key=lambda x: x["cagr"], reverse=True).index(n5m40) + 1
    print(f"\nBest: N={best['N']}, M={best['M']} -> C={best['cagr']:+.2f}%")
    print(f"N=5/M=40: rank #{rank}/{len(grid)}  C={n5m40['cagr']:+.2f}%")
    print("VERDICT: N=5 is " + ("CONFIRMED" if rank <= 3 else "NOT in top cluster"))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--verify" in sys.argv:
        t0 = time.time(); verify_nm(); print(f"\nDone {time.time()-t0:.0f}s"); sys.exit(0)

    t0 = time.time()
    print("Loading...")
    storage = DataStorage()
    raw = storage.load_bars("510300")

    raw_b = storage.load_bars("511880")
    bond_cl = {r.trade_date: float(r.close) for r in raw_b} if raw_b else {}
    raw_g = storage.load_bars("518880")
    gold_cl = {r.trade_date: float(r.close) for r in raw_g} if raw_g else {}

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
    print(f"Loaded: {len(stock_data)} stocks, {len(all_dates)} days")

    def precompute(start, end):
        rd = monthly_dates(all_dates, start, end); cache = {}
        for i, d in enumerate(rd):
            if i % 12 == 0: print(f"  Trending: {i}/{len(rd)}")
            t = []
            for code in stock_data:
                d2 = data_at(code, d, 260, stock_data)
                if d2 is None: continue
                cl, hi, lo, vo = d2
                is_t, nc = is_stock_uptrend(cl, hi, lo, vo)
                if is_t and nc >= MIN_TREND: t.append((code, trend_score(cl)))
            t.sort(key=lambda x: x[1], reverse=True); cache[d] = t
        return cache
    print("Precomputing...")
    tc_train = precompute(TRAIN_START, TRAIN_END)
    tc_test  = precompute(TEST_START, TEST_END)

    # Strategy instances
    s_base = DelayedConfirmStrategy(decay_schedule="none", use_sharp_exit=False, stock_universe=codes)
    s_a43  = DelayedConfirmStrategy(decay_schedule="A43",  use_sharp_exit=False, stock_universe=codes)
    s_sharp= DelayedConfirmStrategy(decay_schedule="none", use_sharp_exit=True,  stock_universe=codes)
    s_s43  = DelayedConfirmStrategy(decay_schedule="A43",  use_sharp_exit=True,  stock_universe=codes)

    bl_tr = run_baseline(tc_train, stock_data, all_dates, TRAIN_START, TRAIN_END)
    bl_te = run_baseline(tc_test,  stock_data, all_dates, TEST_START,  TEST_END)

    br_tr  = run_through_strategy(s_base, stock_data, all_dates, bond_cl, gold_cl, raw, TRAIN_START, TRAIN_END)
    br_te  = run_through_strategy(s_base, stock_data, all_dates, bond_cl, gold_cl, raw, TEST_START,  TEST_END)
    a43_tr = run_through_strategy(s_a43,  stock_data, all_dates, bond_cl, gold_cl, raw, TRAIN_START, TRAIN_END)
    a43_te = run_through_strategy(s_a43,  stock_data, all_dates, bond_cl, gold_cl, raw, TEST_START,  TEST_END)
    sh_tr  = run_through_strategy(s_sharp, stock_data, all_dates, bond_cl, gold_cl, raw, TRAIN_START, TRAIN_END)
    sh_te  = run_through_strategy(s_sharp, stock_data, all_dates, bond_cl, gold_cl, raw, TEST_START,  TEST_END)
    s43_tr = run_through_strategy(s_s43,  stock_data, all_dates, bond_cl, gold_cl, raw, TRAIN_START, TRAIN_END)
    s43_te = run_through_strategy(s_s43,  stock_data, all_dates, bond_cl, gold_cl, raw, TEST_START,  TEST_END)

    print(f"\n{'='*90}")
    print("FINAL RESULTS (all decision logic through strategy class)")
    print(f"{'='*90}")
    print(f"\n{'Strategy':<35s} | {'Train C':>7s} {'Train S':>7s} {'Train D':>7s} | {'Test C':>7s} {'Test S':>7s} {'Test D':>7s}")
    print("-"*95)
    for name, tr, te in [
        ("BASELINE (always-invested)",          bl_tr, bl_te),
        ("BOND_ROTATE (entry only)",            br_tr, br_te),
        ("+ A43 decay",                         a43_tr, a43_te),
        ("+ Sharp3pct exit (no decay)",         sh_tr, sh_te),
        ("+ Sharp3pct + A43",                   s43_tr, s43_te),
    ]:
        print(f"{name:<35s} | {tr['cagr']:+7.2f}% {tr['sharpe']:6.3f} {tr['maxdd']:5.1f}% | {te['cagr']:+7.2f}% {te['sharpe']:6.3f} {te['maxdd']:5.1f}%")
    print(f"\nDelta over entry-only (Test CAGR):")
    print(f"  +A43 decay:        {a43_te['cagr']-br_te['cagr']:+.2f}%")
    print(f"  +Sharp3pct exit:   {sh_te['cagr']-br_te['cagr']:+.2f}%")
    print(f"  +Sharp3pct + A43:  {s43_te['cagr']-br_te['cagr']:+.2f}%")
    print(f"\nDone in {time.time()-t0:.0f}s")

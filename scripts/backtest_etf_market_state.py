"""
ETF multi-sector rotation with market-state-aware defense (run_backtest.py logic).

Core strategy (ported from run_backtest.py):
  1. Market state detection on CSI300:
     State 0=Bear, 1=Confirming, 2=Bull confirmed, 3=Cooldown, 4=Failed confirmation
  2. Position sizing = base_sm * decay(months_in_cycle)
     - State 2: base_sm=1.0  State 4: base_sm=0.5  Others: base_sm=0 (bond/gold)
     - A43 decay: 1.0 m1-4, 0.75 m5-8, 0.50 m9+
  3. Sharp3pct exit: CSI300 5-day < -3% -> all bond+gold, M_COOLDOWN=40
  4. Bond+Gold (80/20) defensive allocation
  5. Scoring: 3M return(50%) + 6M return(50%) + stability
     Entry filter: price>120MA, new highs, MA align, vol surge, ADX>25  (>=3/5)
  6. HWM -10% stop, DD -15% breaker, top N=5

Universe: our 25 multi-sector ETFs via etf_universe.py
"""
import sys, os, time
sys.path.insert(0, r"C:\study\AIWorkspace\quanti")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from quanti.data.storage import DataStorage
from quanti.config.etf_universe import ETF_UNIVERSE_MULTI, get_sector

# ── Constants ──────────────────────────────────────────────────────────
CAPITAL = 90000; COMM = 0.00025
TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START,  TEST_END  = "20220101", "20251231"
N_CONFIRM, M_COOLDOWN = 5, 40
VT, MD = 0.85, -0.02
TOP_N = 3; STOP_PCT = -10; MIN_TREND = 3; DD_EXIT_PCT = 15
SHARP_THRESHOLD = -0.03

# ── Indicator helpers ──────────────────────────────────────────────────
def sma(arr, p):
    if len(arr) < p: return None
    o = np.full(len(arr), np.nan)
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p
    return o

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

# ── ETF trend detection & scoring ──────────────────────────────────────
def is_etf_uptrend(cl, hi, lo, vol):
    """5-condition filter for ETFs (same logic as stock version)."""
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
    """3M + 6M momentum + stability."""
    if len(cl) < 130: return 0
    r3 = cl[-1]/cl[-63]-1 if cl[-63] > 1e-6 else 0
    r6 = cl[-1]/cl[-126]-1 if cl[-126] > 1e-6 else 0
    m3 = min(max(r3/0.5, 0), 1) if r3 > 0 else 0
    m6 = min(max(r6/0.8, 0), 1) if r6 > 0 else 0
    mom = (0.5*m3 + 0.5*m6)*100
    w = cl[-61:]; dr = np.diff(w)/(w[:-1]+1e-10)
    vs = (1 - min(np.nanstd(dr)/0.04, 1))*100
    return 0.6*mom + 0.4*vs

# ── Market state detection ─────────────────────────────────────────────
def build_state_map(csi_dates, csi_closes, csi_volumes, N, M, vt=None, md=None):
    n = len(csi_closes)
    ma120 = sma(csi_closes, 120); ma60 = sma(csi_closes, 60)
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
                if vt is not None and md is not None:
                    ws, we = cf, i+1
                    wv = np.mean(csi_volumes[ws:we]); pv = np.mean(csi_volumes[max(0, ws-20):ws])
                    cr = csi_closes[i]/csi_closes[cf]-1.0
                    if wv >= vt*pv and cr >= md: st[i] = 2; fd = i
                    else: st[i] = 4; cf = -1
                else:
                    st[i] = 2; fd = i; cf = -1
            else: st[i] = 1; continue
        if a120[i] and not a120[i-1]: cf = i; st[i] = 1
        elif a60[i]: st[i] = 4
        else: st[i] = 0
    return {str(csi_dates[j]): int(st[j]) for j in range(n)}

# ── Data loading helpers ───────────────────────────────────────────────
def load_etf_data(storage, codes):
    """Load into numpy arrays compatible with run_backtest.py data format."""
    stock_data = {}
    for code in codes:
        raw = storage.load_bars(code)
        if not raw or len(raw) < 200: continue
        d = [r.trade_date for r in raw]
        stock_data[code] = (
            np.array([r.close for r in raw], dtype=np.float64),
            np.array([r.high for r in raw], dtype=np.float64),
            np.array([r.low for r in raw], dtype=np.float64),
            np.array([r.volume for r in raw], dtype=np.float64),
            d,
        )
    return stock_data

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

def monthly_dates(dates, s, e):
    m = []
    for d in dates:
        if d < s or d > e: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

def precompute_trending(stock_data, all_dates, start, end):
    rd = monthly_dates(all_dates, start, end)
    cache = {}
    for i, d in enumerate(rd):
        if i % 12 == 0: print(f"  Trending: {i}/{len(rd)}")
        t = []
        for code in stock_data:
            d2 = data_at(code, d, 260, stock_data)
            if d2 is None: continue
            cl, hi, lo, vo = d2
            is_t, nc = is_etf_uptrend(cl, hi, lo, vo)
            if is_t and nc >= MIN_TREND: t.append((code, trend_score(cl)))
        t.sort(key=lambda x: x[1], reverse=True)
        cache[d] = t
    return cache

def metrics(eq_curve):
    eq = np.array(eq_curve); ny = len(eq_curve) / 12.0
    if eq[0] <= 0 or ny <= 0: return {"cagr": 0, "sharpe": 0, "maxdd": 100, "final": float(eq[-1])}
    cagr = ((eq[-1] / eq[0]) ** (1 / ny) - 1) * 100
    mr = np.diff(eq) / (eq[:-1] + 1e-10)
    sh = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 else 0
    peak = eq[0]; mdd = 0.0
    for v in eq:
        if v > peak: peak = v
        d = (peak - v) / peak * 100
        if d > mdd: mdd = d
    return {"cagr": cagr, "sharpe": sh, "maxdd": mdd, "final": float(eq[-1])}

# ── Core strategy runner ───────────────────────────────────────────────
def run_gold_sharp_decay(state_map, tc, stock_data, all_dates, bond_cl, gold_cl, start, end,
                          decay_fn, csi_ret5_arr, csi_idx_map, sharp_threshold=-0.03,
                          bond_pct=0.80, gold_pct=0.20, use_concentration=True):
    """Gold defensive (bond/gold split) + Sharp3pct exit + A43 decay.

    This is the best-performing strategy from run_backtest.py (gold+sharp+decay merged).
    Adapted for multi-sector ETFs with optional concentration limits.
    """
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; bu = 0; gu = 0
    eq = [cash]; max_eq = cash; dd_active = False
    months_in_cycle = 0; prev_mst = 0; genuine_prev = 0; sharp_cd = -1
    sector_counts_history = []  # track concentration enforcement

    for rd in rebal:
        mst = state_map.get(rd, 0); bp = bond_cl.get(rd); gp = gold_cl.get(rd)
        ci = csi_idx_map.get(str(rd))
        if ci is not None and ci >= 0 and ci >= sharp_cd and sharp_cd >= 0: sharp_cd = -1
        sharp_fired = False
        if ci is not None and ci >= 5 and ci < len(csi_ret5_arr) and len(holdings) > 0:
            r5 = csi_ret5_arr[ci]
            if not np.isnan(r5) and r5 < sharp_threshold: sharp_fired = True

        if sharp_fired:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0
            if gu > 0 and gp: cash += gu * gp * (1-COMM); gu = 0
            if cash > 1000:
                tcf = cash
                if bp:
                    bb = int(tcf * bond_pct * 0.99 / bp)
                    if bb > 0: bu += bb; cash -= bb * bp * (1+COMM)
                if gp:
                    bg = int(tcf * gold_pct * 0.99 / gp)
                    if bg > 0: gu += bg; cash -= bg * gp * (1+COMM)
            sharp_cd = ci + M_COOLDOWN
            eq.append(cash + bu*(bp or 1.0) + gu*(gp or 1.0))
            continue

        emst = 3 if (ci is not None and ci >= 0 and ci < sharp_cd) else mst
        if emst in (2, 4):
            if genuine_prev not in (2, 4): months_in_cycle = 1
            else: months_in_cycle += 1
            genuine_prev = emst
        elif emst == 3 and mst != 3: pass
        else: months_in_cycle = 0; genuine_prev = emst
        prev_mst = emst
        base_sm = 1.0 if emst == 2 else (0.5 if emst == 4 else 0.0)
        sm = base_sm * decay_fn(months_in_cycle)

        # Exit bond/gold when entering equity
        if sm > 0 and bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0
        if sm > 0 and gu > 0 and gp: cash += gu * gp * (1-COMM); gu = 0

        # HWM stops
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"] * 0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p / holdings[sym]["hwm"] - 1) * 100 < STOP_PCT:
                cash += holdings[sym]["val"] * (1-COMM); del holdings[sym]

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values()) + bu*(bp or 1.0) + gu*(gp or 1.0)
        if total > max_eq: max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0

        # DD breaker
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if bu > 0 and bp: cash += bu*bp*(1-COMM); bu = 0
            if gu > 0 and gp: cash += gu*gp*(1-COMM); gu = 0; dd_active = True
        elif dd_active and total / max_eq > 0.92: dd_active = False

        if dd_active:
            eq.append(cash)
            continue

        # Defensive: all in bond+gold
        if sm == 0:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if cash > 1000:
                tcf = cash
                if bp:
                    bb = int(tcf * bond_pct * 0.99 / bp)
                    if bb > 0: bu += bb; cash -= bb * bp * (1+COMM)
                if gp:
                    bg = int(tcf * gold_pct * 0.99 / gp)
                    if bg > 0: gu += bg; cash -= bg * gp * (1+COMM)
            eq.append(cash + bu*(bp or 1.0) + gu*(gp or 1.0))
            continue

        # Equity: select trending ETFs
        trending = tc.get(rd, [])
        if not trending:
            eq.append(cash + sum(h["qty"]*h["price"] for h in holdings.values()) + bu*(bp or 1.0) + gu*(gp or 1.0))
            continue

        # Apply concentration limits if enabled
        if use_concentration:
            selected = set()
            sector_counts = {}
            for t_code, t_score in trending:
                if len(selected) >= TOP_N: break
                sector = get_sector(t_code)
                if sector not in ("宽基", "防御"):
                    if sector_counts.get(sector, 0) >= 2:
                        continue  # skip this ETF, sector at limit
                selected.add(t_code)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            selected = {t[0] for t in trending[:TOP_N]}

        # Rotate: sell ETFs no longer selected
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]

        n_pos = max(len(selected), 1)
        per = total * sm / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: continue
            tq = int(per / p / 100) * 100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost*(1+COMM): cash -= cost*(1+COMM); holdings[sym]["qty"] = tq
                    elif diff < 0: cash += cost*(1-COMM); holdings[sym]["qty"] = tq
            else:
                cost = tq * p
                if cash >= cost*(1+COMM):
                    cash -= cost*(1+COMM)
                    holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values()) + bu*(bp or 1.0) + gu*(gp or 1.0)
        eq.append(total)

    return metrics(eq), sector_counts_history


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Loading data...")
    storage = DataStorage()

    # CSI300 as market state index
    raw = storage.load_bars("510300")
    csi_dates = np.array([r.trade_date for r in raw])
    csi_closes = np.array([r.close for r in raw], dtype=np.float64)
    csi_volumes = np.array([r.volume for r in raw], dtype=np.float64)
    nc = len(csi_closes)
    csi_ret5 = np.full(nc, np.nan)
    for i in range(5, nc): csi_ret5[i] = csi_closes[i] / csi_closes[i-5] - 1.0
    csi_idx_map = {str(d): i for i, d in enumerate(csi_dates)}

    # Bond & Gold
    raw_b = storage.load_bars("511880")
    bond_cl = {r.trade_date: float(r.close) for r in raw_b} if raw_b else {}
    raw_g = storage.load_bars("518880")
    gold_cl = {r.trade_date: float(r.close) for r in raw_g} if raw_g else {}

    # All ETF universe
    etf_codes = [e["code"] for e in ETF_UNIVERSE_MULTI]
    stock_data = load_etf_data(storage, etf_codes)
    all_ds = set()
    for v in stock_data.values(): all_ds.update(v[4])
    all_dates = sorted(all_ds)
    print(f"Loaded: {len(stock_data)} ETFs, {len(all_dates)} days")

    # Build market state map
    sm_all = build_state_map(csi_dates, csi_closes, csi_volumes,
                             N_CONFIRM, M_COOLDOWN, VT, MD)

    # Precompute trending cache
    print("Precomputing trending cache...")
    tc_train = precompute_trending(stock_data, all_dates, TRAIN_START, TRAIN_END)
    tc_test  = precompute_trending(stock_data, all_dates, TEST_START, TEST_END)

    # Decay function
    a43_fn = lambda m: 1.0 if m <= 4 else (0.75 if m <= 8 else 0.50)

    # Run strategies
    print("\n" + "="*80)
    print("ETF MULTI-SECTOR MARKET-STATE STRATEGY")
    print("="*80)

    results = {}
    strategies = [
        ("ETF+Baseline (120MA)", dict(decay_fn=lambda m: 1.0, sharp_threshold=-999, bond_pct=1.0, gold_pct=0.0)),
        ("ETF+BondRotate", dict(decay_fn=lambda m: 1.0, sharp_threshold=-999, bond_pct=1.0, gold_pct=0.0)),
        ("ETF+A43 decay", dict(decay_fn=a43_fn, sharp_threshold=-999, bond_pct=1.0, gold_pct=0.0)),
        ("ETF+Sharp3pct+A43", dict(decay_fn=a43_fn, sharp_threshold=SHARP_THRESHOLD, bond_pct=1.0, gold_pct=0.0)),
        ("ETF+Gold(80/20)+A43", dict(decay_fn=a43_fn, sharp_threshold=-999, bond_pct=0.80, gold_pct=0.20)),
        ("ETF+Gold+Sharp+A43 merged", dict(decay_fn=a43_fn, sharp_threshold=SHARP_THRESHOLD, bond_pct=0.80, gold_pct=0.20)),
    ]
    for label, a in strategies:
        tr, _ = run_gold_sharp_decay(sm_all, tc_train, stock_data, all_dates, bond_cl, gold_cl,
                 TRAIN_START, TRAIN_END, csi_ret5_arr=csi_ret5, csi_idx_map=csi_idx_map, **a)
        te, _ = run_gold_sharp_decay(sm_all, tc_test, stock_data, all_dates, bond_cl, gold_cl,
                 TEST_START, TEST_END, csi_ret5_arr=csi_ret5, csi_idx_map=csi_idx_map, **a)
        results[label] = {"train": tr, "test": te}

    print(f"\n{'Strategy':<35s} | {'Train C':>7s} {'Train S':>7s} {'Train D':>7s} | {'Test C':>7s} {'Test S':>7s} {'Test D':>7s}")
    print("-"*105)
    for label, r in results.items():
        tr, te = r["train"], r["test"]
        print(f"{label:<35s} | {tr['cagr']:+7.2f}% {tr['sharpe']:6.3f} {tr['maxdd']:5.1f}% | {te['cagr']:+7.2f}% {te['sharpe']:6.3f} {te['maxdd']:5.1f}%")

    # Gold ratio sweep (test only)
    print("\n" + "="*80)
    print("GOLD RATIO SWEEP (Test 2022-2025 only)")
    print("="*80)
    for bpct, gpct, label in [(100,0,"100/0"),(90,10,"90/10"),(80,20,"80/20"),
                               (70,30,"70/30"),(60,40,"60/40"),(50,50,"50/50")]:
        te, _ = run_gold_sharp_decay(
            sm_all, tc_test, stock_data, all_dates, bond_cl, gold_cl,
            TEST_START, TEST_END, a43_fn, csi_ret5, csi_idx_map,
            SHARP_THRESHOLD, bond_pct=bpct/100, gold_pct=gpct/100)
        print(f"  Gold{label}: CAGR={te['cagr']:+6.2f}% Sharpe={te['sharpe']:+6.3f} MaxDD={te['maxdd']:+5.1f}%")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

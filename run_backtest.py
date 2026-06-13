"""Canonical backtest. python run_backtest.py [--verify]"""
import os, sys, time, itertools
_PROJECT_ROOT = r"C:\study\AIWorkspace\quanti"
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025
TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END = "20220101", "20251231"
N_CONFIRM = 5; M_COOLDOWN = 40; VT = 0.85; MD = -0.02
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

def run_baseline(tc, stock_data, all_dates, start, end):
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; eq = [cash]; max_eq = cash; dd_active = False
    for rd in rebal:
        d = data_at("510300", rd, 200, stock_data)
        mkt_ok = True
        if d is not None:
            c = d[0]; m120 = sma(c, 120)
            if m120 is not None and not np.isnan(m120[-1]): mkt_ok = c[-1] > m120[-1]
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"] * 0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p / holdings[sym]["hwm"] - 1) * 100 < STOP_PCT:
                cash += holdings[sym]["val"] * (1-COMM); del holdings[sym]
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())
        if total > max_eq: max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            dd_active = True
        elif dd_active and total / max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue
        if not mkt_ok:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            eq.append(cash); continue
        trending = tc.get(rd, [])
        if not trending: eq.append(cash); continue
        selected = {t[0] for t in trending[:TOP_N]}
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
        n_pos = max(len(selected), 1); per = total / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: continue
            tq = int(per / p / 100) * 100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1+COMM): cash -= cost * (1+COMM); holdings[sym]["qty"] = tq
                    elif diff < 0: cash += cost * (1-COMM); holdings[sym]["qty"] = tq
            else:
                cost = tq * p
                if cash >= cost * (1+COMM): cash -= cost * (1+COMM); holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())
        eq.append(total)
    return metrics(eq)

def run_bond_rotate(state_map, tc, stock_data, all_dates, bond_cl, start, end):
    return run_bond_rotate_decay(state_map, tc, stock_data, all_dates, bond_cl, start, end, lambda m: 1.0)

def run_bond_rotate_decay(state_map, tc, stock_data, all_dates, bond_cl, start, end, decay_fn):
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; bu = 0; eq = [cash]; max_eq = cash; dd_active = False
    months_in_cycle = 0; prev_mst = 0
    for rd in rebal:
        mst = state_map.get(rd, 0)
        if mst in (2, 4):
            if prev_mst not in (2, 4): months_in_cycle = 1
            else: months_in_cycle += 1
        else: months_in_cycle = 0
        prev_mst = mst
        base_sm = 1.0 if mst == 2 else (0.5 if mst == 4 else 0.0)
        sm = base_sm * decay_fn(months_in_cycle)
        bp = bond_cl.get(rd)
        if sm > 0 and bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"] * 0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p / holdings[sym]["hwm"] - 1) * 100 < STOP_PCT:
                cash += holdings[sym]["val"] * (1-COMM); del holdings[sym]
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0)
        if total > max_eq: max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0; dd_active = True
        elif dd_active and total / max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue
        if sm == 0:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if cash > 1000 and bp:
                buy = int(cash * 0.99 / bp)
                if buy > 0: bu += buy; cash -= buy * bp * (1+COMM)
            eq.append(cash + bu * (bp or 1.0)); continue
        trending = tc.get(rd, [])
        if not trending: eq.append(cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0)); continue
        selected = {t[0] for t in trending[:TOP_N]}
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
        n_pos = max(len(selected), 1); per = total * sm / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: continue
            tq = int(per / p / 100) * 100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1+COMM): cash -= cost * (1+COMM); holdings[sym]["qty"] = tq
                    elif diff < 0: cash += cost * (1-COMM); holdings[sym]["qty"] = tq
            else:
                cost = tq * p
                if cash >= cost * (1+COMM): cash -= cost * (1+COMM); holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0)
        eq.append(total)
    return metrics(eq)

def run_sharp_decay(state_map, tc, stock_data, all_dates, bond_cl, start, end, decay_fn, csi_ret5_arr, csi_idx_map, sharp_threshold=-0.03):
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; bu = 0; eq = [cash]; max_eq = cash; dd_active = False
    months_in_cycle = 0; prev_mst = 0; genuine_prev = 0; sharp_cd = -1
    for rd in rebal:
        mst = state_map.get(rd, 0); bp = bond_cl.get(rd); ci = csi_idx_map.get(str(rd))
        if ci is not None and ci >= 0 and ci >= sharp_cd and sharp_cd >= 0: sharp_cd = -1
        sharp_fired = False
        if ci is not None and ci >= 5 and ci < len(csi_ret5_arr) and len(holdings) > 0:
            r5 = csi_ret5_arr[ci]
            if not np.isnan(r5) and r5 < sharp_threshold: sharp_fired = True
        if sharp_fired:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0
            if cash > 1000 and bp: buy_b = int(cash * 0.99 / bp); bu += buy_b; cash -= buy_b * bp * (1+COMM) if buy_b > 0 else 0
            sharp_cd = ci + M_COOLDOWN; eq.append(cash + bu * (bp or 1.0)); continue
        emst = 3 if (ci is not None and ci >= 0 and ci < sharp_cd) else mst
        # Use genuine_prev to skip over Sharp cooldown periods
        if emst in (2, 4):
            if genuine_prev not in (2, 4): months_in_cycle = 1
            else: months_in_cycle += 1
            genuine_prev = emst
        elif emst == 3 and mst != 3:
            pass  # Sharp cooldown: don't touch anything
        else:
            months_in_cycle = 0
            genuine_prev = emst
        prev_mst = emst
        base_sm = 1.0 if emst == 2 else (0.5 if emst == 4 else 0.0)
        sm = base_sm * decay_fn(months_in_cycle)
        if sm > 0 and bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"] * 0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p / holdings[sym]["hwm"] - 1) * 100 < STOP_PCT:
                cash += holdings[sym]["val"] * (1-COMM); del holdings[sym]
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0)
        if total > max_eq: max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0; dd_active = True
        elif dd_active and total / max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue
        if sm == 0:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if cash > 1000 and bp:
                buy = int(cash * 0.99 / bp)
                if buy > 0: bu += buy; cash -= buy * bp * (1+COMM)
            eq.append(cash + bu * (bp or 1.0)); continue
        trending = tc.get(rd, [])
        if not trending: eq.append(cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0)); continue
        selected = {t[0] for t in trending[:TOP_N]}
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
        n_pos = max(len(selected), 1); per = total * sm / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: continue
            tq = int(per / p / 100) * 100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1+COMM): cash -= cost * (1+COMM); holdings[sym]["qty"] = tq
                    elif diff < 0: cash += cost * (1-COMM); holdings[sym]["qty"] = tq
            else:
                cost = tq * p
                if cash >= cost * (1+COMM): cash -= cost * (1+COMM); holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0)
        eq.append(total)
    return metrics(eq)

def run_gold_decay(state_map, tc, stock_data, all_dates, bond_cl, gold_cl, start, end, decay_fn,
                    bond_pct=0.80, gold_pct=0.20):
    """BOND_ROTATE + bond/gold defensive. bond_pct+gold_pct should equal 1.0."""
    rebal = monthly_dates(all_dates, start, end)
    cash = CAPITAL; holdings = {}; bu = 0; gu = 0; eq = [cash]; max_eq = cash; dd_active = False
    months_in_cycle = 0; prev_mst = 0
    for rd in rebal:
        mst = state_map.get(rd, 0)
        if mst in (2, 4):
            if prev_mst not in (2, 4): months_in_cycle = 1
            else: months_in_cycle += 1
        else: months_in_cycle = 0
        prev_mst = mst
        base_sm = 1.0 if mst == 2 else (0.5 if mst == 4 else 0.0)
        sm = base_sm * decay_fn(months_in_cycle)
        bp = bond_cl.get(rd); gp = gold_cl.get(rd)
        if sm > 0 and bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0
        if sm > 0 and gu > 0 and gp: cash += gu * gp * (1-COMM); gu = 0
        for sym in list(holdings.keys()):
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: cash += holdings[sym]["val"] * 0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]: holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p / holdings[sym]["hwm"] - 1) * 100 < STOP_PCT:
                cash += holdings[sym]["val"] * (1-COMM); del holdings[sym]
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0) + gu * (gp or 1.0)
        if total > max_eq: max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if bu > 0 and bp: cash += bu * bp * (1-COMM); bu = 0
            if gu > 0 and gp: cash += gu * gp * (1-COMM); gu = 0
            dd_active = True
        elif dd_active and total / max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue
        if sm == 0:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
            if cash > 1000:
                total_cash_for_defensive = cash  # snapshot before any deduction
                if bp:
                    buy_b = int(total_cash_for_defensive * bond_pct * 0.99 / bp)
                    if buy_b > 0: bu += buy_b; cash -= buy_b * bp * (1+COMM)
                if gp:
                    buy_g = int(total_cash_for_defensive * gold_pct * 0.99 / gp)
                    if buy_g > 0: gu += buy_g; cash -= buy_g * gp * (1+COMM)
            eq.append(cash + bu * (bp or 1.0) + gu * (gp or 1.0)); continue
        trending = tc.get(rd, [])
        if not trending: eq.append(cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0) + gu * (gp or 1.0)); continue
        selected = {t[0] for t in trending[:TOP_N]}
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1-COMM); del holdings[sym]
        n_pos = max(len(selected), 1); per = total * sm / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, rd, stock_data)
            if p is None or p < 0.01: continue
            tq = int(per / p / 100) * 100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1+COMM): cash -= cost * (1+COMM); holdings[sym]["qty"] = tq
                    elif diff < 0: cash += cost * (1-COMM); holdings[sym]["qty"] = tq
            else:
                cost = tq * p
                if cash >= cost * (1+COMM): cash -= cost * (1+COMM); holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values()) + bu * (bp or 1.0) + gu * (gp or 1.0)
        eq.append(total)
    return metrics(eq)

def verify_nm():
    storage = DataStorage()
    raw = storage.load_bars("510300")
    cd_ = np.array([r.trade_date for r in raw])
    cc = np.array([r.close for r in raw], dtype=np.float64)
    cv = np.array([r.volume for r in raw], dtype=np.float64)
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
    def precompute(start, end):
        rd = monthly_dates(all_dates, start, end); cache = {}
        for i, d in enumerate(rd):
            t = []
            for code in stock_data:
                d2 = data_at(code, d, 260, stock_data)
                if d2 is None: continue
                cl, hi, lo, vo = d2
                is_t, nc = is_stock_uptrend(cl, hi, lo, vo)
                if is_t and nc >= MIN_TREND: t.append((code, trend_score(cl)))
            t.sort(key=lambda x: x[1], reverse=True); cache[d] = t
        return cache
    tc_test = precompute(TEST_START, TEST_END)
    N_VALS, M_VALS = [3,4,5,6,7], [20,30,40,50,60]
    grid = list(itertools.product(N_VALS, M_VALS))
    results = []
    for N, M in grid:
        sm = build_state_map(cd_, cc, cv, N, M)
        r = run_bond_rotate_decay(sm, tc_test, stock_data, all_dates, bond_cl, TEST_START, TEST_END, lambda m: 1.0)
        results.append({"N": N, "M": M, "cagr": r["cagr"]})
    print(f"\nNxM Test CAGR matrix:")
    print(f"  N\\M  " + "  ".join(f"{m:6d}" for m in M_VALS))
    for N in N_VALS:
        vals = [next(x["cagr"] for x in results if x["N"]==N and x["M"]==M) for M in M_VALS]
        print(f"  {N:>3d}   " + "  ".join(f"{v:+5.1f}%" for v in vals))
    best = max(results, key=lambda x: x["cagr"])
    n5m40 = next(x for x in results if x["N"]==5 and x["M"]==40)
    rank = sorted(results, key=lambda x: x["cagr"], reverse=True).index(n5m40) + 1
    print(f"\nBest: N={best['N']}, M={best['M']} -> C={best['cagr']:+.2f}%")
    print(f"N=5/M=40: rank #{rank}/{len(grid)}  C={n5m40['cagr']:+.2f}%")
    n4_avg = np.mean([x["cagr"] for x in results if x["N"]==4])
    n6_avg = np.mean([x["cagr"] for x in results if x["N"]==6])
    print(f"N=4 avg: {n4_avg:+.2f}%  N=6 avg: {n6_avg:+.2f}%")
    print("VERDICT: N=5 is " + ("CONFIRMED in top cluster" if rank <= 3 else "NOT in top cluster"))

if __name__ == "__main__":
    if "--verify" in sys.argv:
        t0 = time.time(); verify_nm(); print(f"\nDone {time.time()-t0:.0f}s"); sys.exit(0)
    t0 = time.time()
    print("Loading...")
    storage = DataStorage()
    raw = storage.load_bars("510300")
    csi_dates = np.array([r.trade_date for r in raw])
    csi_closes = np.array([r.close for r in raw], dtype=np.float64)
    csi_volumes = np.array([r.volume for r in raw], dtype=np.float64)
    nc = len(csi_closes)
    csi_ret5 = np.full(nc, np.nan)
    for i in range(5, nc): csi_ret5[i] = csi_closes[i] / csi_closes[i-5] - 1.0
    csi_idx_map = {str(d): i for i, d in enumerate(csi_dates)}
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
    sm_all = build_state_map(csi_dates, csi_closes, csi_volumes, N_CONFIRM, M_COOLDOWN, VT, MD)
    a43_fn = lambda m: 1.0 if m <= 4 else (0.75 if m <= 8 else 0.50)
    noop_fn = lambda m: 1.0
    bl_tr = run_baseline(tc_train, stock_data, all_dates, TRAIN_START, TRAIN_END)
    bl_te = run_baseline(tc_test,  stock_data, all_dates, TEST_START,  TEST_END)
    br_tr = run_bond_rotate(sm_all, tc_train, stock_data, all_dates, bond_cl, TRAIN_START, TRAIN_END)
    br_te = run_bond_rotate(sm_all, tc_test,  stock_data, all_dates, bond_cl, TEST_START,  TEST_END)
    br43_tr = run_bond_rotate_decay(sm_all, tc_train, stock_data, all_dates, bond_cl, TRAIN_START, TRAIN_END, a43_fn)
    br43_te = run_bond_rotate_decay(sm_all, tc_test,  stock_data, all_dates, bond_cl, TEST_START,  TEST_END, a43_fn)
    s43_tr = run_sharp_decay(sm_all, tc_train, stock_data, all_dates, bond_cl, TRAIN_START, TRAIN_END, a43_fn, csi_ret5, csi_idx_map)
    s43_te = run_sharp_decay(sm_all, tc_test,  stock_data, all_dates, bond_cl, TEST_START,  TEST_END, a43_fn, csi_ret5, csi_idx_map)
    s0_tr = run_sharp_decay(sm_all, tc_train, stock_data, all_dates, bond_cl, TRAIN_START, TRAIN_END, noop_fn, csi_ret5, csi_idx_map)
    s0_te = run_sharp_decay(sm_all, tc_test,  stock_data, all_dates, bond_cl, TEST_START,  TEST_END, noop_fn, csi_ret5, csi_idx_map)
    # Gold overlay (80% bond + 20% gold in defensive)
    g43_tr = run_gold_decay(sm_all, tc_train, stock_data, all_dates, bond_cl, gold_cl, TRAIN_START, TRAIN_END, a43_fn)
    g43_te = run_gold_decay(sm_all, tc_test,  stock_data, all_dates, bond_cl, gold_cl, TEST_START,  TEST_END, a43_fn)
    print(f"\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}")
    print(f"\n{'Strategy':<40s} | {'Train C':>7s} {'Train S':>7s} {'Train D':>7s} | {'Test C':>7s} {'Test S':>7s} {'Test D':>7s}")
    print("-"*105)
    for name, tr, te in [
        ("BASELINE (120MA binary)",             bl_tr, bl_te),
        ("BOND_ROTATE (entry only)",            br_tr, br_te),
        ("+ A43 decay",                         br43_tr, br43_te),
        ("+ Sharp3pct exit (no decay)",         s0_tr, s0_te),
        ("+ Sharp3pct + A43",                   s43_tr, s43_te),
        ("+ A43 + Gold (80/20)",                g43_tr, g43_te),
    ]:
        print(f"{name:<40s} | {tr['cagr']:+7.2f}% {tr['sharpe']:6.3f} {tr['maxdd']:5.1f}% | {te['cagr']:+7.2f}% {te['sharpe']:6.3f} {te['maxdd']:5.1f}%")
    print(f"\nDelta over BOND_ROTATE entry-only (Test CAGR):")
    print(f"  +A43 decay:              {br43_te['cagr']-br_te['cagr']:+.2f}%")
    print(f"  +Sharp3pct exit:         {s0_te['cagr']-br_te['cagr']:+.2f}%")
    print(f"  +Sharp3pct + A43:        {s43_te['cagr']-br_te['cagr']:+.2f}%")
    print(f"  +A43+Gold (80/20):       {g43_te['cagr']-br_te['cagr']:+.2f}%")
    print(f"\nDone in {time.time()-t0:.0f}s")

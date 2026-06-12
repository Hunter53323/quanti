"""
5 Enhanced Strategies (S5-S9) that build on the working S1-S4 backtest.
Passes precomputed trend data from the main runner.
Train: 2015-2021, Test: 2022-2025.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path(_PROJECT_ROOT) / "data" / "clean"
TEST_START = "2022-01-01"; TEST_END = "2025-12-31"
TRAIN_START = "2015-01-01"; TRAIN_END = "2021-12-31"
RF = 0.03; CAPITAL = 90000.0
S1_TREND = 50000.0; S1_MR_C = 40000.0
TOP_N = 3; STOP_PCT = -0.10; MA_PERIOD = 120

# ---------------------------------------------------------------------------
# Thin wrappers (copied from working script)
# ---------------------------------------------------------------------------
def sma(s, p):
    return s.rolling(p).mean()

def compute_rsi(cl, period=14):
    d = cl.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def _ws(data, period):
    n = len(data); result = np.full(n, np.nan); seed = period
    while seed < n:
        window = data[1:seed+1]; valid = window[~np.isnan(window)]
        if len(valid) > 0: result[seed] = np.mean(valid); break
        seed += 1
    if seed >= n: return result
    prev = result[seed]
    for i in range(seed+1, n):
        cur = data[i]
        if np.isnan(cur): result[i] = prev
        elif np.isnan(prev): prev = cur; result[i] = prev
        else: prev = (cur + (period-1)*prev)/period; result[i] = prev
    return result

def adx_np(hi, lo, cl, period=14):
    n = len(cl)
    if n < period*2: return np.full(n, np.nan)
    tr = np.zeros(n); tr[0] = hi[0] - lo[0]
    for i in range(1, n): tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        u = hi[i]-hi[i-1]; d = lo[i-1]-lo[i]
        if u>d and u>0: pdm[i]=u
        if d>u and d>0: mdm[i]=d
    atr_s = _ws(tr, period); pdi_s = _ws(pdm, period); mdi_s = _ws(mdm, period)
    pdi = np.divide(pdi_s, atr_s, where=atr_s!=0)*100
    mdi = np.divide(mdi_s, atr_s, where=atr_s!=0)*100
    dx = np.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100
    return _ws(dx, period)

def bollinger(cl, period=20, std=2.0):
    mid = sma(cl, period)
    s = cl.rolling(period).std(ddof=1)
    return mid, mid + s*std, mid - s*std

def annual_metrics(daily_ret, years):
    n = len(daily_ret)
    if n < 2: return {"CAGR": 0, "Sharpe": 0, "MaxDD": 0, "Calmar": 0,
                      "AnnVol": 0, "WinRate": 0, "TotalReturn": 0}
    cum = np.cumprod(1 + daily_ret)
    tr = cum[-1] - 1; y = max(years, 0.01)
    cagr = (1+tr)**(1/y) - 1
    ann_vol = np.std(daily_ret, ddof=1)*np.sqrt(252)
    sharpe = (cagr - RF)/max(ann_vol, 1e-10)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum-peak)/peak))
    calmar = cagr/max(abs(max_dd), 1e-10) if max_dd < 0 else 0
    wr = float(np.mean(daily_ret > 0))
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "Calmar": calmar,
            "AnnVol": ann_vol, "WinRate": wr, "TotalReturn": tr}

# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------
def sc(st, c, d):
    if c not in st: return None
    s = st[c][st[c].index <= d]
    return None if len(s) == 0 else float(s["close"].iloc[-1])

def ec(et, c, d):
    if c not in et: return None
    s = et[c][et[c].index <= d]
    return None if len(s) == 0 else float(s["close"].iloc[-1])

# ---------------------------------------------------------------------------
# Trade log & trailing stop
# ---------------------------------------------------------------------------
def log_t(tr, d, s, side, q, p, pnl, tag=""):
    tr.append({"date": d, "symbol": s, "side": side, "qty": q, "price": p,
               "pnl": pnl, "tag": tag})

def trail_stop(positions, cash_ref, st, et, code, d, pct, trades, tag):
    pr = sc(st, code, d)
    if pr is None:
        pr = ec(et, code, d)
    if pr is None:
        return False
    pos = positions[code]
    hwm = pos.get("hwm", pr)
    if pr > hwm:
        hwm = pr
    pos["hwm"] = hwm
    if (pr/hwm - 1) < pct:
        pnl = (pr - pos["avg"]) * pos["qty"]
        cash_ref[0] += pr * pos["qty"]
        log_t(trades, d, code, "sell", pos["qty"], pr, pnl, tag)
        del positions[code]
        return True
    return False

def trade_stats(tr):
    sells = [t for t in (tr or []) if t.get("side") == "sell"]
    if not sells:
        return {"n": 0, "wr": 0, "pf": 0, "aw": 0, "al": 0, "tp": 0}
    w = [t["pnl"] for t in sells if t["pnl"] > 0]
    l = [t["pnl"] for t in sells if t["pnl"] < 0]
    n = len(sells); nw = len(w); nl = len(l)
    sw = sum(w) if w else 0
    sl = abs(sum(l)) if l else 0
    return {"n": n, "wr": nw/n if n else 0,
            "pf": sw/sl if sl > 0 else (999.0 if sw > 0 else 0.0),
            "aw": sw/nw if nw else 0,
            "al": -sl/nl if nl else 0,
            "tp": sum(t["pnl"] for t in sells)}

# ---------------------------------------------------------------------------
# Trend selection (reuses precomputed masks/scores)
# ---------------------------------------------------------------------------
def select_top(masks, scores, month_ends, dt, top_n=TOP_N):
    mi = int(np.searchsorted(month_ends, dt, 'right') - 1)
    if mi < 0:
        return [], {}
    trending = []
    for c in masks:
        if masks[c][mi] and scores[c][mi] > 0:
            trending.append((c, scores[c][mi]))
    trending.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in trending[:top_n]], {t[0]: t[1] for t in trending[:top_n]}


# ====================================================================
# S5: PE-Band Dynamic Allocation
# ====================================================================
def s5_pe_band(st, et, csi, dates, masks, scores, me_arr, trades, pe_df,
               money_etf="511880"):
    """
    CSI300 PE percentile determines equity allocation:
    PE < 30th pctl (cheap) => 80% equity
    30-70th pctl (fair) => 50% equity
    PE > 70th pctl (expensive) => 20% equity
    Equity portion in trend stocks; cash in money ETF.
    """
    dr = []
    cash = CAPITAL
    pos = {}
    peq = CAPITAL

    # Precompute PE percentile (trailing 10-year window)
    pe_vals = pe_df.copy()
    pe_vals["pct"] = pe_vals["pe"].rolling(250 * 10, min_periods=250).apply(
        lambda x: (x <= x.iloc[-1]).mean() * 100, raw=False
    )

    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = (d.month != dates[i - 1].month)

        # Current PE percentile
        pe_now = pe_vals[pe_vals.index <= d]
        pctl = 50.0
        if len(pe_now) > 0 and not pd.isna(pe_now["pct"].iloc[-1]):
            pctl = float(pe_now["pct"].iloc[-1])

        if pctl < 30:
            eq_pct = 0.80
        elif pctl < 70:
            eq_pct = 0.50
        else:
            eq_pct = 0.20

        eq_target = CAPITAL * eq_pct

        # Trailing stops
        cr = [cash]
        for c in list(pos):
            trail_stop(pos, cr, st, et, c, d, STOP_PCT, trades, "s5_t")
        cash = cr[0]

        if is_me:
            sel, _ = select_top(masks, scores, me_arr, d)

            # Sell rotated-out stocks
            for c in list(pos):
                if pos[c].get("type") == "stock" and c not in sel:
                    pr = sc(st, c, d)
                    if pr:
                        pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                        cash += pr * pos[c]["qty"]
                        log_t(trades, d, c, "sell", pos[c]["qty"], pr, pnl, "s5_t")
                        del pos[c]

            # Current allocations
            cur_eq = sum(
                p["qty"] * (sc(st, c, d) or 0)
                for c, p in pos.items()
                if p.get("type") == "stock"
            )
            cur_bd = sum(
                p["qty"] * (ec(et, c, d) or 0)
                for c, p in pos.items()
                if p.get("type") == "bond"
            )

            # Rebalance: stock side
            stock_target = eq_target
            stock_gap = stock_target - cur_eq
            max(len(sel), 1)

            if stock_gap > 0:
                # Buy more stocks
                new_stocks = [s for s in sel if s not in pos]
                if new_stocks:
                    per_stock = stock_gap / len(new_stocks) * 0.92
                    for c in new_stocks:
                        pr = sc(st, c, d)
                        if pr and pr > 0.01 and per_stock >= pr * 100:
                            q = int(per_stock / pr / 100) * 100
                            if q >= 100 and q * pr <= cash:
                                cash -= q * pr
                                pos[c] = {"qty": q, "avg": pr, "hwm": pr, "type": "stock"}
                                log_t(trades, d, c, "buy", q, pr, 0, "s5_t")

            elif stock_gap < 0 and cur_eq > 0:
                # Sell proportionally
                for c in list(pos):
                    if pos[c].get("type") == "stock":
                        pr = sc(st, c, d)
                        if pr:
                            ratio = stock_target / max(cur_eq, 1)
                            target_val = pos[c]["qty"] * pr * ratio
                            sell_val = max(0, pos[c]["qty"] * pr - target_val)
                            sq = int(sell_val / pr / 100) * 100
                            if sq >= 100:
                                pnl = (pr - pos[c]["avg"]) * sq
                                cash += sq * pr
                                pos[c]["qty"] -= sq
                                log_t(trades, d, c, "sell", sq, pr, pnl, "s5_t")
                            if pos[c]["qty"] < 100:
                                pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                                cash += pr * pos[c]["qty"]
                                log_t(trades, d, c, "sell", pos[c]["qty"], pr, pnl, "s5_t")
                                del pos[c]

            # Rebalance: bond side
            bd_target = CAPITAL - eq_target
            bd_gap = bd_target - cur_bd
            mpr = ec(et, money_etf, d)

            if mpr and mpr > 0.01:
                if bd_gap > 100:
                    q = int(bd_gap / mpr / 100) * 100
                    if q >= 100 and q * mpr <= cash:
                        cash -= q * mpr
                        if money_etf in pos:
                            pos[money_etf]["qty"] += q
                        else:
                            pos[money_etf] = {"qty": q, "avg": mpr, "hwm": mpr, "type": "bond"}
                        log_t(trades, d, money_etf, "buy", q, mpr, 0, "s5_bd")

                elif bd_gap < -100 and money_etf in pos:
                    sq = min(int(abs(bd_gap) / mpr / 100) * 100, pos[money_etf]["qty"])
                    if sq >= 100:
                        pnl = (mpr - pos[money_etf]["avg"]) * sq
                        cash += sq * mpr
                        pos[money_etf]["qty"] -= sq
                        log_t(trades, d, money_etf, "sell", sq, mpr, pnl, "s5_bd")
                    if pos[money_etf]["qty"] < 100:
                        del pos[money_etf]

        # Daily equity
        eq = cash
        for c, p in pos.items():
            pr = sc(st, c, d)
            if pr is None:
                pr = ec(et, c, d)
            if pr:
                eq += p["qty"] * pr
        if peq > 0:
            dr.append((eq - peq) / peq)
        peq = eq

    return np.array(dr)


# ====================================================================
# S6: Double-MA Confirmation Filter (enhanced S2)
# ====================================================================
def s6_ma_double(st, et, csi, dates, masks, scores, me_arr, trades):
    """
    Enhanced Core+Sat: market must be above BOTH 60MA AND 120MA.
    Reduces whipsaw entries from single-MA false breakouts.
    """
    dr = []
    core_cash = CAPITAL * 0.6
    sat_cash = CAPITAL * 0.4
    pos = {}
    peq = CAPITAL
    sat_etfs = ["510300", "510500", "159915"]

    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = (d.month != dates[i - 1].month)

        # Double-MA check
        sd = csi[csi.index <= d]
        mkt_ok = False
        if len(sd) >= MA_PERIOD:
            m60 = sma(sd["close"], 60)
            m120 = sma(sd["close"], MA_PERIOD)
            if not pd.isna(m60.iloc[-1]) and not pd.isna(m120.iloc[-1]):
                price_now = float(sd["close"].iloc[-1])
                mkt_ok = (price_now > float(m60.iloc[-1])
                         and price_now > float(m120.iloc[-1]))

        # Stops
        for c in list(pos):
            tag = "s6_c" if pos[c].get("type") == "core" else "s6_s"
            cr = [core_cash] if pos[c].get("type") == "core" else [sat_cash]
            trail_stop(pos, cr, st, et, c, d, STOP_PCT, trades, tag)
            if pos[c].get("type") == "core":
                core_cash = cr[0]
            else:
                sat_cash = cr[0]

        if not mkt_ok:
            for c in list(pos):
                pr = sc(st, c, d)
                if pr is None:
                    pr = ec(et, c, d)
                if pr:
                    pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                    cat = "s6_c" if pos[c].get("type") == "core" else "s6_s"
                    if pos[c].get("type") == "core":
                        core_cash += pr * pos[c]["qty"]
                    else:
                        sat_cash += pr * pos[c]["qty"]
                    log_t(trades, d, c, "sell", pos[c]["qty"], pr, pnl, cat)
                    del pos[c]
        elif is_me:
            sel, _ = select_top(masks, scores, me_arr, d)

            # Sell rotated-out core
            for c in list(pos):
                if pos[c].get("type") == "core" and c not in sel:
                    pr = sc(st, c, d)
                    if pr:
                        pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                        core_cash += pr * pos[c]["qty"]
                        log_t(trades, d, c, "sell", pos[c]["qty"], pr, pnl, "s6_c")
                        del pos[c]

            # Buy new core
            nw = [s for s in sel if s not in pos]
            if nw and core_cash > 0:
                per = core_cash / max(len(nw), 1) * 0.92
                for c in nw:
                    pr = sc(st, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= core_cash:
                            core_cash -= q * pr
                            pos[c] = {"qty": q, "avg": pr, "hwm": pr, "type": "core"}
                            log_t(trades, d, c, "buy", q, pr, 0, "s6_c")

            # Satellite rebalance
            n_sat = len(sat_etfs)
            sat_total = sat_cash
            for c, p in list(pos.items()):
                if p.get("type") == "sat":
                    epr = ec(et, c, d)
                    if epr:
                        sat_total += p["qty"] * epr
            per_etf = sat_total / n_sat

            for code in sat_etfs:
                epr = ec(et, code, d)
                if epr is None or epr <= 0:
                    continue
                if code in pos:
                    p = pos[code]
                    cv = p["qty"] * epr
                    diff = per_etf - cv
                    if diff > epr * 100:
                        q = int(diff / epr / 100) * 100
                        if q >= 100 and q * epr <= sat_cash:
                            sat_cash -= q * epr
                            p["qty"] += q
                            log_t(trades, d, code, "buy", q, epr, 0, "s6_s")
                    elif diff < -epr * 100:
                        q = min(int(abs(diff) / epr / 100) * 100, p["qty"])
                        if q >= 100:
                            pnl = (epr - p["avg"]) * q
                            sat_cash += q * epr
                            p["qty"] -= q
                            log_t(trades, d, code, "sell", q, epr, pnl, "s6_s")
                            if p["qty"] == 0:
                                del pos[code]
                else:
                    q = int(per_etf / epr / 100) * 100
                    if q >= 100 and q * epr <= sat_cash:
                        sat_cash -= q * epr
                        pos[code] = {"qty": q, "avg": epr, "hwm": epr, "type": "sat"}
                        log_t(trades, d, code, "buy", q, epr, 0, "s6_s")

        eq = core_cash + sat_cash
        for c, p in pos.items():
            pr = sc(st, c, d)
            if pr is None:
                pr = ec(et, c, d)
            if pr:
                eq += p["qty"] * pr
        if peq > 0:
            dr.append((eq - peq) / peq)
        peq = eq

    return np.array(dr)


# ====================================================================
# S7: MR + Bollinger Band Confirmation (enhanced S1's MR)
# ====================================================================
def s7_mr_bb(st, et, csi, dates, masks, scores, me_arr, trades):
    """
    Enhanced S1: MR entry requires RSI<30 AND price < lower Bollinger Band (2-sigma).
    This avoids catching falling knives.
    Trend component identical to S1.
    """
    dr = []
    tc = S1_TREND
    mc = S1_MR_C
    tp = {}
    mp = {}
    tpr = S1_TREND
    mpr = S1_MR_C

    for i, d in enumerate(dates):
        if i == 0:
            continue

        sd = csi[csi.index <= d]
        bull = False
        if len(sd) >= MA_PERIOD:
            ma = sma(sd["close"], MA_PERIOD)
            if not pd.isna(ma.iloc[-1]):
                bull = float(sd["close"].iloc[-1]) > float(ma.iloc[-1])
        is_me = (d.month != dates[i - 1].month)

        # ---- Trend Strategy (same as S1) ----
        tcr = [tc]
        for c in list(tp):
            trail_stop(tp, tcr, st, et, c, d, STOP_PCT, trades, "s7_t")
        tc = tcr[0]

        if bull and is_me:
            sel, _ = select_top(masks, scores, me_arr, d)
            for c in list(tp):
                if c not in sel:
                    pr = sc(st, c, d)
                    if pr:
                        pnl = (pr - tp[c]["avg"]) * tp[c]["qty"]
                        tc += pr * tp[c]["qty"]
                        log_t(trades, d, c, "sell", tp[c]["qty"], pr, pnl, "s7_t")
                        del tp[c]
            nw = [s for s in sel if s not in tp]
            if nw and tc > 0:
                per = tc / max(len(sel), 1) * 0.92
                for c in nw:
                    pr = sc(st, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= tc:
                            tc -= q * pr
                            tp[c] = {"qty": q, "avg": pr, "hwm": pr}
                            log_t(trades, d, c, "buy", q, pr, 0, "s7_t")

        if not bull:
            for c in list(tp):
                pr = sc(st, c, d)
                if pr:
                    pnl = (pr - tp[c]["avg"]) * tp[c]["qty"]
                    tc += pr * tp[c]["qty"]
                    log_t(trades, d, c, "sell", tp[c]["qty"], pr, pnl, "s7_t")
                    del tp[c]

        te = tc + sum(
            sc(st, c, d) * p["qty"]
            for c, p in tp.items()
            if sc(st, c, d)
        )

        # ---- MR with BB Confirmation ----
        if not bull:
            mr_etfs = ["510300", "510500", "159915"]
            for code in mr_etfs:
                if code not in et:
                    continue
                sub = et[code][et[code].index <= d]
                if len(sub) < 30:
                    continue
                cl = sub["close"]
                rsi_v = float(compute_rsi(cl, 14).iloc[-1])
                if pd.isna(rsi_v):
                    continue

                if code in mp:
                    pr = float(cl.iloc[-1])
                    hwm = mp[code].get("hwm", pr)
                    if pr > hwm:
                        hwm = pr
                    mp[code]["hwm"] = hwm
                    exited = False
                    if rsi_v > 70 or (pr / hwm - 1.0) < STOP_PCT:
                        pnl = (pr - mp[code]["avg"]) * mp[code]["qty"]
                        mc += pr * mp[code]["qty"]
                        log_t(trades, d, code, "sell", mp[code]["qty"], pr, pnl, "s7_m")
                        del mp[code]
                        exited = True
                    if exited:
                        continue

                # Entry: RSI<30 AND price < BB lower band
                if rsi_v < 30 and mc > 0:
                    _, _, bb_lower = bollinger(cl, 20, 2.0)
                    if not pd.isna(bb_lower.iloc[-1]):
                        if float(cl.iloc[-1]) < float(bb_lower.iloc[-1]):
                            pr = float(cl.iloc[-1])
                            inv = mc * 0.5
                            if inv >= pr * 100:
                                q = int(inv / pr / 100) * 100
                                if q >= 100 and q * pr <= mc:
                                    mc -= q * pr
                                    mp[code] = {"qty": q, "avg": pr, "hwm": pr}
                                    log_t(trades, d, code, "buy", q, pr, 0, "s7_m")
        else:
            for c in list(mp):
                pr = ec(et, c, d)
                if pr:
                    pnl = (pr - mp[c]["avg"]) * mp[c]["qty"]
                    mc += pr * mp[c]["qty"]
                    log_t(trades, d, c, "sell", mp[c]["qty"], pr, pnl, "s7_m")
                    del mp[c]

        meq = mc + sum(
            ec(et, c, d) * p["qty"]
            for c, p in mp.items()
            if ec(et, c, d)
        )

        tn = te + meq
        tpv = tpr + mpr
        if tpv > 0:
            dr.append((tn - tpv) / tpv)
        tpr = te
        mpr = meq

    return np.array(dr)


# ====================================================================
# S8: Cash Management (enhanced S2 with money ETF)
# ====================================================================
def s8_cash_mgmt(st, et, csi, dates, masks, scores, me_arr, trades,
                 money_etf="511880"):
    """
    Enhanced S2: when market is below 120MA, hold money ETF (511880) instead
    of idle cash. ~2-3% annualized yield on idle cash.
    """
    dr = []
    core_cash = CAPITAL * 0.6
    sat_cash = CAPITAL * 0.4
    pos = {}
    peq = CAPITAL
    sat_etfs = ["510300", "510500", "159915"]

    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = (d.month != dates[i - 1].month)

        sd = csi[csi.index <= d]
        mkt_ok = False
        if len(sd) >= MA_PERIOD:
            ma = sma(sd["close"], MA_PERIOD)
            if not pd.isna(ma.iloc[-1]):
                mkt_ok = float(sd["close"].iloc[-1]) > float(ma.iloc[-1])

        # Stops
        for c in list(pos):
            typ = pos[c].get("type")
            if typ == "core":
                tag = "s8_c"
                cr = [core_cash]
            elif typ == "sat":
                tag = "s8_s"
                cr = [sat_cash]
            else:
                tag = "s8_bd"
                cr = [sat_cash]
            trail_stop(pos, cr, st, et, c, d, STOP_PCT, trades, tag)
            if typ == "core":
                core_cash = cr[0]
            elif typ in ("sat", "bond"):
                sat_cash = cr[0]

        if not mkt_ok:
            # Liquidate all equity positions
            for c in list(pos):
                if pos[c].get("type") != "bond":
                    pr = sc(st, c, d)
                    if pr is None:
                        pr = ec(et, c, d)
                    if pr:
                        pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                        if pos[c].get("type") == "core":
                            core_cash += pr * pos[c]["qty"]
                        else:
                            sat_cash += pr * pos[c]["qty"]
                        log_t(trades, d, c, "sell", pos[c]["qty"], pr, pnl,
                              "s8_c" if pos[c].get("type") == "core" else "s8_s")
                        del pos[c]

            # Buy money ETF with satellite cash
            mpr = ec(et, money_etf, d)
            if mpr and mpr > 0.01 and sat_cash > 0:
                q = int(sat_cash / mpr / 100) * 100
                if q >= 100:
                    sat_cash -= q * mpr
                    pos[money_etf] = {"qty": q, "avg": mpr, "hwm": mpr, "type": "bond"}
                    log_t(trades, d, money_etf, "buy", q, mpr, 0, "s8_bd")

        elif is_me:
            # Sell money ETF first
            if money_etf in pos:
                mpr = ec(et, money_etf, d)
                if mpr:
                    pnl = (mpr - pos[money_etf]["avg"]) * pos[money_etf]["qty"]
                    sat_cash += mpr * pos[money_etf]["qty"]
                    log_t(trades, d, money_etf, "sell", pos[money_etf]["qty"],
                          mpr, pnl, "s8_bd")
                    del pos[money_etf]

            sel, _ = select_top(masks, scores, me_arr, d)

            # Core: sell rotated-out, buy new
            for c in list(pos):
                if pos[c].get("type") == "core" and c not in sel:
                    pr = sc(st, c, d)
                    if pr:
                        pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                        core_cash += pr * pos[c]["qty"]
                        log_t(trades, d, c, "sell", pos[c]["qty"], pr, pnl, "s8_c")
                        del pos[c]

            nw = [s for s in sel if s not in pos]
            if nw and core_cash > 0:
                per = core_cash / max(len(nw), 1) * 0.92
                for c in nw:
                    pr = sc(st, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= core_cash:
                            core_cash -= q * pr
                            pos[c] = {"qty": q, "avg": pr, "hwm": pr, "type": "core"}
                            log_t(trades, d, c, "buy", q, pr, 0, "s8_c")

            # Satellite rebalance
            n_sat = len(sat_etfs)
            sat_total = sat_cash
            for c, p in list(pos.items()):
                if p.get("type") == "sat":
                    epr = ec(et, c, d)
                    if epr:
                        sat_total += p["qty"] * epr
            per_etf = sat_total / n_sat

            for code in sat_etfs:
                epr = ec(et, code, d)
                if epr is None or epr <= 0:
                    continue
                if code in pos:
                    p = pos[code]
                    cv = p["qty"] * epr
                    diff = per_etf - cv
                    if diff > epr * 100:
                        q = int(diff / epr / 100) * 100
                        if q >= 100 and q * epr <= sat_cash:
                            sat_cash -= q * epr
                            p["qty"] += q
                            log_t(trades, d, code, "buy", q, epr, 0, "s8_s")
                    elif diff < -epr * 100:
                        q = min(int(abs(diff) / epr / 100) * 100, p["qty"])
                        if q >= 100:
                            pnl = (epr - p["avg"]) * q
                            sat_cash += q * epr
                            p["qty"] -= q
                            log_t(trades, d, code, "sell", q, epr, pnl, "s8_s")
                            if p["qty"] == 0:
                                del pos[code]
                else:
                    q = int(per_etf / epr / 100) * 100
                    if q >= 100 and q * epr <= sat_cash:
                        sat_cash -= q * epr
                        pos[code] = {"qty": q, "avg": epr, "hwm": epr, "type": "sat"}
                        log_t(trades, d, code, "buy", q, epr, 0, "s8_s")

        eq = core_cash + sat_cash
        for c, p in pos.items():
            pr = sc(st, c, d)
            if pr is None:
                pr = ec(et, c, d)
            if pr:
                eq += p["qty"] * pr
        if peq > 0:
            dr.append((eq - peq) / peq)
        peq = eq

    return np.array(dr)


# ====================================================================
# S9: Dynamic Weight Dual Parallel (enhanced S4)
# ====================================================================
def s9_dynamic_weight(st, et, csi, dates, masks, scores, me_arr, trades):
    """
    Enhanced S4: allocate capital between trend strategy (A) and ETF rotation
    strategy (B) based on 12-month rolling Sharpe ratio.
    Floor: 30%, Ceiling: 70%.
    """
    rot_etfs = ["510300", "510500", "159915", "510880", "518880"]
    dr = []
    ca = CAPITAL / 2.0
    cb = CAPITAL / 2.0
    pa = {}
    pb = {}
    peq = CAPITAL

    # Rolling returns per sub-strategy
    ra = []
    rb = []
    ea_hist = [CAPITAL / 2.0]
    eb_hist = [CAPITAL / 2.0]

    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = (d.month != dates[i - 1].month)

        # Stops
        car = [ca]
        cbr = [cb]
        for c in list(pa):
            trail_stop(pa, car, st, et, c, d, STOP_PCT, trades, "s9_t")
        for c in list(pb):
            trail_stop(pb, cbr, st, et, c, d, STOP_PCT, trades, "s9_r")
        ca = car[0]
        cb = cbr[0]

        if is_me:
            # Update rolling sub-strategy returns
            da_eq = ca + sum(
                p["qty"] * sc(st, c, d)
                for c, p in pa.items()
                if sc(st, c, d)
            )
            db_eq = cb + sum(
                p["qty"] * ec(et, c, d)
                for c, p in pb.items()
                if ec(et, c, d)
            )

            if ea_hist[-1] > 0:
                ra.append((da_eq - ea_hist[-1]) / ea_hist[-1])
            if eb_hist[-1] > 0:
                rb.append((db_eq - eb_hist[-1]) / eb_hist[-1])
            ea_hist.append(da_eq)
            eb_hist.append(db_eq)

            # Trim to 12 months
            if len(ra) > 12:
                ra = ra[-12:]
            if len(rb) > 12:
                rb = rb[-12:]

            # Compute rolling Sharpe
            def roll_sharpe(rets):
                if len(rets) < 3:
                    return 0.0
                arr = np.array(rets)
                avg = np.mean(arr)
                std = np.std(arr)
                if std < 1e-10:
                    return 0.0
                return avg / std * np.sqrt(12)

            sa = roll_sharpe(ra)
            roll_sharpe(rb)

            # Map Sharpe [-2, 2] to weight [0.3, 0.7]
            def map_weight(sh):
                return max(0.3, min(0.7, 0.5 + sh * 0.1))

            wa = map_weight(sa)
            wb = 1.0 - wa

            # Rebalance capital
            total_eq = ca + cb
            total_eq += sum(
                p["qty"] * sc(st, c, d)
                for c, p in pa.items()
                if sc(st, c, d)
            )
            total_eq += sum(
                p["qty"] * ec(et, c, d)
                for c, p in pb.items()
                if ec(et, c, d)
            )

            target_a = total_eq * wa
            total_eq * wb
            cur_a = ca + sum(
                p["qty"] * sc(st, c, d)
                for c, p in pa.items()
                if sc(st, c, d)
            )
            cb + sum(
                p["qty"] * ec(et, c, d)
                for c, p in pb.items()
                if ec(et, c, d)
            )

            transfer = target_a - cur_a
            if transfer > 0 and cb > 0:
                tr = min(transfer, cb)
                cb -= tr
                ca += tr
            elif transfer < 0 and ca > 0:
                tr = min(-transfer, ca)
                ca -= tr
                cb += tr

            # ---- Strategy A: Trend Stocks ----
            sel, _ = select_top(masks, scores, me_arr, d)
            for c in list(pa):
                if c not in sel:
                    pr = sc(st, c, d)
                    if pr:
                        pnl = (pr - pa[c]["avg"]) * pa[c]["qty"]
                        ca += pr * pa[c]["qty"]
                        log_t(trades, d, c, "sell", pa[c]["qty"], pr, pnl, "s9_t")
                        del pa[c]

            nwa = [s for s in sel if s not in pa]
            if nwa and ca > 0:
                per = ca / max(len(sel), 1) * 0.92
                for c in nwa:
                    pr = sc(st, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= ca:
                            ca -= q * pr
                            pa[c] = {"qty": q, "avg": pr, "hwm": pr}
                            log_t(trades, d, c, "buy", q, pr, 0, "s9_t")

            # ---- Strategy B: ETF Rotation ----
            eret = {}
            for code in rot_etfs:
                if code not in et:
                    continue
                sub = et[code][et[code].index <= d]
                lbd = d - pd.DateOffset(months=3)
                sub_lb = sub[sub.index <= lbd]
                if len(sub_lb) == 0 or len(sub) < 2:
                    continue
                past = float(sub_lb["close"].iloc[-1])
                cur = float(sub["close"].iloc[-1])
                if past > 0:
                    eret[code] = (cur - past) / past

            ranked = sorted(eret.items(), key=lambda x: x[1], reverse=True)
            top2 = set(r[0] for r in ranked[:2])

            for c in list(pb):
                if c not in top2:
                    pr = ec(et, c, d)
                    if pr:
                        pnl = (pr - pb[c]["avg"]) * pb[c]["qty"]
                        cb += pr * pb[c]["qty"]
                        log_t(trades, d, c, "sell", pb[c]["qty"], pr, pnl, "s9_r")
                        del pb[c]

            nwb = [s for s in top2 if s not in pb]
            if nwb and cb > 0:
                per = cb / max(len(top2), 1) * 0.92
                for c in nwb:
                    pr = ec(et, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= cb:
                            cb -= q * pr
                            pb[c] = {"qty": q, "avg": pr, "hwm": pr}
                            log_t(trades, d, c, "buy", q, pr, 0, "s9_r")

        eq = ca + cb
        for c, p in pa.items():
            pr = sc(st, c, d)
            if pr:
                eq += p["qty"] * pr
        for c, p in pb.items():
            pr = ec(et, c, d)
            if pr:
                eq += p["qty"] * pr
        if peq > 0:
            dr.append((eq - peq) / peq)
        peq = eq

    return np.array(dr)

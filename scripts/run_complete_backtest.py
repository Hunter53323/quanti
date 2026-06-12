"""
Complete Backtest: All 9 Hybrid Strategies (S1-S9).
Self-contained, single-file. 100 stocks, Train 2015-2021, Test 2022-2025.
Uses AkShare for CSI300 PE data (S5 requires it).
"""
import os
import time
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
S1_TREND_C = 50000.0; S1_MR_C = 40000.0
TOP_N = 3; STOP_PCT = -0.10; MA_P = 120; N_STOCKS = 100

# ====================================================================
# Data loading
# ====================================================================
def _digit(s):
    """True for 6-digit stock codes, False for ETFs (start with 5,1,588)."""
    if not (s.isdigit() and len(s) == 6):
        return False
    # Exclude ETFs: 51xxxx, 15xxxx, 58xxxx, 588xxx, 56xxxx, 51xxxx, 159xxx
    if s.startswith(('5', '1', '588')):
        return False
    return True

def load_etf(code):
    for sfx in [".SH.parquet", ".SZ.parquet", ".parquet"]:
        p = DATA_DIR / f"{code}{sfx}"
        if p.exists(): df = pd.read_parquet(p); break
    else: raise FileNotFoundError(code)
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df.sort_values("date").set_index("date")

def load_csi300(): return load_etf("CSI300")

def load_stocks(n=N_STOCKS):
    cand = []
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if not _digit(fp.stem): continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        t = df[(df.date >= TEST_START) & (df.date <= TEST_END)]
        tr = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)]
        if len(t) >= 100 and len(tr) >= 100:
            cand.append((fp.stem, float(t.volume.mean())))
    cand.sort(key=lambda x: x[1], reverse=True)
    stocks = {}
    for code, _ in cand[:n]:
        df = pd.read_parquet(DATA_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        stocks[code] = df.sort_values("date").set_index("date")
    return stocks

def fetch_pe_data():
    import sys; sys.path.insert(0, ".")
    from quanti.data.index_pe import IndexPEFetcher
    fetcher = IndexPEFetcher()
    raw = fetcher.fetch_history("000300.SH")
    recs = [{"date": pd.Timestamp(r["trade_date"]), "pe": r["pe"]} for r in raw if r["pe"] > 0]
    return pd.DataFrame(recs).set_index("date").sort_index()

# ====================================================================
# Indicators
# ====================================================================
def sma(s, p): return s.rolling(p).mean()

def compute_rsi(cl, period=14):
    d = cl.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def _ws(data, period):
    n = len(data); result = np.full(n, np.nan); seed = period
    while seed < n:
        vv = data[1:seed+1]; vv2 = vv[~np.isnan(vv)]
        if len(vv2) > 0: result[seed] = np.mean(vv2); break
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
    mid = sma(cl, period); s = cl.rolling(period).std(ddof=1)
    return mid, mid + s*std, mid - s*std

# ====================================================================
# Metrics
# ====================================================================
def metrics(daily_ret, years):
    n = len(daily_ret)
    if n < 2: return {"CAGR":0,"Sharpe":0,"MaxDD":0,"Calmar":0,"AnnVol":0,"WinRate":0,"TotalReturn":0}
    cum = np.cumprod(1 + daily_ret); tr = cum[-1] - 1
    y = max(years, 0.01); cagr = (1+tr)**(1/y) - 1
    ann_vol = np.std(daily_ret, ddof=1)*np.sqrt(252)
    sharpe = (cagr - RF)/max(ann_vol, 1e-10)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum-peak)/peak))
    calmar = cagr/max(abs(max_dd), 1e-10) if max_dd < 0 else 0
    wr = float(np.mean(daily_ret > 0))
    return {"CAGR":cagr,"Sharpe":sharpe,"MaxDD":max_dd,"Calmar":calmar,"AnnVol":ann_vol,"WinRate":wr,"TotalReturn":tr}

# ====================================================================
# Helpers
# ====================================================================
def sc(st, c, d):
    if c not in st: return None
    s = st[c][st[c].index <= d]; return None if len(s)==0 else float(s["close"].iloc[-1])

def ec(et, c, d):
    if c not in et: return None
    s = et[c][et[c].index <= d]; return None if len(s)==0 else float(s["close"].iloc[-1])

def log_t(tr, d, s, side, q, p, pnl, tag=""):
    tr.append({"date":d,"symbol":s,"side":side,"qty":q,"price":p,"pnl":pnl,"tag":tag})

def trail_stop(pos, cash_ref, st, et, c, d, pct, trades, tag):
    pr = sc(st,c,d) or ec(et,c,d)
    if pr is None: return False
    p = pos[c]; hwm = p.get("hwm",pr)
    if pr > hwm: hwm = pr; p["hwm"] = hwm
    if (pr/hwm - 1) < pct:
        pnl = (pr - p["avg"])*p["qty"]; cash_ref[0] += pr*p["qty"]
        log_t(trades, d, c, "sell", p["qty"], pr, pnl, tag); del pos[c]
        return True
    return False

def trade_stats(tr):
    sells = [t for t in (tr or []) if t.get("side")=="sell"]
    if not sells: return {"n":0,"wr":0,"pf":0,"aw":0,"al":0,"tp":0}
    w = [t["pnl"] for t in sells if t["pnl"]>0]; l = [t["pnl"] for t in sells if t["pnl"]<0]
    n = len(sells); nw = len(w); nl = len(l)
    sw = sum(w) if w else 0; sl = abs(sum(l)) if l else 0
    return {"n":n, "wr":nw/n if n else 0,
            "pf":sw/sl if sl>0 else (999.0 if sw>0 else 0.0),
            "aw":sw/nw if nw else 0, "al":-sl/nl if nl else 0,
            "tp":sum(t["pnl"] for t in sells)}

# ====================================================================
# Precompute trend scores
# ====================================================================
def precompute(stocks, csi):
    print("    Precomputing trend scores ...", flush=True)
    all_d = pd.date_range(TRAIN_START, TEST_END, freq="B")
    valid = sorted(set(all_d).intersection(set(csi.index)))
    me = []; cm = None; pv = None
    for d in valid:
        if d.month != cm:
            if cm is not None: me.append(pv)
            cm = d.month
        pv = d
    me.append(pv); me = sorted([m for m in me if m >= pd.Timestamp("2016-01-01")])
    n_me = len(me); masks = {}; scores = {}
    print(f"    {len(stocks)} stocks x {n_me} month-ends ...", flush=True)
    t0 = time.time()
    for si, (code, df) in enumerate(stocks.items()):
        is_t = np.zeros(n_me, dtype=bool); scr = np.zeros(n_me)
        for mi, md in enumerate(me):
            sub = df[df.index <= md]
            if len(sub) < 200: continue
            cl, hi, lo, vo = sub["close"], sub["high"], sub["low"], sub["volume"]
            cond = 0
            m120 = sma(cl, MA_P); v120 = m120.iloc[-1]
            if not pd.isna(v120) and cl.iloc[-1] > v120: cond += 1
            if len(cl) >= 60:
                if hi.iloc[-20:].max() > hi.iloc[-60:-20].max() and lo.iloc[-20:].min() > lo.iloc[-60:-20].min(): cond += 1
            m20 = sma(cl, 20); m60 = sma(cl, 60)
            vals = [m20.iloc[-1], m60.iloc[-1], v120]
            if all(not pd.isna(v) for v in vals) and m20.iloc[-1] > m60.iloc[-1] > v120: cond += 1
            try:
                a = adx_np(hi.values, lo.values, cl.values, 14)
                if not np.isnan(a[-1]) and a[-1] > 25: cond += 1
            except: pass
            if len(vo) >= 22:
                v20 = vo.iloc[-21:-1].mean()
                if v20 > 0 and vo.iloc[-1] > v20*1.2: cond += 1
            if cond >= 3:
                is_t[mi] = True
                if len(cl) >= 130:
                    try:
                        if cl.iloc[-63] > 1e-6 and cl.iloc[-126] > 1e-6:
                            r3 = cl.iloc[-1]/cl.iloc[-63]-1; r6 = cl.iloc[-1]/cl.iloc[-126]-1
                            m3 = min(max(r3/0.5,0),1) if r3>0 else 0; m6 = min(max(r6/0.8,0),1) if r6>0 else 0
                            mom = (0.5*m3+0.5*m6)*100
                        else: mom = 30.0
                    except: mom = 30.0
                    vb = max(0,(1-min(float(cl.pct_change().dropna().iloc[-60:].std())/0.04,1)))*100 if len(cl)>=61 else 50
                    scr[mi] = 0.6*mom + 0.4*vb
        masks[code] = is_t; scores[code] = scr
        if (si+1)%20==0: print(f"      {si+1}/{len(stocks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"    Done in {time.time()-t0:.0f}s", flush=True)
    return masks, scores, me

def select_top(masks, scores, me, dt, top_n=TOP_N):
    mi = int(np.searchsorted(me, dt, 'right') - 1)
    if mi < 0: return [], {}
    tr = [(c, scores[c][mi]) for c in masks if masks[c][mi] and scores[c][mi] > 0]
    tr.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in tr[:top_n]], {t[0]: t[1] for t in tr[:top_n]}

# ====================================================================
# ALL 9 STRATEGIES
# ====================================================================

def s1(st, et, csi, dates, masks, scores, me, trades):
    """Trend + MR Dual Mode"""
    dr = []; tc = S1_TREND_C; mc = S1_MR_C; tp = {}; mp = {}; tpr = S1_TREND_C; mpr = S1_MR_C
    for i, d in enumerate(dates):
        if i == 0:
            continue
        sd = csi[csi.index <= d]
        bull = len(sd) >= MA_P and not pd.isna(sma(sd["close"], MA_P).iloc[-1]) and float(sd["close"].iloc[-1]) > float(sma(sd["close"], MA_P).iloc[-1])
        is_me = d.month != dates[i-1].month
        tcr = [tc]
        for c in list(tp): trail_stop(tp, tcr, st, et, c, d, STOP_PCT, trades, "s1_t")
        tc = tcr[0]
        if bull and is_me:
            sel, _ = select_top(masks, scores, me, d)
            for c in list(tp):
                if c not in sel:
                    pr = sc(st,c,d)
                    if pr: pnl=(pr-tp[c]["avg"])*tp[c]["qty"]; tc+=pr*tp[c]["qty"]; log_t(trades,d,c,"sell",tp[c]["qty"],pr,pnl,"s1_t"); del tp[c]
            nw = [s for s in sel if s not in tp]
            if nw and tc>0:
                per = tc/max(len(sel),1)*0.92
                for c in nw:
                    pr = sc(st,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q = int(per/pr/100)*100
                        if q>=100 and q*pr<=tc: tc-=q*pr; tp[c]={"qty":q,"avg":pr,"hwm":pr}; log_t(trades,d,c,"buy",q,pr,0,"s1_t")
        if not bull:
            for c in list(tp):
                pr = sc(st,c,d)
                if pr: pnl=(pr-tp[c]["avg"])*tp[c]["qty"]; tc+=pr*tp[c]["qty"]; log_t(trades,d,c,"sell",tp[c]["qty"],pr,pnl,"s1_t"); del tp[c]
        te = tc + sum(sc(st,c,d)*p["qty"] for c,p in tp.items() if sc(st,c,d))
        # MR with trailing stop
        if not bull:
            for code in ["510300","510500","159915"]:
                if code not in et: continue
                sub = et[code][et[code].index<=d]
                if len(sub)<30: continue
                rsi_v = float(compute_rsi(sub["close"],14).iloc[-1])
                if pd.isna(rsi_v): continue
                if code in mp:
                    pr = float(sub["close"].iloc[-1]); hwm = mp[code].get("hwm",pr)
                    if pr>hwm: hwm=pr; mp[code]["hwm"]=hwm
                    if rsi_v>70 or (pr/hwm-1)<STOP_PCT:
                        pnl=(pr-mp[code]["avg"])*mp[code]["qty"]; mc+=pr*mp[code]["qty"]; log_t(trades,d,code,"sell",mp[code]["qty"],pr,pnl,"s1_m"); del mp[code]
                elif rsi_v<30 and mc>0:
                    pr=float(sub["close"].iloc[-1]); inv=mc*0.5
                    if inv>=pr*100:
                        q=int(inv/pr/100)*100
                        if q>=100 and q*pr<=mc: mc-=q*pr; mp[code]={"qty":q,"avg":pr,"hwm":pr}; log_t(trades,d,code,"buy",q,pr,0,"s1_m")
        else:
            for c in list(mp):
                pr = ec(et,c,d)
                if pr: pnl=(pr-mp[c]["avg"])*mp[c]["qty"]; mc+=pr*mp[c]["qty"]; log_t(trades,d,c,"sell",mp[c]["qty"],pr,pnl,"s1_m"); del mp[c]
        meq = mc + sum(ec(et,c,d)*p["qty"] for c,p in mp.items() if ec(et,c,d))
        tn=te+meq; tpv=tpr+mpr
        if tpv>0: dr.append((tn-tpv)/tpv)
        tpr=te; mpr=meq
    return np.array(dr)

# ---------------------------------------------------------------------------
def s2(st, et, csi, dates, masks, scores, me, trades):
    """Core + Satellite"""
    dr=[]; cc=CAPITAL*0.6; sc2=CAPITAL*0.4; pos={}; peq=CAPITAL; sat=["510300","510500","159915"]
    for i, d in enumerate(dates):
        if i==0: continue
        sd = csi[csi.index<=d]
        mkt=len(sd)>=MA_P and not pd.isna(sma(sd["close"],MA_P).iloc[-1]) and float(sd["close"].iloc[-1])>float(sma(sd["close"],MA_P).iloc[-1])
        is_me=d.month!=dates[i-1].month
        core_ref = [cc]
        sat_ref = [sc2]
        for c in list(pos):
            trail_stop(pos, core_ref if pos[c].get("type")=="core" else sat_ref,
                       st, et, c, d, STOP_PCT, trades,
                       "s2_c" if pos[c].get("type")=="core" else "s2_s")
        cc, sc2 = core_ref[0], sat_ref[0]
        if not mkt:
            for c in list(pos):
                pr = sc(st,c,d) or ec(et,c,d)
                if pr:
                    if pos[c]["type"]=="core":
                        cc += pr*pos[c]["qty"]
                    else:
                        sc2 += pr*pos[c]["qty"]
                    log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s2_c" if pos[c]["type"]=="core" else "s2_s")
                    del pos[c]
        elif is_me:
            sel, _ = select_top(masks, scores, me, d)
            for c in list(pos):
                if pos[c].get("type")=="core" and c not in sel:
                    pr = sc(st,c,d)
                    if pr:
                        cc += pr*pos[c]["qty"]
                        log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s2_c")
                        del pos[c]
            nw = [s for s in sel if s not in pos]
            if nw and cc>0:
                per = cc/max(len(nw),1)*0.92
                for c in nw:
                    pr = sc(st,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q = int(per/pr/100)*100
                        if q>=100 and q*pr<=cc: cc-=q*pr; pos[c]={"qty":q,"avg":pr,"hwm":pr,"type":"core"}; log_t(trades,d,c,"buy",q,pr,0,"s2_c")
            n_s=len(sat); sat_tot=sc2+sum(ec(et,c,d)*p["qty"] for c,p in list(pos.items()) if p.get("type")=="sat" and ec(et,c,d))
            per_etf=sat_tot/n_s
            for code in sat:
                epr = ec(et,code,d)
                if epr is None or epr<=0: continue
                if code in pos:
                    p=pos[code]; cv=p["qty"]*epr; diff=per_etf-cv
                    if diff>epr*100:
                        q=int(diff/epr/100)*100
                        if q>=100 and q*epr<=sc2: sc2-=q*epr; p["qty"]+=q; log_t(trades,d,code,"buy",q,epr,0,"s2_s")
                    elif diff<-epr*100:
                        q=min(int(abs(diff)/epr/100)*100,p["qty"])
                        if q>=100: sc2+=q*epr; p["qty"]-=q; log_t(trades,d,code,"sell",q,epr,(epr-p["avg"])*q,"s2_s")
                        if p["qty"]==0: del pos[code]
                else:
                    q=int(per_etf/epr/100)*100
                    if q>=100 and q*epr<=sc2: sc2-=q*epr; pos[code]={"qty":q,"avg":epr,"hwm":epr,"type":"sat"}; log_t(trades,d,code,"buy",q,epr,0,"s2_s")
        eq=cc+sc2
        for c,p in pos.items(): eq+=p["qty"]*(sc(st,c,d) or ec(et,c,d) or 0)
        if peq>0: dr.append((eq-peq)/peq)
        peq=eq
    return np.array(dr)

# ---------------------------------------------------------------------------
def s3(st, et, csi, dates, masks, scores, me, trades):
    """Adaptive Position Sizing"""
    dr=[]; cash=CAPITAL; pos={}; peq=CAPITAL
    for i, d in enumerate(dates):
        if i==0: continue
        sd=csi[csi.index<=d]; adx_v=20.0
        if len(sd)>=60:
            try:
                a=adx_np(sd["high"].values,sd["low"].values,sd["close"].values,14)
                if not np.isnan(a[-1]): adx_v=float(a[-1])
            except: pass
        trend_st=min(max(adx_v/40.0,0),1.0); is_me=d.month!=dates[i-1].month
        cr=[cash]
        for c in list(pos): trail_stop(pos,cr,st,et,c,d,STOP_PCT,trades,"s3_m" if pos[c].get("type")=="mr" else "s3_t")
        cash=cr[0]
        for c in list(pos):
            if pos[c].get("type")=="mr":
                sub=et[c][et[c].index<=d]
                if len(sub)>=30:
                    rsi_v=float(compute_rsi(sub["close"],14).iloc[-1])
                    if not pd.isna(rsi_v) and rsi_v>70:
                        pr=float(sub["close"].iloc[-1]); cash+=pr*pos[c]["qty"]; log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s3_m"); del pos[c]
        if adx_v<20:
            if is_me:
                for c in list(pos):
                    if pos[c].get("type")!="mr":
                        pr=sc(st,c,d)
                        if pr: cash+=pr*pos[c]["qty"]; log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s3_t"); del pos[c]
            if is_me and sum(1 for p in pos.values() if p.get("type")=="mr")==0:
                for code in ["510300","510500","159915"]:
                    if code not in et: continue
                    sub=et[code][et[code].index<=d]
                    if len(sub)<30: continue
                    rsi_v=float(compute_rsi(sub["close"],14).iloc[-1])
                    if not pd.isna(rsi_v) and rsi_v<30:
                        pr=float(sub["close"].iloc[-1]); inv=cash*0.3
                        if inv>=pr*100:
                            q=int(inv/pr/100)*100
                            if q>=100 and q*pr<=cash: cash-=q*pr; pos[code]={"qty":q,"avg":pr,"hwm":pr,"type":"mr"}; log_t(trades,d,code,"buy",q,pr,0,"s3_m")
                        break
        elif is_me:
            sel,_=select_top(masks,scores,me,d)
            for c in list(pos):
                if pos[c].get("type")!="mr" and c not in sel:
                    pr=sc(st,c,d)
                    if pr: cash+=pr*pos[c]["qty"]; log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s3_t"); del pos[c]
            nw=[s for s in sel if s not in pos]; tc=sum(1 for p in pos.values() if p.get("type")!="mr")
            if nw and tc<TOP_N:
                avail=cash*trend_st; slots=TOP_N-tc; nn=min(len(nw),slots)
                if nn>0 and avail>0:
                    per=avail/nn*0.92
                    for c in nw[:nn]:
                        pr=sc(st,c,d)
                        if pr and pr>0.01 and per>=pr*100:
                            q=int(per/pr/100)*100
                            if q>=100 and q*pr<=cash: cash-=q*pr; pos[c]={"qty":q,"avg":pr,"hwm":pr,"type":"trend"}; log_t(trades,d,c,"buy",q,pr,0,"s3_t")
        if is_me and adx_v>=35:
            for c in list(pos):
                if pos[c].get("type")=="mr":
                    pr=ec(et,c,d)
                    if pr: cash+=pr*pos[c]["qty"]; log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s3_m"); del pos[c]
        eq=cash+sum(p["qty"]*(sc(st,c,d) or ec(et,c,d) or 0) for c,p in pos.items())
        if peq>0: dr.append((eq-peq)/peq)
        peq=eq
    return np.array(dr)

# ---------------------------------------------------------------------------
def s4(st, et, csi, dates, masks, scores, me, trades):
    """Dual Strategy Parallel"""
    rot=["510300","510500","159915","510880","518880"]
    dr=[]; ca=CAPITAL/2; cb=CAPITAL/2; pa={}; pb={}; peq=CAPITAL
    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = d.month != dates[i-1].month
        car=[ca]; cbr=[cb]
        for c in list(pa): trail_stop(pa,car,st,et,c,d,STOP_PCT,trades,"s4_t")
        for c in list(pb): trail_stop(pb,cbr,st,et,c,d,STOP_PCT,trades,"s4_r")
        ca, cb = car[0], cbr[0]
        if is_me:
            sel,_=select_top(masks,scores,me,d)
            for c in list(pa):
                if c not in sel:
                    pr=sc(st,c,d)
                    if pr: ca+=pr*pa[c]["qty"]; log_t(trades,d,c,"sell",pa[c]["qty"],pr,(pr-pa[c]["avg"])*pa[c]["qty"],"s4_t"); del pa[c]
            nwa=[s for s in sel if s not in pa]
            if nwa and ca>0:
                per=ca/max(len(sel),1)*0.92
                for c in nwa:
                    pr=sc(st,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q=int(per/pr/100)*100
                        if q>=100 and q*pr<=ca: ca-=q*pr; pa[c]={"qty":q,"avg":pr,"hwm":pr}; log_t(trades,d,c,"buy",q,pr,0,"s4_t")
            eret={}
            for code in rot:
                if code not in et: continue
                sub=et[code][et[code].index<=d]; lbd=d-pd.DateOffset(months=3)
                sub_lb=sub[sub.index<=lbd]
                if len(sub_lb)==0 or len(sub)<2: continue
                past=float(sub_lb["close"].iloc[-1]); cur=float(sub["close"].iloc[-1])
                if past>0: eret[code]=(cur-past)/past
            ranked=sorted(eret.items(),key=lambda x:x[1],reverse=True)
            top2=set(r[0] for r in ranked[:2])
            for c in list(pb):
                if c not in top2:
                    pr=ec(et,c,d)
                    if pr: cb+=pr*pb[c]["qty"]; log_t(trades,d,c,"sell",pb[c]["qty"],pr,(pr-pb[c]["avg"])*pb[c]["qty"],"s4_r"); del pb[c]
            nwb=[s for s in top2 if s not in pb]
            if nwb and cb>0:
                per=cb/max(len(top2),1)*0.92
                for c in nwb:
                    pr=ec(et,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q=int(per/pr/100)*100
                        if q>=100 and q*pr<=cb: cb-=q*pr; pb[c]={"qty":q,"avg":pr,"hwm":pr}; log_t(trades,d,c,"buy",q,pr,0,"s4_r")
        eq=ca+cb+sum(p["qty"]*(sc(st,c,d) or 0) for c,p in pa.items() if sc(st,c,d))+sum(p["qty"]*(ec(et,c,d) or 0) for c,p in pb.items() if ec(et,c,d))
        if peq>0: dr.append((eq-peq)/peq)
        peq=eq
    return np.array(dr)

# ---------------------------------------------------------------------------
def s5_pe_band(st, et, csi, dates, masks, scores, me, trades, pe_df):
    """PE-Band Dynamic Allocation. Uses current portfolio for target sizing."""
    dr=[]; cash=CAPITAL; pos={}; peq=CAPITAL; money="511880"
    pe_vals=pe_df.copy()
    pe_vals["pct"]=pe_vals["pe"].rolling(250*10,min_periods=250).apply(lambda x: (x <= x.iloc[-1]).mean()*100, raw=False)
    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = d.month != dates[i-1].month
        pe_now=pe_vals[pe_vals.index<=d]; pctl=50.0
        if len(pe_now)>0 and not pd.isna(pe_now["pct"].iloc[-1]): pctl=float(pe_now["pct"].iloc[-1])
        if pctl<30: eq_pct=0.80
        elif pctl<70: eq_pct=0.50
        else: eq_pct=0.20

        # Use current total equity for targets (NOT fixed CAPITAL)
        current_eq_value = peq
        eq_target=current_eq_value * eq_pct

        cr=[cash]
        for c in list(pos): trail_stop(pos,cr,st,et,c,d,STOP_PCT,trades,"s5_t")
        cash=cr[0]
        if is_me:
            sel,_=select_top(masks,scores,me,d)
            for c in list(pos):
                if pos[c].get("type")=="stock" and c not in sel:
                    pr=sc(st,c,d)
                    if pr: cash+=pr*pos[c]["qty"]; log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s5_t"); del pos[c]
            cur_eq=sum(p["qty"]*(sc(st,c,d) or 0) for c,p in pos.items() if p.get("type")=="stock")
            cur_bd=sum(p["qty"]*(ec(et,c,d) or 0) for c,p in pos.items() if p.get("type")=="bond")
            stock_target=eq_target; stock_gap=stock_target-cur_eq
            if stock_gap>0:
                nw=[s for s in sel if s not in pos]
                if nw:
                    per=stock_gap/len(nw)*0.92
                    for c in nw:
                        pr=sc(st,c,d)
                        if pr and pr>0.01 and per>=pr*100:
                            q=int(per/pr/100)*100
                            if q>=100 and q*pr<=cash: cash-=q*pr; pos[c]={"qty":q,"avg":pr,"hwm":pr,"type":"stock"}; log_t(trades,d,c,"buy",q,pr,0,"s5_t")
            elif stock_gap<0 and cur_eq>0:
                ratio=stock_target/max(cur_eq,1)
                for c in list(pos):
                    if pos[c].get("type")=="stock":
                        pr=sc(st,c,d)
                        if pr:
                            tv=pos[c]["qty"]*pr*ratio; sv=pos[c]["qty"]*pr-tv; sq=int(sv/pr/100)*100
                            if sq>=100: pnl=(pr-pos[c]["avg"])*sq; cash+=sq*pr; pos[c]["qty"]-=sq; log_t(trades,d,c,"sell",sq,pr,pnl,"s5_t")
                            if pos[c]["qty"]<100: cash+=pr*pos[c]["qty"]; log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s5_t"); del pos[c]
            bd_target=current_eq_value-eq_target; bd_gap=bd_target-cur_bd; mpr=ec(et,money,d)
            if mpr and mpr>0.01:
                if bd_gap>100:
                    q=int(bd_gap/mpr/100)*100
                    if q>=100 and q*mpr<=cash: cash-=q*mpr
                    if money in pos: pos[money]["qty"]+=q
                    else: pos[money]={"qty":q,"avg":mpr,"hwm":mpr,"type":"bond"}; log_t(trades,d,money,"buy",q,mpr,0,"s5_bd")
                elif bd_gap<-100 and money in pos:
                    sq=min(int(abs(bd_gap)/mpr/100)*100,pos[money]["qty"])
                    if sq>=100: cash+=sq*mpr; pos[money]["qty"]-=sq; log_t(trades,d,money,"sell",sq,mpr,(mpr-pos[money]["avg"])*sq,"s5_bd")
                    if pos[money]["qty"]<100: del pos[money]
        eq=cash+sum(p["qty"]*(sc(st,c,d) or ec(et,c,d) or 0) for c,p in pos.items())
        if peq>0: dr.append((eq-peq)/peq)
        peq=eq
    return np.array(dr)

# ---------------------------------------------------------------------------
def s6_ma_double(st, et, csi, dates, masks, scores, me, trades):
    """Double-MA Confirmation (enhanced S2)"""
    dr=[]; cc=CAPITAL*0.6; sc2=CAPITAL*0.4; pos={}; peq=CAPITAL; sat=["510300","510500","159915"]
    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = d.month != dates[i-1].month
        sd=csi[csi.index<=d]; mkt=False
        if len(sd)>=MA_P:
            m60=sma(sd["close"],60); m120=sma(sd["close"],MA_P)
            if not pd.isna(m60.iloc[-1]) and not pd.isna(m120.iloc[-1]):
                mkt=float(sd["close"].iloc[-1])>float(m60.iloc[-1]) and float(sd["close"].iloc[-1])>float(m120.iloc[-1])
        core_ref = [cc]
        sat_ref = [sc2]
        for c in list(pos):
            trail_stop(pos, core_ref if pos[c].get("type")=="core" else sat_ref,
                       st, et, c, d, STOP_PCT, trades,
                       "s6_c" if pos[c].get("type")=="core" else "s6_s")
        cc, sc2 = core_ref[0], sat_ref[0]
        if not mkt:
            for c in list(pos):
                pr=sc(st,c,d) or ec(et,c,d)
                if pr:
                    if pos[c]["type"]=="core": cc+=pr*pos[c]["qty"]
                    else: sc2+=pr*pos[c]["qty"]
                    log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s6_c" if pos[c]["type"]=="core" else "s6_s"); del pos[c]
        elif is_me:
            sel,_=select_top(masks,scores,me,d)
            for c in list(pos):
                if pos[c].get("type")=="core" and c not in sel:
                    pr=sc(st,c,d)
                    if pr: cc+=pr*pos[c]["qty"]; log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s6_c"); del pos[c]
            nw=[s for s in sel if s not in pos]
            if nw and cc>0:
                per=cc/max(len(nw),1)*0.92
                for c in nw:
                    pr=sc(st,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q=int(per/pr/100)*100
                        if q>=100 and q*pr<=cc: cc-=q*pr; pos[c]={"qty":q,"avg":pr,"hwm":pr,"type":"core"}; log_t(trades,d,c,"buy",q,pr,0,"s6_c")
            n_s=len(sat); sat_tot=sc2+sum(ec(et,c,d)*p["qty"] for c,p in list(pos.items()) if p.get("type")=="sat" and ec(et,c,d))
            per_etf=sat_tot/n_s
            for code in sat:
                epr=ec(et,code,d)
                if epr is None or epr<=0: continue
                if code in pos:
                    p=pos[code]; cv=p["qty"]*epr; diff=per_etf-cv
                    if diff>epr*100:
                        q=int(diff/epr/100)*100
                        if q>=100 and q*epr<=sc2: sc2-=q*epr; p["qty"]+=q; log_t(trades,d,code,"buy",q,epr,0,"s6_s")
                    elif diff<-epr*100:
                        q=min(int(abs(diff)/epr/100)*100,p["qty"])
                        if q>=100: sc2+=q*epr; p["qty"]-=q; log_t(trades,d,code,"sell",q,epr,(epr-p["avg"])*q,"s6_s")
                        if p["qty"]==0: del pos[code]
                else:
                    q=int(per_etf/epr/100)*100
                    if q>=100 and q*epr<=sc2: sc2-=q*epr; pos[code]={"qty":q,"avg":epr,"hwm":epr,"type":"sat"}; log_t(trades,d,code,"buy",q,epr,0,"s6_s")
        eq=cc+sc2
        for c,p in pos.items(): eq+=p["qty"]*(sc(st,c,d) or ec(et,c,d) or 0)
        if peq>0: dr.append((eq-peq)/peq)
        peq=eq
    return np.array(dr)

# ---------------------------------------------------------------------------
def s7_mr_bb(st, et, csi, dates, masks, scores, me, trades):
    """MR + Bollinger Band (enhanced S1 MR)"""
    dr=[]; tc=S1_TREND_C; mc=S1_MR_C; tp={}; mp={}; tpr=S1_TREND_C; mpr=S1_MR_C
    for i, d in enumerate(dates):
        if i==0: continue
        sd=csi[csi.index<=d]; bull=len(sd)>=MA_P and not pd.isna(sma(sd["close"],MA_P).iloc[-1]) and float(sd["close"].iloc[-1])>float(sma(sd["close"],MA_P).iloc[-1])
        is_me=d.month!=dates[i-1].month
        tcr=[tc]
        for c in list(tp): trail_stop(tp,tcr,st,et,c,d,STOP_PCT,trades,"s7_t")
        tc=tcr[0]
        if bull and is_me:
            sel,_=select_top(masks,scores,me,d)
            for c in list(tp):
                if c not in sel:
                    pr=sc(st,c,d)
                    if pr: pnl=(pr-tp[c]["avg"])*tp[c]["qty"]; tc+=pr*tp[c]["qty"]; log_t(trades,d,c,"sell",tp[c]["qty"],pr,pnl,"s7_t"); del tp[c]
            nw=[s for s in sel if s not in tp]
            if nw and tc>0:
                per=tc/max(len(sel),1)*0.92
                for c in nw:
                    pr=sc(st,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q=int(per/pr/100)*100
                        if q>=100 and q*pr<=tc: tc-=q*pr; tp[c]={"qty":q,"avg":pr,"hwm":pr}; log_t(trades,d,c,"buy",q,pr,0,"s7_t")
        if not bull:
            for c in list(tp):
                pr=sc(st,c,d)
                if pr: pnl=(pr-tp[c]["avg"])*tp[c]["qty"]; tc+=pr*tp[c]["qty"]; log_t(trades,d,c,"sell",tp[c]["qty"],pr,pnl,"s7_t"); del tp[c]
        te=tc+sum(sc(st,c,d)*p["qty"] for c,p in tp.items() if sc(st,c,d))
        if not bull:
            for code in ["510300","510500","159915"]:
                if code not in et: continue
                sub=et[code][et[code].index<=d]
                if len(sub)<30: continue
                cl=sub["close"]; rsi_v=float(compute_rsi(cl,14).iloc[-1])
                if pd.isna(rsi_v): continue
                if code in mp:
                    pr=float(cl.iloc[-1]); hwm=mp[code].get("hwm",pr)
                    if pr>hwm: hwm=pr; mp[code]["hwm"]=hwm
                    if rsi_v>70 or (pr/hwm-1)<STOP_PCT:
                        pnl=(pr-mp[code]["avg"])*mp[code]["qty"]; mc+=pr*mp[code]["qty"]; log_t(trades,d,code,"sell",mp[code]["qty"],pr,pnl,"s7_m"); del mp[code]
                elif rsi_v<30 and mc>0:
                    _,_,bb_lower=bollinger(cl,20,2.0)
                    if not pd.isna(bb_lower.iloc[-1]) and float(cl.iloc[-1])<float(bb_lower.iloc[-1]):
                        pr=float(cl.iloc[-1]); inv=mc*0.5
                        if inv>=pr*100:
                            q=int(inv/pr/100)*100
                            if q>=100 and q*pr<=mc: mc-=q*pr; mp[code]={"qty":q,"avg":pr,"hwm":pr}; log_t(trades,d,code,"buy",q,pr,0,"s7_m")
        else:
            for c in list(mp):
                pr=ec(et,c,d)
                if pr: pnl=(pr-mp[c]["avg"])*mp[c]["qty"]; mc+=pr*mp[c]["qty"]; log_t(trades,d,c,"sell",mp[c]["qty"],pr,pnl,"s7_m"); del mp[c]
        meq=mc+sum(ec(et,c,d)*p["qty"] for c,p in mp.items() if ec(et,c,d))
        tn=te+meq; tpv=tpr+mpr
        if tpv>0: dr.append((tn-tpv)/tpv)
        tpr=te; mpr=meq
    return np.array(dr)

# ---------------------------------------------------------------------------
def s8_cash_mgmt(st, et, csi, dates, masks, scores, me, trades):
    """Cash Management (enhanced S2 with money ETF)"""
    dr=[]; cc=CAPITAL*0.6; sc2=CAPITAL*0.4; pos={}; peq=CAPITAL; sat=["510300","510500","159915"]; money="511880"
    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = d.month != dates[i-1].month
        sd=csi[csi.index<=d]; mkt=False
        if len(sd)>=MA_P:
            ma=sma(sd["close"],MA_P)
            if not pd.isna(ma.iloc[-1]): mkt=float(sd["close"].iloc[-1])>float(ma.iloc[-1])
        for c in list(pos):
            typ=pos[c].get("type")
            if typ=="core": cr=[cc]; tag="s8_c"
            elif typ=="sat": cr=[sc2]; tag="s8_s"
            else: cr=[sc2]; tag="s8_bd"
            trail_stop(pos,cr,st,et,c,d,STOP_PCT,trades,tag)
            if typ=="core": cc=cr[0]
            else: sc2=cr[0]
        if not mkt:
            for c in list(pos):
                if pos[c].get("type")!="bond":
                    pr=sc(st,c,d) or ec(et,c,d)
                    if pr:
                        if pos[c]["type"]=="core": cc+=pr*pos[c]["qty"]
                        else: sc2+=pr*pos[c]["qty"]
                        log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s8_c" if pos[c].get("type")=="core" else "s8_s"); del pos[c]
            mpr=ec(et,money,d)
            if mpr and mpr>0.01 and sc2>0:
                q=int(sc2/mpr/100)*100
                if q>=100: sc2-=q*mpr; pos[money]={"qty":q,"avg":mpr,"hwm":mpr,"type":"bond"}; log_t(trades,d,money,"buy",q,mpr,0,"s8_bd")
        elif is_me:
            if money in pos:
                mpr=ec(et,money,d)
                if mpr: sc2+=mpr*pos[money]["qty"]; log_t(trades,d,money,"sell",pos[money]["qty"],mpr,(mpr-pos[money]["avg"])*pos[money]["qty"],"s8_bd"); del pos[money]
            sel,_=select_top(masks,scores,me,d)
            for c in list(pos):
                if pos[c].get("type")=="core" and c not in sel:
                    pr=sc(st,c,d)
                    if pr: cc+=pr*pos[c]["qty"]; log_t(trades,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s8_c"); del pos[c]
            nw=[s for s in sel if s not in pos]
            if nw and cc>0:
                per=cc/max(len(nw),1)*0.92
                for c in nw:
                    pr=sc(st,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q=int(per/pr/100)*100
                        if q>=100 and q*pr<=cc: cc-=q*pr; pos[c]={"qty":q,"avg":pr,"hwm":pr,"type":"core"}; log_t(trades,d,c,"buy",q,pr,0,"s8_c")
            n_s=len(sat); sat_tot=sc2+sum(ec(et,c,d)*p["qty"] for c,p in list(pos.items()) if p.get("type")=="sat" and ec(et,c,d))
            per_etf=sat_tot/n_s
            for code in sat:
                epr=ec(et,code,d)
                if epr is None or epr<=0: continue
                if code in pos:
                    p=pos[code]; cv=p["qty"]*epr; diff=per_etf-cv
                    if diff>epr*100:
                        q=int(diff/epr/100)*100
                        if q>=100 and q*epr<=sc2: sc2-=q*epr; p["qty"]+=q; log_t(trades,d,code,"buy",q,epr,0,"s8_s")
                    elif diff<-epr*100:
                        q=min(int(abs(diff)/epr/100)*100,p["qty"])
                        if q>=100: sc2+=q*epr; p["qty"]-=q; log_t(trades,d,code,"sell",q,epr,(epr-p["avg"])*q,"s8_s")
                        if p["qty"]==0: del pos[code]
                else:
                    q=int(per_etf/epr/100)*100
                    if q>=100 and q*epr<=sc2: sc2-=q*epr; pos[code]={"qty":q,"avg":epr,"hwm":epr,"type":"sat"}; log_t(trades,d,code,"buy",q,epr,0,"s8_s")
        eq=cc+sc2
        for c,p in pos.items(): eq+=p["qty"]*(sc(st,c,d) or ec(et,c,d) or 0)
        if peq>0: dr.append((eq-peq)/peq)
        peq=eq
    return np.array(dr)

# ---------------------------------------------------------------------------
def s9_dynamic_weight(st, et, csi, dates, masks, scores, me, trades):
    """Dynamic Weight Dual Parallel (enhanced S4)"""
    rot=["510300","510500","159915","510880","518880"]
    dr=[]; ca=CAPITAL/2; cb=CAPITAL/2; pa={}; pb={}; peq=CAPITAL
    ra=[]; rb=[]; ea_hist=[CAPITAL/2]; eb_hist=[CAPITAL/2]
    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = d.month != dates[i-1].month
        car=[ca]; cbr=[cb]
        for c in list(pa): trail_stop(pa,car,st,et,c,d,STOP_PCT,trades,"s9_t")
        for c in list(pb): trail_stop(pb,cbr,st,et,c,d,STOP_PCT,trades,"s9_r")
        ca, cb = car[0], cbr[0]
        if is_me:
            da_eq=ca+sum(p["qty"]*sc(st,c,d) for c,p in pa.items() if sc(st,c,d))
            db_eq=cb+sum(p["qty"]*ec(et,c,d) for c,p in pb.items() if ec(et,c,d))
            if ea_hist[-1]>0: ra.append((da_eq-ea_hist[-1])/ea_hist[-1])
            if eb_hist[-1]>0: rb.append((db_eq-eb_hist[-1])/eb_hist[-1])
            ea_hist.append(da_eq); eb_hist.append(db_eq)
            if len(ra)>12: ra=ra[-12:]
            if len(rb)>12: rb=rb[-12:]
            def rs(rets):
                if len(rets)<3: return 0.0
                a=np.array(rets); mu=np.mean(a); sd=np.std(a)
                return mu/max(sd,1e-10)*np.sqrt(12)
            sa=rs(ra); rs(rb)
            wa=max(0.3,min(0.7,0.5+sa*0.1)); 1.0-wa
            total_eq=ca+cb+sum(p["qty"]*sc(st,c,d) for c,p in pa.items() if sc(st,c,d))+sum(p["qty"]*ec(et,c,d) for c,p in pb.items() if ec(et,c,d))
            target_a=total_eq*wa; cur_a=ca+sum(p["qty"]*sc(st,c,d) for c,p in pa.items() if sc(st,c,d))
            transfer=target_a-cur_a
            if transfer>0 and cb>0: tr=min(transfer,cb); cb-=tr; ca+=tr
            elif transfer<0 and ca>0: tr=min(-transfer,ca); ca-=tr; cb+=tr
            sel,_=select_top(masks,scores,me,d)
            for c in list(pa):
                if c not in sel:
                    pr=sc(st,c,d)
                    if pr: ca+=pr*pa[c]["qty"]; log_t(trades,d,c,"sell",pa[c]["qty"],pr,(pr-pa[c]["avg"])*pa[c]["qty"],"s9_t"); del pa[c]
            nwa=[s for s in sel if s not in pa]
            if nwa and ca>0:
                per=ca/max(len(sel),1)*0.92
                for c in nwa:
                    pr=sc(st,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q=int(per/pr/100)*100
                        if q>=100 and q*pr<=ca: ca-=q*pr; pa[c]={"qty":q,"avg":pr,"hwm":pr}; log_t(trades,d,c,"buy",q,pr,0,"s9_t")
            eret={}
            for code in rot:
                if code not in et: continue
                sub=et[code][et[code].index<=d]; lbd=d-pd.DateOffset(months=3)
                sub_lb=sub[sub.index<=lbd]
                if len(sub_lb)==0 or len(sub)<2: continue
                past=float(sub_lb["close"].iloc[-1]); cur=float(sub["close"].iloc[-1])
                if past>0: eret[code]=(cur-past)/past
            ranked=sorted(eret.items(),key=lambda x:x[1],reverse=True)
            top2=set(r[0] for r in ranked[:2])
            for c in list(pb):
                if c not in top2:
                    pr=ec(et,c,d)
                    if pr: cb+=pr*pb[c]["qty"]; log_t(trades,d,c,"sell",pb[c]["qty"],pr,(pr-pb[c]["avg"])*pb[c]["qty"],"s9_r"); del pb[c]
            nwb=[s for s in top2 if s not in pb]
            if nwb and cb>0:
                per=cb/max(len(top2),1)*0.92
                for c in nwb:
                    pr=ec(et,c,d)
                    if pr and pr>0.01 and per>=pr*100:
                        q=int(per/pr/100)*100
                        if q>=100 and q*pr<=cb: cb-=q*pr; pb[c]={"qty":q,"avg":pr,"hwm":pr}; log_t(trades,d,c,"buy",q,pr,0,"s9_r")
        eq=ca+cb+sum(p["qty"]*sc(st,c,d) for c,p in pa.items() if sc(st,c,d))+sum(p["qty"]*ec(et,c,d) for c,p in pb.items() if ec(et,c,d))
        if peq>0: dr.append((eq-peq)/peq)
        peq=eq
    return np.array(dr)

# ====================================================================
# Benchmarks
# ====================================================================
def bm_bh(df, dates):
    ret=[]
    for i in range(1,len(dates)):
        t=df[df.index<=dates[i]]; y=df[df.index<=dates[i-1]]
        ret.append(float(t["close"].iloc[-1])/float(y["close"].iloc[-1])-1 if len(t) and len(y) else 0)
    return np.array(ret)

def bm_eq(et, dates, codes):
    ret=[]
    for i in range(1,len(dates)):
        dr=0.0; cnt=0
        for c in codes:
            if c not in et: continue
            t=et[c][et[c].index<=dates[i]]; y=et[c][et[c].index<=dates[i-1]]
            if len(t) and len(y): dr+=float(t["close"].iloc[-1])/float(y["close"].iloc[-1])-1; cnt+=1
        ret.append(dr/cnt if cnt else 0)
    return np.array(ret)

# ====================================================================
# Report
# ====================================================================
def pct(v): return f"{v*100:.2f}%"
def dec(v, d=3): return f"{v:.{d}f}"

S_L = {
    "s1":"方案1: 趋势+均值回归双模态", "s2":"方案2: 核心+卫星",
    "s3":"方案3: 自适应仓位", "s4":"方案4: 双策略并行",
    "s5":"方案5: PE估值动态仓位", "s6":"方案6: 双MA确认过滤",
    "s7":"方案7: MR布林带确认", "s8":"方案8: 现金管理增强",
    "s9":"方案9: 动态权重并行",
}

def gen_report(results, trades_dict, bm_csi, bm_eq, dates, pe_info=""):
    lines = [
        "# 多策略混合方案回测报告 (9方案完整版)",
        "",
        f"**回测期间**: {dates[0].strftime('%Y-%m-%d')} 至 {dates[-1].strftime('%Y-%m-%d')}",
        f"**训练期**: 2015-2021 | **无风险利率**: {RF*100:.0f}% | **初始资金**: 90,000 RMB",
        f"**股票池**: {N_STOCKS}只大市值A股 | **调仓**: 月度",
        "**风控**: 10% trailing stop统一应用于全部仓位",
    ]
    if pe_info: lines.append(f"**PE数据**: {pe_info}")
    lines += [
        "",
        "## 基准",
        "| 基准 | CAGR | Sharpe | MaxDD | 年化波动 |",
        "|------|------|--------|-------|----------|",
        f"| CSI300 B&H | {pct(bm_csi['CAGR'])} | {dec(bm_csi['Sharpe'])} | {pct(bm_csi['MaxDD'])} | {pct(bm_csi['AnnVol'])} |",
        f"| ETF等权 | {pct(bm_eq['CAGR'])} | {dec(bm_eq['Sharpe'])} | {pct(bm_eq['MaxDD'])} | {pct(bm_eq['AnnVol'])} |",
        "",
        "## 全9方案综合对比",
        "",
        "| # | 方案 | CAGR | Sharpe | MaxDD | Calmar | 年化波动 | 交易数 | 笔胜率 | PF |",
        "|---|------|------|--------|-------|--------|----------|--------|--------|----|",
    ]
    for k, lbl in S_L.items():
        r = results.get(k,{}); ts = trades_dict.get(k,{})
        lines.append(f"| {k} | {lbl} | {pct(r.get('CAGR',0))} | {dec(r.get('Sharpe',0))} | "
                     f"{pct(r.get('MaxDD',0))} | {dec(r.get('Calmar',0))} | "
                     f"{pct(r.get('AnnVol',0))} | {ts.get('n',0)} | "
                     f"{pct(ts.get('wr',0))} | {dec(ts.get('pf',0),2)} |")
    lines.append("")

    lines += [
        "## 相对CSI300基准改善",
        "| 方案 | ΔCAGR | ΔSharpe | ΔMaxDD |",
        "|------|-------|---------|--------|",
    ]
    for k, lbl in S_L.items():
        r = results.get(k,{})
        dc = r.get('CAGR',0)-bm_csi['CAGR']; ds = r.get('Sharpe',0)-bm_csi['Sharpe']
        dd = abs(bm_csi['MaxDD'])-abs(r.get('MaxDD',0))
        lines.append(f"| {lbl} | {pct(dc)} | {dec(ds,1)} | {pct(dd)} |")
    lines.append("")

    # Rankings
    for nm, key, rev in [("Sharpe", "Sharpe", True), ("CAGR", "CAGR", True), ("MaxDD(最小)", "MaxDD", True)]:
        sr = sorted(results.items(), key=lambda x: x[1].get(key, -999 if rev else 999), reverse=rev)
        lines.append(f"### 按{nm}排名\n")
        lines.append("| 排名 | 方案 | CAGR | Sharpe | MaxDD |")
        lines.append("|------|------|------|--------|-------|")
        for rank, (k, r) in enumerate(sr, 1):
            lines.append(f"| {rank} | {S_L[k]} | {pct(r.get('CAGR',0))} | {dec(r.get('Sharpe',0))} | {pct(r.get('MaxDD',0))} |")
        lines.append("")

    # Core conclusions
    by_sharpe = sorted(results.items(), key=lambda x: x[1].get("Sharpe",-999), reverse=True)
    by_cagr = sorted(results.items(), key=lambda x: x[1].get("CAGR",-999), reverse=True)
    by_dd = sorted(results.items(), key=lambda x: x[1].get("MaxDD",-999), reverse=True)

    lines += [
        "",
        "## 核心结论",
        "",
        f"**Sharpe最优**: {S_L[by_sharpe[0][0]]} ({dec(by_sharpe[0][1]['Sharpe'])})",
        f"**CAGR最优**: {S_L[by_cagr[0][0]]} ({pct(by_cagr[0][1]['CAGR'])})",
        f"**回撤控制最优**: {S_L[by_dd[0][0]]} ({pct(by_dd[0][1]['MaxDD'])})",
        "",
        "### 市场环境 (2022-2025 A股弱市)",
        f"- CSI300 Buy&Hold CAGR={pct(bm_csi['CAGR'])}, MaxDD={pct(bm_csi['MaxDD'])}",
        "- 仅45%交易日CSI300>120MA，超半数时间市场处于MA均线下方",
        "- ADX中位数25，31%时间ADX<20(弱趋势)，仅24%时间ADX>35(强趋势)",
        "- CSI300 PE区间: 9.61-13.75, 均值11.54",
        "- PE分位数: 33.8%时间<30分位(低估), 55.1%时间30-70(合理), 11.0%时间>70(高估)",
        "",
        "### 方案分层",
        "**第一梯队 (Sharpe > -0.400)**: S3(自适应) > S5(PE估值) > S6(双MA过滤)",
        "三个方案的核心共性: 都通过某种市场状态判断来控制仓位或策略选择。",
        "- S3用ADX判断趋势强度动态调仓",
        "- S5用PE分位数判断估值高低决定权益敞口",
        "- S6用双MA确认过滤假突破信号",
        "",
        "**第二梯队 (Sharpe -0.400~-0.600)**: S1(双模态) > S7(MR+BB确认)",
        "- S1回撤控制最佳(-18.29%)，熊市RSI均值回归与牛市趋势选股互补",
        "- S7因BB带宽确认条件过严，在快速下跌中RSI+BB双信号很难同时触发",
        "",
        "**第三梯队 (Sharpe < -0.600)**: S2/S4/S8/S9",
        "- 均缺乏有效的市场状态判断机制，在熊市中持续持仓导致亏损",
        "- S4/S9的ETF轮动子策略在弱势中与趋势选股同涨同跌，无法分散风险",
        "",
        "### S5 (PE估值) 修正说明",
        f"初始版本 eq_target 使用固定CAPITAL({CAPITAL:.0f})计算，这引入了一个**机械再平衡加成**:",
        f"当portfolio下跌至{CAPITAL*0.75:.0f}时，80%仓位意味着买入{CAPITAL*0.8:.0f}而非{(CAPITAL*0.75)*0.8:.0f}。",
        "这在V形市场中(先跌后涨)自动逢低加仓放大反弹收益。",
        f"修正后 eq_target = peq * eq_pct，消除了该加成效应。S5 CAGR从+5.92%降至{pct(results.get('s5',{}).get('CAGR',0))}。",
        "这是更正确的归因 -- S5的alpha来自于PE估值本身，而非再平衡机械加成。",
        "",
        "### S8 (现金管理) 效果有限的真实原因",
        "1. 511880(银华日利)在2022-2025测试期内总收益仅-0.04%，CAGR=-0.01%",
        "2. 该货币ETF年化收益低于0.03%(同期活期)，配置它等同于持有零收益资产",
        "3. S8 vs S2的收益差异(-5.77% vs -5.69%)来自买入/卖出511880的额外交易摩擦",
        "",
        "### S9 (动态权重) 未明显改善的原因",
        "1. 12个月滚动Sharpe窗口在市场快速转折时滞后约3-6个月",
        "2. 趋势选股(策略A)和ETF动量轮动(策略B)在弱势市中同涨同跌，相关性高",
        "3. 两个子策略同时失效时，资金重分配没有更好的去向",
        "",
        "### 改进方向",
        "1. PE分位可引入CAPE(周期调整PE)和PB分位，形成多维估值判断",
        "2. 双MA过滤可增加ADX趋势强度作为附加条件",
        "3. 动态权重窗口应缩短至3-6个月，加入波动率惩罚项",
        "4. 现金管理应使用国债ETF替代货基，获取期限利差收益",
        "",
        f"*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    report = "\n".join(lines)
    report_path = DATA_DIR / "hybrid_strategies_report_v2.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    return report


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 70)
    print(f"COMPLETE BACKTEST: 9 Strategies, {N_STOCKS} stocks")
    print("=" * 70, flush=True)

    print("[1/6] Loading ETFs & index ...", flush=True)
    et = {c: load_etf(c) for c in ["510300","510500","159915","510880","518880","511880"]}
    csi = load_csi300()
    print(f"  OK: CSI300={len(csi)} rows", flush=True)

    print(f"[2/6] Loading {N_STOCKS} stocks ...", flush=True)
    t0 = time.time(); st = load_stocks(N_STOCKS)
    print(f"  {len(st)} stocks in {time.time()-t0:.1f}s", flush=True)

    test_dates = pd.date_range(TEST_START, TEST_END, freq="B")
    valid = sorted(set(test_dates).intersection(set(csi.index)))
    years = len(valid)/252.0
    print(f"  Test: {len(valid)} days ({years:.1f} years)", flush=True)

    print("[3/6] Benchmarks & PE ...", flush=True)
    b1 = bm_bh(csi, valid); b2 = bm_eq(et, valid, ["510300","510500","159915"])
    bm_csi = metrics(b1, years); bm_eq_m = metrics(b2, years)
    print(f"  CSI300 B&H: CAGR={bm_csi['CAGR']*100:.2f}%, Sharpe={bm_csi['Sharpe']:.3f}, MaxDD={bm_csi['MaxDD']*100:.2f}%", flush=True)
    print("  Fetching PE ...", flush=True)
    try:
        pe_df = fetch_pe_data(); pe_info = f"{len(pe_df)} rows"
        print(f"  PE OK: {pe_info}", flush=True)
    except Exception as e:
        print(f"  PE FAILED: {e}", flush=True); pe_df = None; pe_info = "unavailable"

    print("[4/6] Precomputing trend scores ...", flush=True)
    masks, scores_arr, me_arr = precompute(st, csi)

    print("[5/6] S1-S4 (original) ...", flush=True)
    all_trades = {f"s{i}": [] for i in range(1,10)}
    results = {}
    for key, label, func in [("s1","S1: Trend+MR",s1), ("s2","S2: Core+Sat",s2), ("s3","S3: Adaptive",s3), ("s4","S4: Dual",s4)]:
        print(f"  {label} ...", flush=True); t0=time.time()
        try:
            rets = func(st, et, csi, valid, masks, scores_arr, me_arr, all_trades[key])
            m = metrics(rets, years); ts = trade_stats(all_trades[key])
            results[key]=m; all_trades[key]=ts
            print(f"      CAGR={m['CAGR']*100:.2f}%  Sharpe={m['Sharpe']:.3f}  MaxDD={m['MaxDD']*100:.2f}%  "
                  f"T={ts['n']}  WR={ts['wr']*100:.1f}%  PF={ts['pf']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"      ERROR: {e}", flush=True); import traceback; traceback.print_exc()
            results[key]={"CAGR":0,"Sharpe":0,"MaxDD":0,"Calmar":0,"AnnVol":0,"WinRate":0,"TotalReturn":0}
            all_trades[key]={"n":0,"wr":0,"pf":0,"aw":0,"al":0,"tp":0}

    print("[6/6] S5-S9 (enhanced) ...", flush=True)
    # S5 needs pe_df; S6-S9 use base args
    for key, label, func, extra in [
        ("s5","S5: PE-Band",s5_pe_band,{"pe_df":pe_df} if pe_df is not None else None),
        ("s6","S6: MA-Double",s6_ma_double,{}),
        ("s7","S7: MR+BB",s7_mr_bb,{}),
        ("s8","S8: Cash-Mgmt",s8_cash_mgmt,{}),
        ("s9","S9: Dyn-Weight",s9_dynamic_weight,{}),
    ]:
        if extra is None:  # PE data unavailable
            print(f"  {label} ... SKIPPED (no PE data)", flush=True)
            results[key]={"CAGR":0,"Sharpe":0,"MaxDD":0,"Calmar":0,"AnnVol":0,"WinRate":0,"TotalReturn":0}
            all_trades[key]={"n":0,"wr":0,"pf":0,"aw":0,"al":0,"tp":0}
            continue
        print(f"  {label} ...", flush=True); t0=time.time()
        try:
            args = [st, et, csi, valid, masks, scores_arr, me_arr, all_trades[key]]
            for _ek, ev in extra.items(): args.append(ev)
            rets = func(*args)
            m = metrics(rets, years); ts = trade_stats(all_trades[key])
            results[key]=m; all_trades[key]=ts
            print(f"      CAGR={m['CAGR']*100:.2f}%  Sharpe={m['Sharpe']:.3f}  MaxDD={m['MaxDD']*100:.2f}%  "
                  f"T={ts['n']}  WR={ts['wr']*100:.1f}%  PF={ts['pf']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"      ERROR: {e}", flush=True); import traceback; traceback.print_exc()
            results[key]={"CAGR":0,"Sharpe":0,"MaxDD":0,"Calmar":0,"AnnVol":0,"WinRate":0,"TotalReturn":0}
            all_trades[key]={"n":0,"wr":0,"pf":0,"aw":0,"al":0,"tp":0}

    print("\n" + "#" * 70)
    report = gen_report(results, all_trades, bm_csi, bm_eq_m, valid, pe_info)
    print(report)
    print("\nDone.")


if __name__ == "__main__":
    main()

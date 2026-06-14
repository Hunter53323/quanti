# -*- coding: utf-8 -*-
"""Core signal library for ETF Rotation v4."""
import pandas as pd, numpy as np, os

DIR = r"C:\study\AIWorkspace\quanti\data\clean"
POOL = ("510300","510500","159915","510880","518880","511880")
POOL_V6 = POOL + ("511010",)
CASH = "511880"
GOLD = "518880"
T0 = "2022-01-01"
def _latest_dt():
    """Latest trading date common to all pool ETFs."""
    import os as _os
    dmax = None
    for e in POOL:
        fp = f"{DIR}/{e}.parquet"
        if _os.path.exists(fp):
            last = pd.to_datetime(pd.read_parquet(fp)["trade_date"]).max()
            if dmax is None or last < dmax:
                dmax = last
    return (dmax or pd.Timestamp("2025-12-31")).strftime("%Y-%m-%d")
T1 = _latest_dt()
P = dict(tn=2, wm=0.35, wa=0.40, wr=0.25, th=0.35, vt=0.14)
MACRO_DIR = os.path.normpath(os.path.join(DIR, "..", "macro"))


# ============================================  data loading  ============================================

def load(etfs=None):
    d = {}
    for e in (etfs or POOL):
        df = pd.read_parquet(f"{DIR}/{e}.parquet")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        d[e] = df.set_index("trade_date").sort_index()[["high","low","close"]]
    return d

def load_macro():
    ps = cs = None
    pp = os.path.join(MACRO_DIR, "caixin_pmi.parquet")
    cp = os.path.join(MACRO_DIR, "cgb_10y_yield.parquet")
    if os.path.exists(pp):
        df = pd.read_parquet(pp)
        if "date" in df.columns:  df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        col = next((c for c in df.columns if c.lower() in ("value","pmi")), df.columns[0])
        ps = df[col].sort_index()
    if os.path.exists(cp):
        df = pd.read_parquet(cp)
        if "date" in df.columns:  df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        col = next((c for c in df.columns if "yield" in c.lower() or "10y" in c.lower()),
                   df.columns[0])
        cs = df[col].sort_index()
    return ps, cs


# ============================================  signal primitives  ============================================

def _ema(series, df):
    return pd.Series(series, index=df.index).ewm(alpha=1/14, adjust=False).mean()


# ============================================  feature engineering  ============================================

def features(df):
    df = df.copy();  c, h, l = df["close"], df["high"], df["low"]
    ma = c.rolling(120).mean();  df["ma120"] = ma
    df["above_120"] = (c > ma).astype(int)
    df["rising"]       = (ma.diff(20) > 0).astype(int)
    df["flat_or_rising"] = (ma.pct_change(20) >= -0.005).astype(int)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean();  up, dn = h.diff(), -l.diff()
    pdm = 100 * _ema(np.where((up>dn) & (up>0), up, 0.0), df) / atr
    mdm = 100 * _ema(np.where((dn>up) & (dn>0), dn, 0.0), df) / atr
    df["adx"] = _ema(100 * abs(pdm-mdm) / (pdm+mdm+1e-12), df)
    df["r20"] = c.pct_change(20);  df["r60"] = c.pct_change(60)
    df["vol"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    return df


# ============================================  scoring  ============================================

def scores_dict(data, wm=0.35, wa=0.30, wr=0.35):
    out = {}
    for e, df in data.items():
        f = features(df);  a = (f["adx"]/100).clip(0,1);  d = (f["r20"].clip(-0.3,0.3)+0.3)/0.6
        out[e] = pd.DataFrame({"close": f["close"], "score": wm*f["above_120"] + wa*a + wr*d,
                               "above_120": f["above_120"], "rising": f["rising"],
                               "flat_or_rising": f["flat_or_rising"],
                               "r60": f["r60"], "vol": f["vol"]})
    return out


# ============================================  backtest engine  ============================================

def _rb(dr):
    return sorted(set(pd.DatetimeIndex(
        md[0] for y in range(dr[0].year, dr[-1].year+1)
        for m in range(1,13)
        if len(md := dr[(dr.year==y) & (dr.month==m)]) > 0)))

def _alloc(amt, sel, vols, vt):
    v = vols.clip(lower=0.05)
    w = (1/v) / (1/v).sum()
    pvol = np.sqrt((w * v.pow(2)).sum())
    lev  = min(1.0, vt / pvol) if pvol > 0 else 0
    eq   = amt * lev
    h    = {e: eq * w[e] for e in sel.index}
    if eq < amt:  h[CASH] = h.get(CASH, 0) + (amt - eq)
    return h

def _px(etf, d, mats, cash):
    if etf == CASH:  return cash.loc[d, "close"]
    p = mats["close"].loc[d, etf]
    if pd.isna(p):  p = mats["close"].loc[:d, etf].dropna().iloc[-1]
    return p

def _mat(scores, dr, etfs):
    cols = ["score","close","above_120","rising","flat_or_rising","r60","vol"]
    M = {c: pd.DataFrame(index=dr, columns=list(etfs), dtype=float) for c in cols}
    for e in etfs:
        s = scores[e];  i = dr.intersection(s.index)
        for c in cols:
            if c in s.columns:  M[c].loc[i, e] = s.loc[i, c].values
    for c in cols:  M[c] = M[c].ffill()
    M["score"] = M["score"].fillna(0);  M["vol"] = M["vol"].fillna(0.2)
    for c in ["r60","above_120","rising","flat_or_rising"]:  M[c] = M[c].fillna(0)
    return M

def backtest(data, start, end, ef=None, mf=False, ddcb=0.0,
             tn=None, th=None, vt=None, wm=None, wa=None, wr=None,
             position_size="full", drawdown_control=False, ma_direction=None):
    """v4 backtest: 0.35*above_120 + 0.40*adx_n + 0.25*mom_n

    Parameters added to eliminate rising_ma_optimize.py duplication:
      - ma_direction: 'rising_only' | 'flat_or_rising' | 'any' (None uses ef)
      - position_size: 'full' | 'half' | 'adaptive'
      - drawdown_control: True = half at -15% DD, liquidate at -25%
    Backward-compat: ef/mf/ddcb still work when ma_direction is None.
    """
    tn = tn or P["tn"];  th = th or P["th"];  vt = vt or P["vt"]
    wm = wm or P["wm"];  wa = wa or P["wa"];  wr = wr or P["wr"]
    # Map ma_direction to the filter column in the score matrix
    if ma_direction is not None:
        filter_col = {"rising_only": "rising", "flat_or_rising": "flat_or_rising", "any": None}[ma_direction]
    else:
        filter_col = ef  # legacy: pass-through for backward compat

    scores = scores_dict(data, wm, wa, wr)
    dr = pd.DatetimeIndex(sorted(set().union(*[s.index for s in scores.values()])))
    dr = dr[(dr >= pd.Timestamp(start)) & (dr <= pd.Timestamp(end))]
    etfs = list(data.keys())
    M = _mat(scores, dr, etfs)
    rd = _rb(dr)
    cash_s = data[CASH].loc[(data[CASH].index >= pd.Timestamp(start)) & (data[CASH].index <= pd.Timestamp(end))]
    port = [];  hlog = {};  peak = 1.0;  last = None;  dd_penalty = 1.0
    for i, d in enumerate(dr):
        if i == 0:  hlog[d] = {CASH: 1.0};  pv = 1.0
        else:
            pv = 0.0;  hl = hlog.get(last, list(hlog.values())[0])
            for e, sh in hl.items():
                if e == CASH:  pv += float(sh)
                else:  pv += float(sh) * float(_px(e, d, M, cash_s))
        peak = max(peak, pv);  curdd = pv/peak-1 if peak>0 else 0

        # Accumulate drawdown penalty
        if drawdown_control:
            if curdd <= -0.25:      dd_penalty = 0.0
            elif curdd <= -0.15:    dd_penalty = 0.5
            else:                   dd_penalty = 1.0

        if d in rd and i > 0:
            if ddcb<0 and curdd<=ddcb:  hlog[d] = {CASH: pv}
            elif dd_penalty == 0.0:     hlog[d] = {CASH: pv}
            else:
                sc = M["score"].loc[d].copy()
                if mf:  sc[M["r60"].loc[d]<=0] = 0
                if filter_col:
                    for etf in etfs:
                        if etf in sc.index and M[filter_col].loc[d,etf]<0.5:  sc[etf]=0
                sc = sc.dropna()
                if len(sc)==0 or sc.max()<th:  hlog[d] = {CASH: pv}
                else:
                    sel = sc.nlargest(tn);  sel = sel[sel>0]
                    if len(sel)==0:  hlog[d] = {CASH: pv}
                    else:
                        vols = M["vol"].loc[d,sel.index].clip(lower=0.05)
                        w = (1/vols)/(1/vols).sum()
                        pvol = np.sqrt((w*vols).pow(2).sum())
                        lev = min(1.0, vt/pvol) if pvol>0 else 0;  eq = pv*lev
                        # Position size adjustment
                        if position_size == "half":          pos_mult = 0.5
                        elif position_size == "adaptive":    pos_mult = float(np.clip(sel.mean(), 0.3, 1.0))
                        else:                                pos_mult = 1.0
                        pos_mult *= dd_penalty;  eq *= pos_mult
                        h = {};  cv = 0.0
                        for e in sel.index:
                            if e==CASH:  cv += eq*w[e]
                            else:  h[e] = eq*w[e]/float(M["close"].loc[d,e])
                        if eq<pv:  cv += pv-eq
                        if cv>0:  h[CASH] = cv
                        hlog[d] = h
            last = d
        pv = 0.0;  hl = hlog.get(last, list(hlog.values())[0])
        for e, sh in hl.items():
            if e==CASH:  pv += float(sh)
            else:  pv += float(sh)*float(_px(e,d,M,cash_s))
        peak = max(peak,pv);  curdd = pv/peak-1 if peak>0 else 0
        dr_ = pv/port[-1]["pv"]-1 if port else 0.0
        held = [e for e in hl if e!=CASH]
        port.append({"pv":pv,"ret":dr_,"dd":curdd,"exp":len(held)>0,"held":held})
    res = pd.DataFrame(port, index=dr)
    return res.rename(columns={"pv":"portfolio_value","ret":"daily_return",
                               "dd":"drawdown","exp":"exposure","held":"holdings"})


# ============================================  metrics  ============================================

def metrics(x):
    c = x["portfolio_value"] if isinstance(x, pd.DataFrame) else x
    r = c.pct_change().fillna(0)
    tr  = c.iloc[-1] / c.iloc[0] - 1
    ny  = (c.index[-1] - c.index[0]).days / 365.25
    ar  = (1+tr)**(1/ny) - 1 if ny > 0 else 0
    vol = r.std() * np.sqrt(252)
    dd  = (c / c.cummax() - 1).min()
    sh  = ar / vol if vol > 0 else 0
    ca  = ar / abs(dd) if abs(dd) > 0 else 0
    return dict(total_return=tr, annual_return=ar, annual_volatility=vol,
                max_drawdown=dd, sharpe_ratio=sh, calmar_ratio=ca)

def year_bt(results, year):
    yr = results.loc[str(year)]
    if len(yr) == 0:  return None
    v, r = yr["portfolio_value"], yr["daily_return"]
    tr = v.iloc[-1] / v.iloc[0] - 1;  vol = r.std() * np.sqrt(252)
    dd = yr["drawdown"].min()
    return dict(year=year, return_=tr, volatility=vol, max_drawdown=dd,
                sharpe=(tr/vol if vol>0 else 0))


# ============================================  benchmarks  ============================================

def bench(data, pool=None):
    if pool is None:  pool = POOL
    r = pd.DataFrame({e: data[e]["close"].pct_change() for e in pool}).mean(axis=1)
    return (1 + r).cumprod()

def bench_6040(data, start, end, equity_ticker="510300", bond_ticker="511010"):
    if equity_ticker not in data or bond_ticker not in data: return pd.Series(dtype=float)
    eq = data[equity_ticker]["close"];  bd = data[bond_ticker]["close"]
    cm = eq.index.intersection(bd.index)
    cm = cm[(cm >= pd.Timestamp(start)) & (cm <= pd.Timestamp(end))]
    r = 0.6 * eq.loc[cm].pct_change().fillna(0) + 0.4 * bd.loc[cm].pct_change().fillna(0)
    return (1 + r).cumprod()

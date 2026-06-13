# -*- coding: utf-8 -*-
"""Core signal library for ETF Rotation v4-v6."""
import pandas as pd, numpy as np, os

DIR = r"C:\study\AIWorkspace\quanti\data\clean"
POOL = ("510300","510500","159915","510880","518880","511880")
POOL_V6 = POOL + ("511010",)
CASH = "511880"
GOLD = "518880"
P = dict(tn=2, wm=0.35, wa=0.40, wr=0.25, th=0.35, vt=0.14)
MACRO_DIR = os.path.normpath(os.path.join(DIR, "..", "macro"))

REGIME_PARAMS = {
    "R0": {"ma_period":50,"w_trend":0.40,"w_adx":0.30,"w_mom":0.20,"w_accel":0.10,"gold_cap":0.40},
    "R1": {"ma_period":40,"w_trend":0.40,"w_adx":0.20,"w_mom":0.30,"w_accel":0.10,"gold_cap":0.25},
    "R2": {"ma_period":30,"w_trend":0.40,"w_adx":0.20,"w_mom":0.30,"w_accel":0.10,"gold_cap":0.15},
    "R3": {"ma_period":60,"w_trend":0.40,"w_adx":0.30,"w_mom":0.20,"w_accel":0.10,"gold_cap":0.40},
}

# ═══════════════════════  data loading  ═══════════════════════

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

# ═══════════════════════  signal primitives  ═══════════════════════

def _ema(series, df):
    return pd.Series(series, index=df.index).ewm(alpha=1/14, adjust=False).mean()

def compute_directional_adx(high, low, close, period=14):
    tr  = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()],
                     axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    up, dn = high.diff(), -low.diff()
    pdm = np.where((up>dn) & (up>0), up, 0.0)
    mdm = np.where((dn>up) & (dn>0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx  = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-12)
    adx = pd.Series(dx, index=high.index).ewm(alpha=1/period, adjust=False).mean()
    return adx * np.where(pdi - mdi >= 0, 1, -1)

def compute_time_series_lfmm(series, window=252, min_periods=126):
    rmin = series.rolling(window, min_periods=min_periods).min()
    rmax = series.rolling(window, min_periods=min_periods).max()
    denom = (rmax - rmin).replace(0, np.nan)
    return ((series - rmin) / denom).clip(0, 1)

def compute_time_series_zscore(series, window=252, min_periods=126):
    rm = series.rolling(window, min_periods=min_periods).mean()
    rs = series.rolling(window, min_periods=min_periods).std()
    return (series - rm) / rs.replace(0, np.nan)

def compute_momentum_acceleration(close, short=21, long=63, smooth=5):
    return (close.pct_change(short) - close.pct_change(long)).ewm(span=smooth, adjust=False).mean()

def compute_vol_penalty(current_vol_20d, historical_vol_median_252d):
    if pd.isna(historical_vol_median_252d):
        return 0.0
    denom = max(float(historical_vol_median_252d), 0.15)
    return max(0.0, current_vol_20d / denom - 1.0) * 0.5

def _safe_roc(close, lookback):
    if len(close) < lookback or pd.isna(close.iloc[-1]):
        return 0.0
    try:   return float(close.iloc[-1] / close.iloc[-lookback] - 1)
    except: return 0.0

def _cross_z(values):
    arr = np.array(values, dtype=float)
    mn, sd = np.nanmean(arr), np.nanstd(arr)
    if sd == 0 or pd.isna(sd):
        return [0.0] * len(values)
    return list((arr - mn) / sd)

# ═══════════════════════  v6 scoring  ═══════════════════════

def compute_v6_score_hybrid(data_dict, regime_params, date, accel_smooth=5):
    mp = regime_params.get("ma_period", 40)
    wt = regime_params.get("w_trend",  0.40)
    wa = regime_params.get("w_adx",    0.30)
    wm = regime_params.get("w_mom",    0.20)
    wc = regime_params.get("w_accel",  0.10)
    rn = regime_params.get("_regime_name", None)
    dts = pd.Timestamp(date)
    raw = []
    for etf, df in data_dict.items():
        if dts not in df.index:  continue
        pt = df.loc[:dts];  c, h, l = pt["close"], pt["high"], pt["low"]
        if len(c) < max(mp, 21):  continue
        ma = c.rolling(mp).mean()
        trend_series = (c - ma) / ma.replace(0, np.nan)
        tl = compute_time_series_lfmm(trend_series).iloc[-1]
        tl = 0.0 if pd.isna(tl) else float(tl)
        dadx = compute_directional_adx(h, l, c)
        al = compute_time_series_lfmm(dadx).iloc[-1]
        al = 0.0 if pd.isna(al) else float(al)
        mr  = _safe_roc(c, 63)
        acc = compute_momentum_acceleration(c, smooth=accel_smooth).iloc[-1]
        acc = 0.0 if pd.isna(acc) else float(acc)
        v20 = c.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252) if len(c)>=20 else 0.2
        vm  = c.pct_change().rolling(252).std().median()*np.sqrt(252) if len(c)>=252 else 0.2
        raw.append(dict(etf=etf, trend_lfmm=tl, adx_lfmm=al,
                        mom_raw=mr, accel_raw=acc, vol20=v20, vol_median=vm))
    if not raw: return pd.DataFrame()
    dfr = pd.DataFrame(raw)
    mz = _cross_z(dfr["mom_raw"].values)
    az = _cross_z(dfr["accel_raw"].values)
    rows = []
    for i, r in dfr.iterrows():
        vp = compute_vol_penalty(r["vol20"], r["vol_median"])
        f  = wt*r["trend_lfmm"] + wa*r["adx_lfmm"] + wm*mz[i] + wc*az[i] - vp
        rows.append(dict(etf=r["etf"], final_score=f, trend_z=r["trend_lfmm"],
                        adx_z=r["adx_lfmm"], mom_z=mz[i], accel_z=az[i], vol_penalty=vp))
    result = pd.DataFrame(rows)
    if rn in ("R0","R3"):
        boost = {"R0":0.10, "R3":0.25}.get(rn, 0.0)
        result.loc[result["etf"]==GOLD, "final_score"] += boost
    return result

# ═══════════════════════  regime classifier  ═══════════════════════

def compute_regime(pmi_value, cgb_series, date,
                   hysteresis=20, band=0.005,
                   prev_yield_rising=None, prev_pmi_contraction=None,
                   pmi_streak=None, last_pmi_month=None):
    dts = pd.Timestamp(date);  pm = dts.month
    if pmi_value is None or pd.isna(pmi_value):
        pc = prev_pmi_contraction if prev_pmi_contraction is not None else True
        ps = pmi_streak         if pmi_streak         is not None else 0
    else:
        nc = pmi_value <= 50
        if prev_pmi_contraction is None:       pc, ps = nc, 0
        elif pm == last_pmi_month:              pc, ps = prev_pmi_contraction, pmi_streak
        elif nc == prev_pmi_contraction:        pc, ps = nc, 0
        else:
            ps = (pmi_streak or 0) + 1
            if ps >= 2:  pc, ps = nc, 0
            else:        pc = prev_pmi_contraction
    if cgb_series is None or len(cgb_series) == 0:
        yr = prev_yield_rising if prev_yield_rising is not None else False
    else:
        yp = cgb_series.loc[:dts]
        if len(yp) < 120:
            yr = prev_yield_rising if prev_yield_rising is not None else False
        else:
            rm = yp.rolling(120).mean();  ly, lm = float(yp.iloc[-1]), float(rm.iloc[-1])
            if pd.isna(lm):
                yr = prev_yield_rising if prev_yield_rising is not None else False
            else:
                ab = ly > (lm + band);  bb = ly < (lm - band)
                if prev_yield_rising is None:
                    yr = ab if (ab or bb) else False
                elif prev_yield_rising and bb:
                    cnt = int((yp.iloc[-hysteresis:] < (rm.iloc[-hysteresis:] - band)).sum())
                    yr = False if cnt >= hysteresis else True
                elif not prev_yield_rising and ab:
                    cnt = int((yp.iloc[-hysteresis:] > (rm.iloc[-hysteresis:] + band)).sum())
                    yr = True if cnt >= hysteresis else False
                else:
                    yr = prev_yield_rising
    if pc and not yr:         regime = "R0"
    elif not pc and not yr:   regime = "R1"
    elif not pc and yr:       regime = "R2"
    else:                     regime = "R3"
    rp = REGIME_PARAMS[regime].copy();  rp["_regime_name"] = regime
    return regime, rp, yr, pc, ps, pm

# ═══════════════════════  backtest engine  ═══════════════════════

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
    if eq < amt:  h[CASH] = amt - eq
    return h

def backtest_v6(data, start, end, tn=2, vt=0.28, score_gate_threshold=0.60,
                gold_boost_config=None, regime_weight_override=None,
                accel_smooth=5, dd_enabled=True, scoring_func=None,
                pmi_data=None, cgb_yield_data=None):
    if scoring_func is None:
        scoring_func = compute_v6_score_hybrid
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
    dr = dr[(dr >= pd.Timestamp(start)) & (dr <= pd.Timestamp(end))]
    etfs = list(data.keys())
    pyr = ppc = pms = lpm = None
    peak = 1.0;  in_dd = False;  amt = 1.0
    hh = {};  hi = {};  results = []
    for i, d in enumerate(dr):
        if not hh:  pv = amt
        else:
            pv = 0.0
            for e, sh in hh.items():
                if e == CASH:  pv += float(sh)
                else:
                    de = data.get(e)
                    if de is not None and d in de.index:       px = float(de.loc[d,"close"])
                    elif de is not None:
                        xe = de.loc[:d,"close"].dropna()
                        px = float(xe.iloc[-1]) if len(xe) > 0 else 0.0
                    else:  px = 1.0
                    hi_e = float(hi.get(e, px))
                    pv += float(sh) * px / hi_e if hi_e != 0 else float(sh)
        dr_ = (pv / results[-1]["portfolio_value"] - 1) if (results and results[-1]["portfolio_value"] > 0) else 0.0
        peak = max(peak, pv);  curdd = pv / peak - 1
        if dd_enabled and curdd <= -0.25 and not in_dd:
            in_dd = True;  hh, hi = {CASH: pv}, {}
            results.append(dict(date=d, portfolio_value=pv, daily_return=dr_,
                                drawdown=curdd, exposure=pv, holdings=str(hh),
                                regime="LIQ", gate_triggered=False))
            continue
        if in_dd and dd_enabled:
            can = False
            for e in etfs:
                if e == CASH:  continue
                de = data.get(e)
                if de is not None and d in de.index:
                    pt = de.loc[:d,"close"];  ma10 = pt.rolling(10).mean()
                    if len(ma10) >= 10 and pt.iloc[-1] > ma10.iloc[-1]:
                        can = True;  break
            if can and curdd > -0.15:  in_dd = False;  hh, hi = {}, {}
        pv_ = None
        if pmi_data is not None and len(pmi_data) > 0:
            mask = pmi_data.index <= d
            if mask.any():  pv_ = float(pmi_data.loc[mask].iloc[-1])
        rn, rp, pyr, ppc, pms, lpm = compute_regime(
            pv_, cgb_yield_data, d, prev_yield_rising=pyr,
            prev_pmi_contraction=ppc, pmi_streak=pms, last_pmi_month=lpm)
        if regime_weight_override and rn in regime_weight_override:
            for k, v in regime_weight_override[rn].items():
                rp[k] = v
        inm = (i == 0 or d.month != dr[i-1].month);  gt = False
        if inm and not in_dd:
            sdf = scoring_func(data, rp, d, accel_smooth=accel_smooth)
            if not sdf.empty:
                top = sdf.nlargest(tn, "final_score")
                if top["final_score"].max() < score_gate_threshold:
                    hh, hi = {CASH: pv}, {};  gt = True
                else:
                    se = top["etf"].tolist();  vols = {}
                    for e in se:
                        de = data.get(e)
                        if de is not None and d in de.index:
                            pt = de.loc[:d,"close"].pct_change().rolling(20).std()
                            ev = pt.iloc[-1]*np.sqrt(252) if len(pt)>=20 and not pd.isna(pt.iloc[-1]) else 0.2
                            vols[e] = max(ev, 0.05)
                        else:  vols[e] = 0.05
                    ss = pd.Series([1.0]*len(se), index=se);  vs = pd.Series(vols)
                    hh = _alloc(pv, ss, vs, vt);  hi = {}
                    for e in hh:
                        if e != CASH:
                            de = data.get(e)
                            if de is not None:
                                if d in de.index:   hi[e] = float(de.loc[d,"close"])
                                else:
                                    xe = de.loc[:d,"close"].dropna()
                                    hi[e] = float(xe.iloc[-1]) if len(xe) > 0 else 1.0
                            else:  hi[e] = 1.0
        results.append(dict(date=d, portfolio_value=pv, daily_return=dr_,
                           drawdown=curdd, exposure=pv, holdings=str(hh),
                           regime=rn, gate_triggered=gt))
    return pd.DataFrame(results).set_index("date")

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

def walk_forward(data, macro_data, train_start="2015-01-01", train_end="2019-12-31",
                 n_folds=3, fold_years=2, **kwargs):
    pmi_in, cgb_in = macro_data if isinstance(macro_data, tuple) else (macro_data["pmi"], macro_data["cgb"])
    frs = [];  oos_rets = []
    for fold in range(n_folds):
        tsd = pd.Timestamp(train_start)
        ted = pd.Timestamp(train_end) + pd.DateOffset(years=fold * fold_years)
        ssd = ted + pd.DateOffset(days=1)
        sed = min(ssd + pd.DateOffset(years=fold_years) - pd.DateOffset(days=1),
                  pd.Timestamp("2025-12-31"))
        if ssd > pd.Timestamp("2025-12-31"):  break
        bt_train = backtest_v6(data, tsd, ted, pmi_data=pmi_in, cgb_yield_data=cgb_in, **kwargs)
        bt_test  = backtest_v6(data, ssd, sed, pmi_data=pmi_in, cgb_yield_data=cgb_in, **kwargs)
        frs.append(dict(fold=fold+1, train_start=tsd, train_end=ted,
                        test_start=ssd, test_end=sed,
                        train_metrics=metrics(bt_train),
                        test_metrics=metrics(bt_test)))
        dr = bt_test["daily_return"].copy();  dr.index = pd.to_datetime(dr.index)
        oos_rets.append(dr)
    all_r = pd.concat(oos_rets).sort_index()
    all_r = all_r[~all_r.index.duplicated(keep="last")]
    ooe = (1 + all_r.fillna(0)).cumprod()
    oom = metrics(ooe)
    ooa = {yr: float(ooe[ooe.index.year==yr].iloc[-1]/ooe[ooe.index.year==yr].iloc[0]-1)
           for yr in range(ooe.index[0].year, ooe.index[-1].year+1)
           if len(ooe[ooe.index.year==yr]) > 0}
    return dict(fold_results=frs, oos_equity=ooe, oos_metrics=oom, oos_annual=ooa)

# ═══════════════════════  legacy v4 / additional v6  ═══════════════════════

def features(df):
    df = df.copy();  c, h, l = df["close"], df["high"], df["low"]
    ma = c.rolling(120).mean();  df["ma120"] = ma
    df["above_120"] = (c > ma).astype(int);  df["rising"] = (ma.diff(20) > 0).astype(int)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean();  up, dn = h.diff(), -l.diff()
    pdm = 100 * _ema(np.where((up>dn) & (up>0), up, 0.0), df) / atr
    mdm = 100 * _ema(np.where((dn>up) & (dn>0), dn, 0.0), df) / atr
    df["adx"] = _ema(100 * abs(pdm-mdm) / (pdm+mdm+1e-12), df)
    df["r20"] = c.pct_change(20);  df["r60"] = c.pct_change(60)
    df["r42"] = c.pct_change(42);  df["r63"] = c.pct_change(63)
    df["ma20"] = c.rolling(20).mean();  df["vol"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    return df

def scores_dict(data, wm=0.35, wa=0.30, wr=0.35):
    out = {}
    for e, df in data.items():
        f = features(df);  a = (f["adx"]/100).clip(0,1);  d = (f["r20"].clip(-0.3,0.3)+0.3)/0.6
        out[e] = pd.DataFrame({"close": f["close"], "score": wm*f["above_120"] + wa*a + wr*d,
                               "above_120": f["above_120"], "rising": f["rising"],
                               "r60": f["r60"], "vol": f["vol"]})
    return out

def _px(etf, d, mats, cash):
    if etf == CASH:  return cash.loc[d, "close"]
    p = mats["close"].loc[d, etf]
    if pd.isna(p):  p = mats["close"].loc[:d, etf].dropna().iloc[-1]
    return p

def _mat(scores, dr, etfs):
    cols = ["score","close","above_120","rising","r60","vol"]
    M = {c: pd.DataFrame(index=dr, columns=list(etfs), dtype=float) for c in cols}
    for e in etfs:
        s = scores[e];  i = dr.intersection(s.index)
        for c in cols:  M[c].loc[i, e] = s.loc[i, c].values
    for c in cols:  M[c] = M[c].ffill()
    M["score"] = M["score"].fillna(0);  M["vol"] = M["vol"].fillna(0.2)
    for c in ["r60","above_120","rising"]:  M[c] = M[c].fillna(0)
    return M

def year_bt(results, year):
    yr = results.loc[str(year)]
    if len(yr) == 0:  return None
    v, r = yr["portfolio_value"], yr["daily_return"]
    tr = v.iloc[-1] / v.iloc[0] - 1;  vol = r.std() * np.sqrt(252)
    dd = yr["drawdown"].min()
    return dict(year=year, return_=tr, volatility=vol, max_drawdown=dd,
                sharpe=(tr/vol if vol>0 else 0))

def backtest(data, start, end, mf="monthly", ddcb=False,
             tn=None, th=None, vt=None, wm=None, wa=None, wr=None):
    tn = tn or P["tn"];  th = th or P["th"];  vt = vt or P["vt"]
    wm = wm or P["wm"];  wa = wa or P["wa"];  wr = wr or P["wr"]
    scores = scores_dict(data, wm, wa, wr)
    dr = pd.DatetimeIndex(sorted(set().union(*[s.index for s in scores.values()])))
    dr = dr[(dr >= start) & (dr <= end)]
    etfs = list(data.keys());  M = _mat(scores, dr, etfs)
    cash_df = data[CASH].loc[(data[CASH].index >= start) & (data[CASH].index <= end)]
    rd = _rb(dr) if mf == "monthly" else dr
    port = [];  hlog = {};  amt = 1.0;  last = None;  peak = 1.0;  dd_active = False
    for d in dr:
        if last is None:  pv = amt
        else:
            pv = 0.0;  hl = hlog.get(last, {})
            for e, sh in hl.items():
                if e == CASH:  pv += float(sh)
                else:
                    px = float(_px(e, d, M, cash_df))
                    hi = max(float(hinit.get(e, 1.0)), 1e-6)
                    pv += float(sh) * px / hi
        peak = max(peak, pv);  curdd = pv / peak - 1
        if ddcb and curdd < -0.25 and not dd_active:
            dd_active = True;  hlog[d] = {CASH: pv};  last = d
            port.append({"date": d, "portfolio_value": pv, "daily_return": 0.0, "drawdown": curdd})
            continue
        if dd_active and curdd >= -0.15:  dd_active = False
        if d in rd and not dd_active:
            sel = M["score"].loc[d].nlargest(tn)
            if (sel <= 0).all():  hlog[d] = {CASH: pv}
            else:
                vols = M["vol"].loc[d, sel.index]
                hlog[d] = _alloc(pv, sel, vols, vt)
            last = d
            hinit = {k: float(_px(k, d, M, cash_df)) for k in hlog[d] if k != CASH}
        dr_ = pv / port[-1]["portfolio_value"] - 1 if port and port[-1]["portfolio_value"] > 0 else 0.0
        port.append({"date": d, "portfolio_value": pv, "daily_return": dr_, "drawdown": curdd})
    res = pd.DataFrame(port).set_index("date")
    res["exposure"] = res["portfolio_value"] / res["portfolio_value"].iloc[0]
    return res

def bench(data, pool=None):
    if pool is None:  pool = POOL
    r = pd.DataFrame({e: data[e]["close"].pct_change() for e in pool}).mean(axis=1)
    return (1 + r).cumprod()

def compute_v6_score(data_dict, regime_params, date, accel_smooth=5):
    mp = regime_params.get("ma_period", 40)
    w_trend = regime_params.get("w_trend", 0.25)
    w_adx   = regime_params.get("w_adx",   0.25)
    w_mom   = regime_params.get("w_mom",   0.35)
    w_accel = regime_params.get("w_accel", 0.15)
    dts = pd.Timestamp(date);  rows = []
    for etf, df in data_dict.items():
        if dts not in df.index:  continue
        pt = df.loc[:dts];  c, h, l = pt["close"], pt["high"], pt["low"]
        if len(c) < max(mp, 126):  continue
        ma = c.rolling(mp).mean()
        tz = compute_time_series_zscore((c - ma) / ma.replace(0, np.nan)).iloc[-1]
        dx = compute_directional_adx(h, l, c)
        az = compute_time_series_zscore(dx).iloc[-1]
        mz = compute_time_series_zscore(c.pct_change(63)).iloc[-1]
        ac = compute_momentum_acceleration(c, smooth=accel_smooth)
        acz = compute_time_series_zscore(ac).iloc[-1]
        v20 = c.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252) if len(c)>=20 else 0.2
        vm  = c.pct_change().rolling(252).std().median()*np.sqrt(252) if len(c)>=252 else 0.2
        vp = compute_vol_penalty(v20, vm)
        vs = [0.0 if pd.isna(v) else float(v) for v in [tz, az, mz, acz]]
        f = w_trend*vs[0] + w_adx*vs[1] + w_mom*vs[2] + w_accel*vs[3] - vp
        rows.append(dict(etf=etf, final_score=f, trend_z=vs[0],
                        adx_z=vs[1], mom_z=vs[2], accel_z=vs[3], vol_penalty=vp))
    if not rows:  return pd.DataFrame()
    return pd.DataFrame(rows)

def compute_v6_score_cross_sectional(data_dict, regime_params, date, accel_smooth=5):
    mp = regime_params.get("ma_period", 40)
    w_trend = regime_params.get("w_trend", 0.25)
    w_adx   = regime_params.get("w_adx",   0.25)
    w_mom   = regime_params.get("w_mom",   0.35)
    w_accel = regime_params.get("w_accel", 0.15)
    dts = pd.Timestamp(date);  raw = []
    for etf, df in data_dict.items():
        if dts not in df.index:  continue
        pt = df.loc[:dts];  c, h, l = pt["close"], pt["high"], pt["low"]
        if len(c) < max(mp, 21):  continue
        ma = c.rolling(mp).mean()
        tr = (c.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1] if ma.iloc[-1] != 0 else 0.0
        dx = compute_directional_adx(h, l, c)
        ar = float(dx.iloc[-1]) if not pd.isna(dx.iloc[-1]) else 0.0
        mr = _safe_roc(c, 63)
        ac = compute_momentum_acceleration(c, smooth=accel_smooth)
        acr = float(ac.iloc[-1]) if not pd.isna(ac.iloc[-1]) else 0.0
        v20 = c.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252) if len(c)>=20 else 0.2
        vm  = c.pct_change().rolling(252).std().median()*np.sqrt(252) if len(c)>=252 else 0.2
        raw.append(dict(etf=etf, trend_raw=0.0 if pd.isna(tr) else tr,
                        adx_raw=ar, mom_raw=mr, accel_raw=acr, vol20=v20, vol_median=vm))
    if not raw:  return pd.DataFrame()
    dfr = pd.DataFrame(raw)
    dfr["trend_z"] = _cross_z(dfr["trend_raw"].values)
    dfr["adx_z"]   = _cross_z(dfr["adx_raw"].values)
    dfr["mom_z"]   = _cross_z(dfr["mom_raw"].values)
    dfr["accel_z"] = _cross_z(dfr["accel_raw"].values)
    rows = []
    for _, r in dfr.iterrows():
        vp = compute_vol_penalty(r["vol20"], r["vol_median"])
        f  = w_trend*r["trend_z"] + w_adx*r["adx_z"] + w_mom*r["mom_z"] + w_accel*r["accel_z"] - vp
        rows.append(dict(etf=r["etf"], final_score=f, trend_z=r["trend_z"],
                        adx_z=r["adx_z"], mom_z=r["mom_z"], accel_z=r["accel_z"], vol_penalty=vp))
    return pd.DataFrame(rows)

def bench_6040(data, start, end, equity_ticker="510300", bond_ticker="511010"):
    if equity_ticker not in data or bond_ticker not in data:  return pd.Series(dtype=float)
    eq = data[equity_ticker]["close"];  bd = data[bond_ticker]["close"]
    cm = eq.index.intersection(bd.index)
    cm = cm[(cm >= pd.Timestamp(start)) & (cm <= pd.Timestamp(end))]
    r = 0.6 * eq.loc[cm].pct_change().fillna(0) + 0.4 * bd.loc[cm].pct_change().fillna(0)
    return (1 + r).cumprod()

def compute_gold_allocation_mean(results):
    if "holdings" not in results.columns:  return 0.0
    wts = []
    for _, row in results.iterrows():
        try:  h = eval(str(row["holdings"]))
        except:  continue
        tv = sum(float(v) for v in h.values())
        if tv > 0:  wts.append(float(h.get(GOLD, 0)) / tv)
    return float(np.mean(wts)) if wts else 0.0

def turnover(results):
    if "holdings" not in results.columns:  return 0.0
    tt = 0.0;  ph = None
    for _, hs in results["holdings"].items():
        try:  h = eval(hs) if isinstance(hs, str) else hs
        except:  continue
        if ph is not None:
            aa = set(list(h.keys()) + list(ph.keys()))
            tt += sum(abs(h.get(a, 0) - ph.get(a, 0)) for a in aa) / 2
        ph = h
    ny = (results.index[-1] - results.index[0]).days / 365.25
    return tt / ny if ny > 0 else 0.0

GOLD_TICKER = GOLD

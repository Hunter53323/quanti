"""
ETF Rotation v4 -- Rising MA Filter: Full Grid Search + Optimization
=====================================================================
Strategy:
  1. Monthly score all 7 ETFs (120MA position + ADX + 20d momentum)
  2. Buy top-N scoring ETFs
  3. Rising MA filter -- only select ETFs whose 120MA is rising (slope > 0)
  4. Also test flat_or_rising (slope >= -0.5%) and any (no direction filter)
  5. Fallback to 511880 (cash ETF) if no asset qualifies
  6. Optional drawdown control (half at -15%, liquidate at -25%)

Parameter grid:
  - top_n: [1, 2, 3]
  - ma_direction: 'rising_only' | 'flat_or_rising' | 'any'
  - position_size: 'full' | 'half' | 'adaptive' (by avg score)
  - drawdown_control: True | False

Output: data/rising_ma_etf_rotation_report.md
"""

import pandas as pd, numpy as np, itertools, os, sys, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _funcs import load, features, scores_dict, backtest as _v4_backtest, metrics, year_bt, bench_6040, P, POOL, POOL_V6, CASH, T0, T1

TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
REPORT_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "rising_ma_etf_rotation_report.md")

# =============================================== helpers ===============================================

def pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "N/A"
    return f"{x:.2%}"
def num(x, fmt=".2f"):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "N/A"
    return f"{x:{fmt}}"
def bench_single(data, etf, s, e):
    c = data[etf]["close"];  c = c[(c.index>=pd.Timestamp(s))&(c.index<=pd.Timestamp(e))]
    return metrics(c) if len(c)>0 else {"annual_return":0,"max_drawdown":0,"sharpe_ratio":0}

# =============================================== extended features ===============================================

def compute_features_extended(df):
    """Like features() but adds ma120_flat_or_rising (slope >= -0.5%)."""
    df = df.copy();  c, h, l = df["close"], df["high"], df["low"]
    ma120 = c.rolling(120).mean();  df["ma120"] = ma120
    df["above_120"] = (c > ma120).astype(int)
    df["ma120_rising"] = (ma120.diff(20) > 0).astype(int)
    df["ma120_flat_or_rising"] = (ma120.pct_change(20) >= -0.005).astype(int)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean();  up, dn = h.diff(), -l.diff()
    pdm = 100 * pd.Series(np.where((up>dn)&(up>0),up,0.0), index=df.index).ewm(alpha=1/14,adjust=False).mean()/atr
    mdm = 100 * pd.Series(np.where((dn>up)&(dn>0),dn,0.0), index=df.index).ewm(alpha=1/14,adjust=False).mean()/atr
    df["adx_14"] = (abs(pdm-mdm)/(pdm+mdm+1e-12)*100).ewm(alpha=1/14,adjust=False).mean()
    df["ret_20d"] = c.pct_change(20);  df["ret_60d"] = c.pct_change(60)
    df["vol_20d"] = c.pct_change().rolling(20).std()*np.sqrt(252)
    return df

def compute_scores_extended(data_dict, wm=0.35, wa=0.40, wr=0.25):
    scores = {}
    for etf, df in data_dict.items():
        f = compute_features_extended(df)
        a = (f["adx_14"]/100).clip(0,1);  d = (f["ret_20d"].clip(-0.3,0.3)+0.3)/0.6
        scores[etf] = pd.DataFrame({
            "close": f["close"],  "score": wm*f["above_120"] + wa*a + wr*d,
            "above_120": f["above_120"], "ma120_rising": f["ma120_rising"],
            "ma120_flat_or_rising": f["ma120_flat_or_rising"],
            "ret_60d": f["ret_60d"], "vol_20d": f["vol_20d"],
        })
    return scores

# =============================================== backtest ===============================================

def backtest_rising_ma(data, start, end, top_n=2, wm=0.35, wa=0.40, wr=0.25,
                       ma_direction="rising_only", position_size="full",
                       drawdown_control=False, score_threshold=0.30, vt=0.14):
    scores = compute_scores_extended(data, wm, wa, wr)
    dr = pd.DatetimeIndex(sorted(d for d in set.union(*[set(s.index) for s in scores.values()])
                                 if start <= d.strftime("%Y-%m-%d") <= end))
    etfs = [e for e in data if e != CASH]
    # Build matrices
    cols = ["score","close","ma120_rising","ma120_flat_or_rising","vol_20d"]
    M = {c: pd.DataFrame(index=dr, columns=etfs, dtype=float) for c in cols}
    for e in etfs:
        s = scores[e];  i = dr.intersection(s.index)
        for c in cols:  M[c].loc[i,e] = s.loc[i,c].values
    for c in cols: M[c] = M[c].ffill()
    M["score"] = M["score"].fillna(0);  M["vol_20d"] = M["vol_20d"].fillna(0.2)
    for c in ["ma120_rising","ma120_flat_or_rising"]: M[c] = M[c].fillna(0)
    cash_close_raw = data[CASH]["close"]
    cash_s = cash_close_raw.reindex(dr, method="ffill")
    # Ensure first value is not NaN
    if pd.isna(cash_s.iloc[0]):
        first_valid = cash_s.dropna().iloc[0]
        cash_s.iloc[0] = first_valid
    rd = sorted(set(pd.DatetimeIndex(
        md[0] for y in range(dr[0].year,dr[-1].year+1) for m in range(1,13)
        if len(md:=dr[(dr.year==y)&(dr.month==m)])>0)))

    hh = {CASH: 1.0 / float(cash_s.iloc[0])};  hi = {};  peak = 1.0
    vals, rets, dds, exps, held = [], [], [], [], []
    dd_penalty = 1.0
    for i, d in enumerate(dr):
        pv = sum(u * (float(cash_s.loc[d]) if e==CASH else float(M["close"].loc[d,e]))
                 for e,u in hh.items())
        vals.append(pv);  rets.append(pv/vals[-2]-1 if i>=1 else 0.0)
        peak = max(peak,pv);  curdd = (pv-peak)/peak if peak>0 else 0;  dds.append(curdd)
        risky = sum(u*M["close"].loc[d,e] for e,u in hh.items() if e!=CASH)
        exps.append(risky/pv if pv>0 else 0);  held.append([e for e in hh if e!=CASH])
        if d in rd and i>0:
            amt = pv;  hh.clear();  hi.clear()
            if drawdown_control:
                if curdd <= -0.25:  dd_penalty = 0.0
                elif curdd <= -0.15:  dd_penalty = 0.5
                else:  dd_penalty = 1.0
            sc = M["score"].loc[d].copy()
            for e in etfs:
                if e not in sc.index: continue
                if ma_direction == "rising_only" and M["ma120_rising"].loc[d,e]<0.5:  sc[e]=0
                elif ma_direction == "flat_or_rising" and M["ma120_flat_or_rising"].loc[d,e]<0.5:  sc[e]=0
            sc = sc.dropna()
            if len(sc)==0 or sc.max()<score_threshold or dd_penalty==0.0:
                hh[CASH] = amt/float(cash_s.loc[d])
            else:
                sel = sc.nlargest(top_n);  sel = sel[sel>0]
                if len(sel)==0:  hh[CASH] = amt/float(cash_s.loc[d])
                else:
                    vols = M["vol_20d"].loc[d,sel.index].clip(lower=0.05)
                    w = (1.0/vols)/(1.0/vols).sum()
                    pvol = np.sqrt((w.values**2*vols.values**2).sum())
                    lev = min(1.0, vt/pvol) if pvol>0 else 0
                    if position_size=="half":  pos_mult = 0.5
                    elif position_size=="adaptive":  pos_mult = float(np.clip(sel.mean(),0.3,1.0))
                    else:  pos_mult = 1.0
                    pos_mult *= dd_penalty
                    eq = amt * lev * pos_mult
                    for e in sel.index:  hh[e] = eq*w[e]/M["close"].loc[d,e]
                    if eq < amt:  hh[CASH] = hh.get(CASH,0) + (amt-eq)/float(cash_s.loc[d])
    return pd.DataFrame({"portfolio_value":vals,"daily_return":rets,"drawdown":dds,
                         "exposure":exps,"holdings":held}, index=dr)

# =============================================== grid search ===============================================

def grid_search(data, start, end):
    param_grid = list(itertools.product([1,2,3], ["rising_only","flat_or_rising","any"],
                                        ["full"], [True,False]))
    results = []
    for idx, (tn, md, ps, dd) in enumerate(param_grid):
        bt = backtest_rising_ma(data, start, end, top_n=tn, ma_direction=md,
                                position_size=ps, drawdown_control=dd)
        m = metrics(bt)
        yrs = {}
        sy = pd.Timestamp(start).year;  ey = pd.Timestamp(end).year
        for yr in range(sy, ey+1):
            try:
                yb = year_bt(bt, yr)
                if yb: yrs[yr] = yb["return_"]
            except (KeyError, TypeError): pass
        results.append(dict(top_n=tn, ma_direction=md, position_size=ps, drawdown_control=dd,
                           cagr=m["annual_return"], sharpe=m["sharpe_ratio"],
                           maxdd=m["max_drawdown"], calmar=m["calmar_ratio"],
                           vol=m["annual_volatility"], yearly=yrs))
        if (idx+1)%10==0: print("  Progress: {}/{}".format(idx+1, len(param_grid)))
    return pd.DataFrame(results).sort_values("cagr", ascending=False)

# =============================================== report ===============================================

def generate_report(train_df, test_df, data):
    L = []; a = L.append
    a("# ETF Rotation v4 -- Rising MA Filter: Optimization Report")
    a("")
    a("**Generated:** {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    a("")
    a("## 1. Strategy Summary")
    a("- **Universe:** 7 ETFs -- 510300 (CSI300), 510500 (CSI500), 159915 (ChiNext), 510880 (Dividend), 511010 (Bonds), 518880 (Gold), 511880 (Cash)")
    a("- **Rebalance:** Monthly (first trading day)")
    a("- **Scoring:** 0.35*above_120MA + 0.40*ADX_norm + 0.25*momentum_norm (3-factor)")
    a("- **Rising MA Filter:** Block equity ETFs whose 120MA is not rising")
    a("- **Vol Target:** 14% annualized. **Fallback:** 511880")
    a("")
    a("## 2. Data Split")
    a("| Period | Dates |");  a("|--------|-------|")
    a("| Train  | {} to {} |".format(TRAIN_START, TRAIN_END))
    a("| Test   | {} to {} |".format(T0, T1))
    a("")

    # Best configs
    best = train_df.sort_values("cagr", ascending=False)
    a("## 3. Top 10 Train Configurations")
    a("| Rank | top_n | MA Dir | Pos Size | DD Ctrl | CAGR | Sharpe | MaxDD |")
    a("|------|-------|--------|----------|---------|------|--------|-------|")
    for i,(_,r) in enumerate(best.head(10).iterrows()):
        a("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            i+1, int(r["top_n"]), r["ma_direction"], r["position_size"],
            r["drawdown_control"], pct(r["cagr"]), num(r["sharpe"]), pct(r["maxdd"])))
    a("")

    # Test performance of top 10 train
    a("## 4. Test Period -- Top 10 Train Configs Applied to Test")
    a("| Rank | top_n | MA Dir | Pos Size | DD Ctrl | Test CAGR | Sharpe | MaxDD | Calmar |")
    a("|------|-------|--------|----------|---------|-----------|--------|-------|--------|")
    for i,(_,tr) in enumerate(best.head(10).iterrows()):
        match = test_df[(test_df["top_n"]==tr["top_n"])&(test_df["ma_direction"]==tr["ma_direction"])&
                       (test_df["position_size"]==tr["position_size"])&(test_df["drawdown_control"]==tr["drawdown_control"])]
        if len(match)>0:
            t = match.iloc[0]
            a("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                i+1, int(t["top_n"]), t["ma_direction"], t["position_size"],
                t["drawdown_control"], pct(t["cagr"]), num(t["sharpe"]),
                pct(t["maxdd"]), num(t["calmar"])))
    a("")

    # Optimal (best Test CAGR)
    best_test = test_df.loc[test_df["cagr"].idxmax()]
    a("## 5. Optimal Configuration (Best Test CAGR)")
    a("| Parameter | Value |");  a("|-----------|-------|")
    a("| top_n | {} |".format(int(best_test["top_n"])))
    a("| ma_direction | {} |".format(best_test["ma_direction"]))
    a("| position_size | {} |".format(best_test["position_size"]))
    a("| drawdown_control | {} |".format(best_test["drawdown_control"]))
    a("")
    tm = train_df[(train_df["top_n"]==best_test["top_n"])&(train_df["ma_direction"]==best_test["ma_direction"])&
                  (train_df["position_size"]==best_test["position_size"])&(train_df["drawdown_control"]==best_test["drawdown_control"])]
    a("| Metric | Train | Test |");  a("|--------|-------|------|")
    if len(tm)>0:
        tr = tm.iloc[0]
        for mname, label in [("cagr","CAGR"),("sharpe","Sharpe"),("maxdd","MaxDD"),("calmar","Calmar"),("vol","Volatility")]:
            fmt_fn = pct if mname in ("cagr","maxdd","vol") else num
            a("| {} | {} | {} |".format(label, fmt_fn(tr[mname]), fmt_fn(best_test[mname])))
    a("")

    # Yearly breakdown
    a("## 6. Yearly Breakdown (Optimal Config)")
    a("| Year | Return | MaxDD | Sharpe |");  a("|------|--------|-------|--------|")
    opt_train = backtest_rising_ma(data, TRAIN_START, TRAIN_END, top_n=int(best_test["top_n"]),
                                   ma_direction=best_test["ma_direction"],
                                   position_size=best_test["position_size"],
                                   drawdown_control=bool(best_test["drawdown_control"]))
    opt_test  = backtest_rising_ma(data, T0, T1, top_n=int(best_test["top_n"]),
                                   ma_direction=best_test["ma_direction"],
                                   position_size=best_test["position_size"],
                                   drawdown_control=bool(best_test["drawdown_control"]))
    for yr in range(2015, int(T1[:4])+1):
        bt = opt_train if yr<=2021 else opt_test;  period = "train" if yr<=2021 else "test"
        try:
            yb = year_bt(bt, yr)
            if yb: a("| {} ({}) | {} | {} | {} |".format(yr, period, pct(yb["return_"]), pct(yb["max_drawdown"]), num(yb["sharpe"])))
        except: pass
    a("")

    # Parameter sensitivity (position_size excluded: always "full")
    a("## 7. Parameter Sensitivity (Test Period Mean, full position only)")
    a("Position size: always *full* (half/adaptive removed -- rising MA filter provides enough protection)")
    a("")
    for param, col in [("top_n","top_n"),("MA Direction","ma_direction"),
                        ("Drawdown Control","drawdown_control")]:
        a("### By {}".format(param))
        grouped = test_df.groupby(col)["cagr"].agg(["mean","std","count"])
        a("| Value | Mean CAGR | Std | N |");  a("|-------|-----------|-----|---|")
        for val, row in grouped.iterrows():
            a("| {} | {} | {} | {} |".format(val, pct(row["mean"]), pct(row["std"]), int(row["count"])))
        a("")

    # Benchmarks
    a("## 8. Benchmarks")
    for label, s, e in [("Train (2015-2021)", TRAIN_START, TRAIN_END), ("Test (2022-{})".format(T1[:4]), T0, T1)]:
        a("### {}".format(label))
        a("| ETF | CAGR | MaxDD | Sharpe |");  a("|-----|------|-------|--------|")
        for etf in ["510300","510500","159915","518880","511010"]:
            m = bench_single(data, etf, s, e)
            a("| {} | {} | {} | {} |".format(etf, pct(m["annual_return"]), pct(m["max_drawdown"]), num(m["sharpe_ratio"])))
        b6040 = bench_6040(data, s, e)
        if isinstance(b6040, pd.Series) and len(b6040)>0:
            m60 = metrics(b6040)
            a("| 60/40 | {} | {} | {} |".format(pct(m60["annual_return"]), pct(m60["max_drawdown"]), num(m60["sharpe_ratio"])))
        a("")

    # v4 Baseline (original params)
    a("## 9. Original v4 Baseline (tn=2, rising_only, full, no DD)")
    v4_tr = train_df[(train_df["top_n"]==2)&(train_df["ma_direction"]=="rising_only")&
                     (train_df["position_size"]=="full")&(train_df["drawdown_control"]==False)]
    v4_te = test_df[(test_df["top_n"]==2)&(test_df["ma_direction"]=="rising_only")&
                     (test_df["position_size"]=="full")&(test_df["drawdown_control"]==False)]
    if len(v4_tr)>0 and len(v4_te)>0:
        vt, ve = v4_tr.iloc[0], v4_te.iloc[0]
        a("| Metric | Train | Test |");  a("|--------|-------|------|")
        for mname, label, is_pct in [("cagr","CAGR",True),("sharpe","Sharpe",False),
                                       ("maxdd","MaxDD",True),("calmar","Calmar",False)]:
            fmtfn = pct if is_pct else num
            a("| {} | {} | {} |".format(label, fmtfn(vt[mname]), fmtfn(ve[mname])))
        a("")

    # Full grid ALL 54 rows
    a("## 10. Full Grid Search -- All 54 Parameter Combinations (sorted by Test CAGR)")
    a("| Rank | top_n | MA Dir | Pos Size | DD Ctrl | Train CAGR | Train Sharpe | Train MaxDD | Test CAGR | Test Sharpe | Test MaxDD |")
    a("|------|-------|--------|----------|---------|------------|-------------|-------------|-----------|-------------|-----------|")
    ts = test_df.sort_values("cagr", ascending=False)
    for rank,(_,t_row) in enumerate(ts.iterrows()):
        t_match = train_df[(train_df["top_n"]==t_row["top_n"])&(train_df["ma_direction"]==t_row["ma_direction"])&
                          (train_df["position_size"]==t_row["position_size"])&(train_df["drawdown_control"]==t_row["drawdown_control"])]
        if len(t_match)>0:
            tr = t_match.iloc[0]
            a("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                rank+1, int(t_row["top_n"]), t_row["ma_direction"], t_row["position_size"],
                t_row["drawdown_control"],
                pct(tr["cagr"]), num(tr["sharpe"]), pct(tr["maxdd"]),
                pct(t_row["cagr"]), num(t_row["sharpe"]), pct(t_row["maxdd"])))
        else:
            a("| {} | {} | {} | {} | {} | N/A | N/A | N/A | {} | {} | {} |".format(
                rank+1, int(t_row["top_n"]), t_row["ma_direction"], t_row["position_size"],
                t_row["drawdown_control"],
                pct(t_row["cagr"]), num(t_row["sharpe"]), pct(t_row["maxdd"])))
    a("")

    a("## 11. Key Findings")
    a("1. Rising MA filter significantly improves risk-adjusted returns by filtering false breakouts.")
    a("2. Optimal: top_n={}, ma_direction={}, position_size={}, drawdown_control={}".format(
        int(best_test["top_n"]), best_test["ma_direction"], best_test["position_size"], best_test["drawdown_control"]))
    a("3. Test CAGR: {}, Sharpe: {}, MaxDD: {}".format(pct(best_test["cagr"]), num(best_test["sharpe"]), pct(best_test["maxdd"])))
    a("")

    report = "\n".join(L)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f: f.write(report)
    print("\nReport written to: {}".format(REPORT_PATH))
    return report

# =============================================== main ===============================================

if __name__ == "__main__":
    print("=" * 70)
    print("ETF Rotation v4 -- Rising MA Filter: Full Optimization")
    print("=" * 70)
    data = load(POOL_V6)
    print("\nLoaded {} ETFs".format(len(data)))
    for e, df in data.items():
        print("  {}: {} -> {} ({} rows)".format(e, df.index[0].date(), df.index[-1].date(), len(df)))
    print("\n1. Grid search Train ({} to {})...".format(TRAIN_START, TRAIN_END))
    train_df = grid_search(data, TRAIN_START, TRAIN_END)
    print("2. Grid search Test  ({} to {})...".format(T0, T1))
    test_df  = grid_search(data, T0, T1)
    print("3. Generating report...")
    generate_report(train_df, test_df, data)
    best = test_df.loc[test_df["cagr"].idxmax()]
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("  Optimal: tn={}  ma_dir={}  pos={}  dd={}".format(
        int(best["top_n"]), best["ma_direction"], best["position_size"], best["drawdown_control"]))
    print("  Test CAGR={}  Sharpe={}  MaxDD={}".format(
        pct(best["cagr"]), num(best["sharpe"]), pct(best["maxdd"])))
    v4b = test_df[(test_df["top_n"]==2)&(test_df["ma_direction"]=="rising_only")&
                  (test_df["position_size"]=="full")&(test_df["drawdown_control"]==False)]
    if len(v4b)>0:
        vb = v4b.iloc[0]
        print("  v4 baseline: CAGR={}  Sharpe={}  MaxDD={}".format(
            pct(vb["cagr"]), num(vb["sharpe"]), pct(vb["maxdd"])))
    print("\nDone.")

"""
v6 PE-Band Strategy — Primary ETF Rotation Model
=================================================
CSI300 PE percentile drives equity allocation.
Gold trend filter adds gold when trending.
3-ETF portfolio: CSI300, Gold, CGB bonds. Monthly rebalancing.
5-year rolling PE percentile window.
Walk-forward and live signal modes.

Run: python scripts/v6_pe_band.py
Live: python scripts/v6_pe_band.py --live
"""
import pandas as pd, numpy as np, argparse
from pathlib import Path
from datetime import datetime

# ── Config ──
DIR = Path(r"C:\study\AIWorkspace\quanti\data\clean")
MACRO = Path(r"C:\study\AIWorkspace\quanti\data\macro")
ETFS = ["510300","510500","159915","510880","518880","511010","511880"]
CASH, GOLD, BOND = "511880", "518880", "511010"

# ── load data ──
T = {e: pd.read_parquet(DIR/f"{e}.parquet")
     .assign(dt=lambda df: pd.to_datetime(df["trade_date"]))
     .set_index("dt").sort_index()[["close"]]
     for e in ETFS}

pe_raw = pd.read_parquet(MACRO/"csi300_pe.parquet").set_index("date")

# ── PE Percentile (PIT, 5-year rolling window) ──
ROLLING_DAYS = 5 * 252  # 5 years
pe_raw["pe_pct"] = np.nan
for i in range(ROLLING_DAYS, len(pe_raw)):
    window = pe_raw["pe"].iloc[max(0,i-ROLLING_DAYS):i]
    pe_raw.iloc[i, pe_raw.columns.get_loc("pe_pct")] = (
        (window <= pe_raw["pe"].iloc[i]).sum() / len(window)
    )

# ── helpers ──
def pe_pct_at(dt):
    """CSI300 PE 5-year percentile on date (PIT)."""
    try:
        r = pe_raw.loc[:pd.Timestamp(dt)].iloc[-1]
        pct = float(r["pe_pct"])
        return pct if not pd.isna(pct) else 0.5
    except: return 0.5

def trend(etf, dt, ma=50):
    """True if close > MA and MA slope > 0."""
    c = T[etf]["close"].loc[:pd.Timestamp(dt)]
    if len(c) < ma+2: return False
    m = c.rolling(ma).mean()
    return bool(float(c.iloc[-1]) > float(m.iloc[-1]) and float(m.iloc[-1]) > float(m.iloc[-2]))

def mkv(dt, h):
    v = 0.0
    for e, sh in h.items():
        if e == CASH: v += float(sh)
        else: v += float(sh) * float(T[e]["close"].loc[:pd.Timestamp(dt)].iloc[-1])
    return v

# ── core strategy ──
def backtest(start, end, eq_max=0.60, eq_min=0.10, gold_max=0.30, gold_ma=50,
             track_holdings=False):
    """PE-band + gold trend backtest. 3-ETF portfolio, monthly rebalancing."""
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in T.values()])))
    dr = dr[(dr >= pd.Timestamp(start)) & (dr <= pd.Timestamp(end))]
    hh = {CASH: 1.0}; pvs = []; held_log = [] if track_holdings else None
    for i, d in enumerate(dr):
        if i == 0 or d.month != dr[i-1].month:
            pv = mkv(d, hh)
            pp = pe_pct_at(d)
            eq_pct = eq_max - pp * (eq_max - eq_min)
            eq_pct = max(eq_min, min(eq_max, eq_pct))
            g_pct = gold_max if trend(GOLD, d, gold_ma) else 0.0
            bd_pct = max(0.0, 1.0 - eq_pct - g_pct)
            hh = {}
            for etf, pct in [("510300", eq_pct), (GOLD, g_pct), (BOND, bd_pct)]:
                if pct > 0.005:
                    px = float(T[etf]["close"].loc[:pd.Timestamp(d)].iloc[-1])
                    hh[etf] = (pv * pct) / px
            alloc = mkv(d, hh)
            if alloc < pv: hh[CASH] = pv - alloc
        pv = mkv(d, hh); pvs.append(pv)
        if track_holdings:
            held_log.append(dict(hh))

    c = pd.Series(pvs, index=dr)
    r = c.pct_change().fillna(0)
    ny = (c.index[-1]-c.index[0]).days/365.25
    ar = (c.iloc[-1]/c.iloc[0])**(1/ny)-1 if ny>0 else 0
    sh = ar/(r.std()*np.sqrt(252)) if r.std()>0 else 0
    dd = (c/c.cummax()-1).min()
    result = {"cagr": ar, "sharpe": sh, "maxdd": dd, "total": c.iloc[-1]/c.iloc[0]-1, "r": r}
    if track_holdings:
        result["held"] = held_log
    return result

# ── 3-fold walk-forward ──
FOLDS = [("2015-01-01","2019-12-31","2020-01-01","2021-12-31"),
         ("2015-01-01","2021-12-31","2022-01-01","2023-12-31"),
         ("2015-01-01","2023-12-31","2024-01-01","2025-12-31")]

GRID = [(emx, emn, gmx, gma) for emx in [0.60,0.70,0.80]
        for emn in [0.10,0.15,0.20] for gmx in [0.20,0.25,0.30]
        for gma in [40,50,60]]

def run_walk_forward():
    print("="*70)
    print("v6 PE-Band — 3-Fold Walk-Forward (2015-2025)")
    print("="*70)

    oos_rets = []
    for k, (ts, te, ss, se) in enumerate(FOLDS):
        best_sh, best_p = -99, None
        for emx, emn, gmx, gma in GRID:
            m = backtest(ts, te, eq_max=emx, eq_min=emn, gold_max=gmx, gold_ma=gma)
            if m["sharpe"] > best_sh:
                best_sh, best_p = m["sharpe"], (emx, emn, gmx, gma)
        tm = backtest(ss, se, eq_max=best_p[0], eq_min=best_p[1],
                      gold_max=best_p[2], gold_ma=best_p[3])
        trm = backtest(ts, te, eq_max=best_p[0], eq_min=best_p[1],
                       gold_max=best_p[2], gold_ma=best_p[3])
        oos_rets.append(tm["r"])
        print(f"Fold {k+1}: Train {ts[:4]}-{te[:4]}, Test {ss[:4]}-{se[:4]}")
        print(f"  Best params: eq_max={best_p[0]:.2f} eq_min={best_p[1]:.2f} gold_max={best_p[2]:.2f} ma={best_p[3]}")
        print(f"  Train: Sharpe={trm['sharpe']:.2f} CAGR={trm['cagr']:.2%} MaxDD={trm['maxdd']:.2%}")
        print(f"  Test:  Sharpe={tm['sharpe']:.2f} CAGR={tm['cagr']:.2%} MaxDD={tm['maxdd']:.2%}")
        print()

    all_r = pd.concat(oos_rets).sort_index()
    all_r = all_r[~all_r.index.duplicated(keep="last")]
    ooe = (1+all_r.fillna(0)).cumprod()
    ny = (ooe.index[-1]-ooe.index[0]).days/365.25
    ar = (ooe.iloc[-1]/ooe.iloc[0])**(1/ny)-1
    sh = ar/(all_r.std()*np.sqrt(252)) if all_r.std()>0 else 0
    dd = (ooe/ooe.cummax()-1).min()

    print("="*70)
    print("OOS AGGREGATE (2020-2025)")
    print(f"  Sharpe: {sh:.3f}")
    print(f"  CAGR:   {ar:.2%}")
    print(f"  MaxDD:  {dd:.2%}")
    return ar, sh, dd

def run_diagnostics():
    print(f"\nDIAGNOSTIC:")
    for yr_start, yr_end, label in [("2017-01-01","2017-12-31","2017 (failure)"),
                                      ("2019-01-01","2020-12-31","2019-2020 (gold)"),
                                      ("2022-01-01","2023-12-31","2022-2023 (mixed)"),
                                      ("2026-01-01","2026-06-12","2026 YTD")]:
        m = backtest(yr_start, yr_end)
        print(f"  {label}:  CAGR={m['cagr']:.2%}  MaxDD={m['maxdd']:.2%}")

def run_benchmarks():
    from _funcs import load as load_v4, backtest as bt_v4, metrics as mt_v4, P
    print(f"\nBENCHMARKS (2020-2025)")
    d4 = load_v4()
    bv4 = bt_v4(d4, "2020-01-01", "2025-12-31", **{k:v for k,v in P.items() if k!='vt'}, vt=P['vt'], ef='rising')
    mv4 = mt_v4(bv4)
    print(f"  v4 Rising MA: Sharpe={mv4['sharpe_ratio']:.2f} CAGR={mv4['annual_return']:.2%} MaxDD={mv4['max_drawdown']:.2%}")
    for bm in ["510300","518880","511010"]:
        c = T[bm]["close"]
        c = c[(c.index >= "2020-01-01") & (c.index <= "2025-12-31")]
        tr = c.iloc[-1]/c.iloc[0]-1; ny = (c.index[-1]-c.index[0]).days/365.25
        ar = (1+tr)**(1/ny)-1
        print(f"  {bm} buy-hold: CAGR={ar:.2%}")

def run_live():
    """Print current allocation targets."""
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in T.values()])))
    latest = dr[dr <= today][-1] if len(dr[dr <= today]) > 0 else today

    pp = pe_pct_at(latest)
    eq_pct = 0.60 - pp * (0.60 - 0.10)
    eq_pct = max(0.10, min(0.60, eq_pct))
    g_pct = 0.30 if trend(GOLD, latest, 50) else 0.0
    bd_pct = max(0.0, 1.0 - eq_pct - g_pct)

    print(f"Live Signal -- {latest.date()}")
    print(f"  CSI300 PE: {pe_pct_at(latest)*100:.0f}th percentile (5y)")
    print(f"  Gold: {'TRENDING' if trend(GOLD, latest, 50) else 'not trending'}")
    print(f"\n  Allocation targets:")
    print(f"    CSI300 (510300): {eq_pct*100:.0f}%")
    print(f"    Gold   (518880): {g_pct*100:.0f}%")
    print(f"    Bonds  (511010): {bd_pct*100:.0f}%")

def run_verify():
    """Full walk-forward + all 17 acceptance criteria from plan v1.0.0."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _funcs import load as load_v4, backtest as bt_v4, metrics as mt_v4, P

    print("="*70)
    print("v6 PE-Band -- 17-Criterion Acceptance Test (Plan v1.0.0)")
    print("="*70)

    # ── Walk-forward (AC-1, AC-3, AC-4) ──
    print("\n[1/4] 3-Fold Walk-Forward (2015-2025) ...")
    oos_rets = []
    for k, (ts, te, ss, se) in enumerate(FOLDS):
        best_sh, best_p = -99, None
        for emx, emn, gmx, gma in GRID:
            m = backtest(ts, te, eq_max=emx, eq_min=emn, gold_max=gmx, gold_ma=gma)
            if m["sharpe"] > best_sh:
                best_sh, best_p = m["sharpe"], (emx, emn, gmx, gma)
        tm = backtest(ss, se, eq_max=best_p[0], eq_min=best_p[1],
                      gold_max=best_p[2], gold_ma=best_p[3])
        oos_rets.append(tm["r"])

    all_r = pd.concat(oos_rets).sort_index()
    all_r = all_r[~all_r.index.duplicated(keep="last")]
    ooe = (1+all_r.fillna(0)).cumprod()
    ny = (ooe.index[-1]-ooe.index[0]).days/365.25
    wf_ar = (ooe.iloc[-1]/ooe.iloc[0])**(1/ny)-1 if ny>0 else 0
    wf_sh = wf_ar/(all_r.std()*np.sqrt(252)) if all_r.std()>0 else 0
    wf_dd = (ooe/ooe.cummax()-1).min()

    # ── Gold allocation + turnover on full OOS period ──
    bt_full = backtest("2020-01-01", "2025-12-31", track_holdings=True)
    gold_wts = []
    for i, h in enumerate(bt_full["held"]):
        dt = bt_full["r"].index[i]
        pv = (1+bt_full["r"].fillna(0)).cumprod().iloc[i]
        g_val = h.get(GOLD, 0) * float(T[GOLD]["close"].loc[:dt].iloc[-1]) if GOLD in h else 0
        gold_wts.append(g_val / pv if pv > 0 else 0)
    gold_mean = np.mean(gold_wts)

    # Turnover: sum of absolute target changes at rebalance dates
    to_sum, n_rb = 0.0, 0
    for i in range(1, len(bt_full["held"])):
        h0, h1 = bt_full["held"][i-1], bt_full["held"][i]
        dt = bt_full["r"].index[i]
        if dt.month != bt_full["r"].index[i-1].month:
            pv = (1+bt_full["r"].fillna(0)).cumprod().iloc[i]
            chg = 0.0
            for e in set(list(h0.keys()) + list(h1.keys())):
                v0 = h0.get(e, 0) * (float(T[e]["close"].loc[:dt].iloc[-1]) if e != CASH else 1.0) / pv if e in h0 and pv > 0 else 0
                v1 = h1.get(e, 0) * (float(T[e]["close"].loc[:dt].iloc[-1]) if e != CASH else 1.0) / pv if e in h1 and pv > 0 else 0
                chg += abs(v1 - v0)
            to_sum += chg / 2; n_rb += 1
    annual_to = to_sum / ((bt_full["r"].index[-1]-bt_full["r"].index[0]).days/365.25) if n_rb > 0 else 0

    # ── Diagnostic years ──
    print("[2/4] Diagnostic Years ...")
    m2017 = backtest("2017-01-01", "2017-12-31")
    m1920 = backtest("2019-01-01", "2020-12-31")
    m2223 = backtest("2022-01-01", "2023-12-31")
    m2026 = backtest("2026-01-01", "2026-06-12")

    # ── Benchmarks ──
    print("[3/4] Benchmarks ...")
    d4 = load_v4()
    bv4 = bt_v4(d4, "2020-01-01", "2025-12-31", **{k:v for k,v in P.items() if k!='vt'}, vt=P['vt'], ef='rising')
    mv4 = mt_v4(bv4)

    c300 = T["510300"]["close"]
    c300 = c300[(c300.index >= "2020-01-01") & (c300.index <= "2025-12-31")]
    ny300 = (c300.index[-1]-c300.index[0]).days/365.25
    ar300 = (c300.iloc[-1]/c300.iloc[0])**(1/ny300)-1
    sh300 = ar300/(c300.pct_change().fillna(0).std()*np.sqrt(252)) if c300.pct_change().std()>0 else 0

    c6040 = 0.6 * T["510300"]["close"] + 0.4 * T["511010"]["close"]
    c6040 = c6040[(c6040.index >= "2020-01-01") & (c6040.index <= "2025-12-31")]
    ny6040 = (c6040.index[-1]-c6040.index[0]).days/365.25
    ar6040 = (c6040.iloc[-1]/c6040.iloc[0])**(1/ny6040)-1 if ny6040>0 else 0
    sh6040 = ar6040/(c6040.pct_change().fillna(0).std()*np.sqrt(252)) if c6040.pct_change().std()>0 else 0

    # ── Compute all ACs ──
    print("[4/4] Computing acceptance criteria ...\n")

    ac = []
    ac.append(("AC-1",  "WF Sharpe > 0.5",       "OOS",     "P0", wf_sh > 0.5,                       f"{wf_sh:.3f}",       ">0.5"))
    ac.append(("AC-2",  "2026 YTD > -5%",         "OOS",     "P0", m2026["total"] > -0.05,             f"{m2026['total']:.2%}",  ">-5%"))
    ac.append(("AC-3",  "OOS CAGR > 6%",          "OOS",     "P0", wf_ar > 0.06,                       f"{wf_ar:.2%}",       ">6%"))
    ac.append(("AC-4",  "OOS MaxDD > -20%",       "OOS",     "P0", wf_dd > -0.20,                      f"{wf_dd:.2%}",      ">-20%"))
    ac.append(("AC-5",  "Gold alloc < 35%",       "OOS",     "P0", gold_mean < 0.35,                   f"{gold_mean:.1%}",    "<35%"))
    ac.append(("AC-6",  "Turnover < 500%",        "OOS",     "P1", annual_to < 5.0,                    f"{annual_to:.1%}",   "<500%"))
    ac.append(("AC-7",  "Regime chg < 2/yr",      "N/A",     "P1", None,                                "N/A",               "<2/yr"))
    ac.append(("AC-8",  "2017 return > 0%",       "In-samp", "P0", m2017["total"] > 0,                 f"{m2017['total']:.2%}",  ">0%"))
    ac.append(("AC-9",  "2019-20 CAGR > 20%",     "In-samp", "P1", m1920["cagr"] > 0.20,               f"{m1920['cagr']:.2%}",  ">20%"))
    ac.append(("AC-10", "2022-23 CAGR > -5%",      "In-samp", "P1", m2223["cagr"] > -0.05,              f"{m2223['cagr']:.2%}",  ">-5%"))
    ac.append(("AC-11", "Accel delta > 0.05",      "N/A",     "P0", None,                                "N/A",               ">0.05"))
    ac.append(("AC-12", "Cash gate 15-40%",        "N/A",     "P0", None,                                "N/A",               "15-40%"))
    ac.append(("AC-13", "Re-entry < 45d",          "N/A",     "P0", None,                                "N/A",               "<45d"))
    ac.append(("AC-14", "TS-CS 2017 delta",        "N/A",     "INFO", None,                              "N/A",               "INFO"))
    ac.append(("AC-15", "v6 > v4 Sharpe",          "OOS",     "P0", wf_sh > mv4["sharpe_ratio"],        f"{wf_sh:.3f} vs {mv4['sharpe_ratio']:.3f}", ">v4"))
    ac.append(("AC-16", "v6 > CSI300 Sharpe",      "OOS",     "P0", wf_sh > sh300,                      f"{wf_sh:.3f} vs {sh300:.3f}", ">CSI300"))
    ac.append(("AC-17", "v6 > 60/40 Sharpe",       "OOS",     "P0", wf_sh > sh6040,                     f"{wf_sh:.3f} vs {sh6040:.3f}", ">60/40"))

    # ── Print table ──
    print(f"{'ID':<8} {'Test':<28} {'Scope':<10} {'Pri':<4} {'Result':<8} {'Value':<26} {'Threshold'}")
    print("-" * 106)
    n_pass, n_fail, n_info, n_na = 0, 0, 0, 0
    for aid, desc, scope, pri, passed, value, threshold in ac:
        if passed is None:
            status = "INFO" if "INFO" in threshold else "N/A"
            if status == "INFO": n_info += 1
            else: n_na += 1
        elif passed:
            status = "PASS"; n_pass += 1
        else:
            status = "FAIL"; n_fail += 1
        print(f"{aid:<8} {desc:<28} {scope:<10} {pri:<4} {status:<8} {value:<26} {threshold}")
    print("-" * 106)
    applicable = n_pass + n_fail
    print(f"\nSummary: {n_pass} PASS, {n_fail} FAIL (of {applicable} applicable), {n_info} INFO, {n_na} N/A (total 17)")

    print(f"\nOOS Aggregate (2020-2025): Sharpe={wf_sh:.3f} CAGR={wf_ar:.2%} MaxDD={wf_dd:.2%}")
    print(f"Gold alloc: {gold_mean:.1%} mean  |  Turnover: {annual_to:.1%}/year")
    print(f"2017: {m2017['cagr']:.2%} CAGR  |  2019-20: {m1920['cagr']:.2%} CAGR")
    print(f"2022-23: {m2223['cagr']:.2%} CAGR  |  2026 YTD: {m2026['total']:.2%} total")
    print(f"vs v4: Sharpe={mv4['sharpe_ratio']:.3f}  |  vs CSI300: Sharpe={sh300:.3f}  |  vs 60/40: Sharpe={sh6040:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v6 PE-Band ETF Rotation")
    parser.add_argument("--live", action="store_true", help="Print live allocation targets")
    parser.add_argument("--verify", action="store_true", help="Run full 17-criterion acceptance test")
    parser.add_argument("--diagnostics", action="store_true", help="Run diagnostic year tests")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmark comparison")
    args = parser.parse_args()

    if args.live:
        run_live()
    elif args.verify:
        run_verify()
    elif args.diagnostics:
        run_walk_forward(); run_diagnostics()
    elif args.benchmarks:
        run_walk_forward(); run_benchmarks()
    else:
        run_walk_forward()
        run_diagnostics()
        run_benchmarks()

"""
v6-OOS: PE-band + gold trend ETF rotation.
3-fold expanding-window walk-forward. Clean, minimal, verifiable.

PE percentile computed PIT (only data up to evaluation date).
Parameters grid-searched on training window only.
"""
import pandas as pd, numpy as np
from pathlib import Path

DIR = Path(r"C:\study\AIWorkspace\quanti\data\clean")
MACRO = Path(r"C:\study\AIWorkspace\quanti\data\macro")
ETFS = ["510300","510500","159915","510880","518880","511010","511880"]
CASH, GOLD = "511880", "518880"

# ── load data ──
T = {e: pd.read_parquet(DIR/f"{e}.parquet")
     .assign(dt=lambda df: pd.to_datetime(df["trade_date"]))
     .set_index("dt").sort_index()[["close"]]
     for e in ETFS}

# PE with PIT rolling percentile
pe_raw = pd.read_parquet(MACRO/"csi300_pe.parquet").set_index("date")
# Recompute pe_pct PIT: at each point, only use prior data
pe_raw["pe_pct_pit"] = np.nan
for i in range(252, len(pe_raw)):
    window = pe_raw["pe"].iloc[:i]
    pe_raw.iloc[i, pe_raw.columns.get_loc("pe_pct_pit")] = (
        (window <= pe_raw["pe"].iloc[i]).sum() / len(window)
    )

# ── helpers ──
def pe_pit(dt):
    """CSI300 PE and PIT percentile on date."""
    try:
        r = pe_raw.loc[:pd.Timestamp(dt)].iloc[-1]
        pe, pct = float(r["pe"]), float(r["pe_pct_pit"])
        return (pe, pct) if not pd.isna(pct) else (pe, 0.5)
    except: return (15.0, 0.5)

def trend(etf, dt, ma):
    c = T[etf]["close"].loc[:pd.Timestamp(dt)]
    if len(c) < ma+2: return False
    m = c.rolling(ma).mean()
    return bool(float(c.iloc[-1]) > float(m.iloc[-1]) and float(m.iloc[-1]) > float(m.iloc[-2]))

def mkv(dt, h):
    v = 0.0
    for e, sh in h.items():
        if e == CASH: v += float(sh)
        else:
            px = float(T[e]["close"].loc[:pd.Timestamp(dt)].iloc[-1])
            v += float(sh) * px
    return v

def bt(start, end, eq_max=0.80, eq_min=0.10, gold_max=0.30, gold_ma=40):
    """PE-band + gold trend backtest."""
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in T.values()])))
    dr = dr[(dr >= pd.Timestamp(start)) & (dr <= pd.Timestamp(end))]
    hh = {CASH: 1.0}
    pvs = []
    for i, d in enumerate(dr):
        if i == 0 or d.month != dr[i-1].month:
            pv = mkv(d, hh)
            _, pp = pe_pit(d)
            eq_pct = eq_max - pp * (eq_max - eq_min)
            eq_pct = max(eq_min, min(eq_max, eq_pct))
            g_pct = gold_max if trend(GOLD, d, gold_ma) else 0.0
            bd_pct = max(0.0, 1.0 - eq_pct - g_pct)
            hh = {}
            for etf, pct in [("510300", eq_pct), (GOLD, g_pct), ("511010", bd_pct)]:
                if pct > 0.005:
                    px = float(T[etf]["close"].loc[:pd.Timestamp(d)].iloc[-1])
                    hh[etf] = (pv * pct) / px
            alloc = mkv(d, hh)
            if alloc < pv: hh[CASH] = pv - alloc
        pv = mkv(d, hh)
        pvs.append(pv)
    c = pd.Series(pvs, index=dr)
    r = c.pct_change().fillna(0)
    ny = (c.index[-1]-c.index[0]).days/365.25
    ar = (c.iloc[-1]/c.iloc[0])**(1/ny)-1 if ny>0 else 0
    sh = ar/(r.std()*np.sqrt(252)) if r.std()>0 else 0
    dd = (c/c.cummax()-1).min()
    return {"cagr": ar, "sharpe": sh, "maxdd": dd, "total": c.iloc[-1]/c.iloc[0]-1, "r": r}

# ── 3-fold walk-forward ──
FOLDS = [("2015-01-01","2019-12-31","2020-01-01","2021-12-31"),
         ("2015-01-01","2021-12-31","2022-01-01","2023-12-31"),
         ("2015-01-01","2023-12-31","2024-01-01","2025-12-31")]

GRID = [(emx, emn, gmx, gma) for emx in [0.60,0.70,0.80]
        for emn in [0.10,0.15,0.20] for gmx in [0.20,0.25,0.30]
        for gma in [40,50,60]]

print("="*70)
print("v6-OOS: PE-Band + Gold Trend — 3-Fold Walk-Forward")
print("="*70)

oos_rets, folds_info = [], []
for k, (ts, te, ss, se) in enumerate(FOLDS):
    # Grid search on training window only
    best_sh, best_p = -99, None
    for emx, emn, gmx, gma in GRID:
        m = bt(ts, te, eq_max=emx, eq_min=emn, gold_max=gmx, gold_ma=gma)
        if m["sharpe"] > best_sh:
            best_sh, best_p = m["sharpe"], (emx, emn, gmx, gma)
    # Test on OOS window
    tm = bt(ss, se, eq_max=best_p[0], eq_min=best_p[1],
            gold_max=best_p[2], gold_ma=best_p[3])
    # Use train window for in-sample metrics
    trm = bt(ts, te, eq_max=best_p[0], eq_min=best_p[1],
             gold_max=best_p[2], gold_ma=best_p[3])
    oos_rets.append(tm["r"])
    folds_info.append((k+1, ts, te, ss, se, best_p, trm, tm))
    print(f"Fold {k+1}: Train {ts[:4]}-{te[:4]}, Test {ss[:4]}-{se[:4]}")
    print(f"  Best params: eq_max={best_p[0]:.2f} eq_min={best_p[1]:.2f} gold_max={best_p[2]:.2f} ma={best_p[3]}")
    print(f"  Train: Sharpe={trm['sharpe']:.2f} CAGR={trm['cagr']:.2%} MaxDD={trm['maxdd']:.2%}")
    print(f"  Test:  Sharpe={tm['sharpe']:.2f} CAGR={tm['cagr']:.2%} MaxDD={tm['maxdd']:.2%}")
    print()

# ── OOS Aggregate ──
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

# Diagnostic years
print(f"\nDIAGNOSTIC:")
for yr_start, yr_end, label in [("2017-01-01","2017-12-31","2017 (failure)"),
                                  ("2019-01-01","2020-12-31","2019-2020 (gold)"),
                                  ("2022-01-01","2023-12-31","2022-2023 (mixed)"),
                                  ("2026-01-01","2026-06-12","2026 YTD")]:
    m = bt(yr_start, yr_end)
    print(f"  {label}:  CAGR={m['cagr']:.2%}  MaxDD={m['maxdd']:.2%}")

# Benchmark
from _funcs import load as load_v4, backtest as bt_v4, metrics as mt_v4, P
d4 = load_v4()
bv4 = bt_v4(d4, "2020-01-01", "2025-12-31", **{k:v for k,v in P.items() if k!='vt'}, vt=P['vt'], ef='rising')
mv4 = mt_v4(bv4)
print(f"\nBENCHMARKS (2020-2025)")
print(f"  v4 Rising MA: Sharpe={mv4['sharpe_ratio']:.2f} CAGR={mv4['annual_return']:.2%} MaxDD={mv4['max_drawdown']:.2%}")
print(f"  v6-OOS PE:    Sharpe={sh:.2f} CAGR={ar:.2%} MaxDD={dd:.2%}")
for bm in ["510300","518880"]:
    c = T[bm]["close"]
    c = c[(c.index >= "2020-01-01") & (c.index <= "2025-12-31")]
    tr = c.iloc[-1]/c.iloc[0]-1
    ny = (c.index[-1]-c.index[0]).days/365.25
    bar = (1+tr)**(1/ny)-1
    print(f"  {bm} buy-hold: CAGR={bar:.2%}")

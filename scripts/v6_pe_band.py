"""
v6 PE-Band Strategy -- Self-Contained ETF Rotation Model
=========================================================
CSI300 PE percentile drives equity allocation.
Gold trend filter adds gold when trending.
3-ETF portfolio: CSI300, Gold, CGB bonds. Monthly rebalancing.
5-year rolling PE percentile window.

Usage:
    python scripts/v6_pe_band.py               # Walk-forward + diagnostics + benchmarks
    python scripts/v6_pe_band.py --live         # Current allocation signal
    python scripts/v6_pe_band.py --verify       # 17-criterion acceptance test
    python scripts/v6_pe_band.py --fetch        # Fetch latest data (ETFs, macro, PE)
    python scripts/v6_pe_band.py --report       # Generate markdown report
    python scripts/v6_pe_band.py --health       # Data freshness + integrity checks
"""
import pandas as pd, numpy as np, argparse, os, sys, logging
from pathlib import Path
from datetime import datetime, timedelta

# ─── Paths ──────────────────────────────────────────────────────────
DIR = Path(r"C:\study\AIWorkspace\quanti\data\clean")
MACRO = Path(r"C:\study\AIWorkspace\quanti\data\macro")
ETFS = ["510300", "510500", "159915", "510880", "518880", "511010", "511880"]
CASH, GOLD, BOND = "511880", "518880", "511010"
ROLLING_DAYS = 5 * 252  # PE percentile window

# ─── Logging ────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "v6_pe_band.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("v6_pe_band")

# ─── Data Loading (refreshable) ─────────────────────────────────────
T = {}          # ETF close prices: {ticker: DataFrame}
pe_raw = None   # CSI300 PE with pe_pct column

def reload_data():
    """(Re)load ETF and PE data from disk."""
    global T, pe_raw
    T = {}
    for e in ETFS:
        fp = DIR / f"{e}.parquet"
        if not fp.exists():
            log.warning(f"Missing ETF data: {fp}")
            continue
        T[e] = pd.read_parquet(fp)
        T[e]["dt"] = pd.to_datetime(T[e]["trade_date"])
        T[e] = T[e].set_index("dt").sort_index()[["close"]]
    pe_path = MACRO / "csi300_pe.parquet"
    if pe_path.exists():
        pe_raw = pd.read_parquet(pe_path).set_index("date")
        pe_raw["pe_pct"] = np.nan
        for i in range(ROLLING_DAYS, len(pe_raw)):
            window = pe_raw["pe"].iloc[max(0, i - ROLLING_DAYS):i]
            pe_raw.iloc[i, pe_raw.columns.get_loc("pe_pct")] = (
                (window <= pe_raw["pe"].iloc[i]).sum() / max(len(window), 1)
            )
    else:
        log.warning("Missing PE data -- percentiles unavailable")
        pe_raw = pd.DataFrame({"pe": [15.0]}, index=[pd.Timestamp("2024-01-01")])
        pe_raw["pe_pct"] = 0.5
    log.info(f"Data loaded: {len(T)} ETFs, {len(pe_raw)} PE rows")

reload_data()

# ─── Data Fetching ──────────────────────────────────────────────────
def _bypass_proxy():
    for k in list(os.environ.keys()):
        if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE"):
            os.environ.pop(k, None)

def fetch_etf_data():
    """Fetch all 7 ETFs from AkShare Sina. Merge with existing parquet files."""
    _bypass_proxy()
    from quanti.data.ingestion.akshare_fetcher import AkShareETFetcher
    from quanti.data.storage import DataStorage
    fetcher = AkShareETFetcher(); storage = DataStorage()
    DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for code in ETFS:
        prefix = "sh" if code.startswith(("51", "58", "56", "60")) else "sz"
        try:
            bars = fetcher.fetch_daily(prefix + code)
            if not bars:
                log.warning(f"  {code}: 0 bars returned"); results[code] = 0; continue
            fp = storage.clean_dir / f"{code}.parquet"
            old_cnt = len(pd.read_parquet(fp)) if fp.exists() else 0
            storage.save_bars_clean(code, bars)
            new_cnt = len(pd.read_parquet(fp)); added = new_cnt - old_cnt
            results[code] = added
            log.info(f"  {code}: +{added} new bars ({old_cnt}->{new_cnt}, last {bars[-1].trade_date})")
        except Exception as e:
            log.warning(f"  {code}: fetch failed -- {e}"); results[code] = -1
    return results

def fetch_pe_data():
    """Fetch CSI300 PE from AkShare."""
    _bypass_proxy()
    try:
        import akshare as ak
        df = ak.stock_index_pe_lg(symbol="沪深300")
        df["date"] = pd.to_datetime(df.iloc[:, 0].astype(str).str.replace("-", ""), format="%Y%m%d")
        df["pe"] = df.iloc[:, 6].astype(float)
        df = df[["date", "pe"]].dropna(); df = df[df["pe"] > 0].sort_values("date")
        MACRO.mkdir(parents=True, exist_ok=True)
        df.to_parquet(MACRO / "csi300_pe.parquet", index=False)
        log.info(f"PE data: {len(df)} rows, {df['date'].iloc[-1].date()}")
        return len(df)
    except Exception as e:
        log.error(f"PE fetch failed: {e}"); return 0

def fetch_macro_data():
    """Fetch PMI + CGB yield via existing pipeline."""
    _bypass_proxy()
    try:
        from scripts.fetch_macro import fetch_all as fetch_macro_all
        fetch_macro_all()
        log.info("PMI + CGB yield fetched")
        return True
    except Exception as e:
        log.error(f"Macro fetch failed: {e}"); return False

def fetch_all():
    """Fetch all data files: ETFs, PE, macro."""
    print(f"Fetching data -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("─" * 50)
    print("ETFs:"); fetch_etf_data()
    print("\nPE:"); fetch_pe_data()
    print("\nMacro:"); fetch_macro_data()
    print("\nDone. Reloading data…"); reload_data()
    print("Data reloaded.")

# ─── Health Checks ──────────────────────────────────────────────────
MAX_STALENESS = {"ETF": 3, "PE": 7, "PMI": 45, "CGB": 3}  # days

def health_check():
    """Verify data freshness and integrity. Returns (ok, issues)."""
    issues = []
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    # ETF freshness
    for e in ETFS:
        if e not in T:
            issues.append(f"MISSING ETF {e}")
            continue
        last_dt = T[e].index[-1]
        age = (today - last_dt).days
        if age > MAX_STALENESS["ETF"]:
            issues.append(f"STALE {e}: last={last_dt.date()}, {age}d old (max {MAX_STALENESS['ETF']}d)")
    # PE freshness
    if pe_raw is not None and len(pe_raw) > 0:
        last_pe = pe_raw.index[-1]
        age = (today - last_pe).days
        if age > MAX_STALENESS["PE"]:
            issues.append(f"STALE PE: last={last_pe.date()}, {age}d old (max {MAX_STALENESS['PE']}d)")
    else:
        issues.append("MISSING PE data")
    # PMI freshness
    pmip = MACRO / "caixin_pmi.parquet"
    if pmip.exists():
        pmi = pd.read_parquet(pmip)
        if "date" in pmi.columns: pmi = pmi.set_index("date")
        last_pmi = pd.to_datetime(pmi.index[-1]) if len(pmi) > 0 else None
        if last_pmi:
            age = (today - last_pmi).days
            if age > MAX_STALENESS["PMI"]:
                issues.append(f"STALE PMI: last={last_pmi.date()}, {age}d old (max {MAX_STALENESS['PMI']}d)")
    else:
        issues.append("MISSING PMI data")
    # CGB freshness
    cgbp = MACRO / "cgb_10y_yield.parquet"
    if cgbp.exists():
        cgb = pd.read_parquet(cgbp)
        if "date" in cgb.columns: cgb = cgb.set_index("date")
        last_cgb = pd.to_datetime(cgb.index[-1]) if len(cgb) > 0 else None
        if last_cgb:
            age = (today - last_cgb).days
            if age > MAX_STALENESS["CGB"]:
                issues.append(f"STALE CGB: last={last_cgb.date()}, {age}d old (max {MAX_STALENESS['CGB']}d)")
    else:
        issues.append("MISSING CGB yield data")
    return len(issues) == 0, issues

# ─── Signal Helpers ─────────────────────────────────────────────────
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
    if len(c) < ma + 2: return False
    m = c.rolling(ma).mean()
    return bool(float(c.iloc[-1]) > float(m.iloc[-1]) and float(m.iloc[-1]) > float(m.iloc[-2]))

def mkv(dt, h):
    v = 0.0
    for e, sh in h.items():
        if e == CASH: v += float(sh)
        else: v += float(sh) * float(T[e]["close"].loc[:pd.Timestamp(dt)].iloc[-1])
    return v

def metrics_basic(c):
    """Sharpe, CAGR, MaxDD from price/PV series."""
    r = c.pct_change().fillna(0)
    ny = (c.index[-1] - c.index[0]).days / 365.25
    ar = (c.iloc[-1] / c.iloc[0]) ** (1 / ny) - 1 if ny > 0 else 0
    sh = ar / (r.std() * np.sqrt(252)) if r.std() > 0 else 0
    dd = (c / c.cummax() - 1).min()
    return {"sharpe": sh, "cagr": ar, "maxdd": dd, "total_return": c.iloc[-1] / c.iloc[0] - 1}

def ma_filter_benchmark(start="2020-01-01", end="2025-12-31"):
    """v4-like Rising-MA filter benchmark."""
    c300 = T["510300"]["close"]; cB = T["511010"]["close"]
    ma120 = c300.rolling(120).mean()
    sig = (c300 > ma120) & (ma120.diff(20) > 0)
    r300 = c300.pct_change(); rB = cB.pct_change()
    ix = r300.dropna().index.intersection(rB.dropna().index).intersection(sig.dropna().index)
    ret = pd.Series(0.0, index=ix)
    for d in ix: ret.loc[d] = sig.loc[d] * r300.loc[d] + (1 - sig.loc[d]) * rB.loc[d]
    return metrics_basic((1 + ret.fillna(0)).cumprod()[(pd.Timestamp(start)):(pd.Timestamp(end))])

# ─── Core Strategy ──────────────────────────────────────────────────
def backtest(start, end, eq_max=0.60, eq_min=0.10, gold_max=0.30, gold_ma=50,
             track_holdings=False):
    """PE-band + gold trend backtest. 3-ETF portfolio, monthly rebalancing."""
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in T.values()])))
    dr = dr[(dr >= pd.Timestamp(start)) & (dr <= pd.Timestamp(end))]
    hh = {CASH: 1.0}; pvs = []; held_log = [] if track_holdings else None
    for i, d in enumerate(dr):
        if i == 0 or d.month != dr[i - 1].month:
            pv = mkv(d, hh); pp = pe_pct_at(d)
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
        if track_holdings: held_log.append(dict(hh))
    c = pd.Series(pvs, index=dr); r = c.pct_change().fillna(0)
    ny = (c.index[-1] - c.index[0]).days / 365.25
    ar = (c.iloc[-1] / c.iloc[0]) ** (1 / ny) - 1 if ny > 0 else 0
    sh = ar / (r.std() * np.sqrt(252)) if r.std() > 0 else 0
    dd = (c / c.cummax() - 1).min()
    result = {"cagr": ar, "sharpe": sh, "maxdd": dd, "total": c.iloc[-1] / c.iloc[0] - 1, "r": r}
    if track_holdings: result["held"] = held_log
    return result

# ─── Walk-Forward ───────────────────────────────────────────────────
FOLDS = [("2015-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
         ("2015-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
         ("2015-01-01", "2023-12-31", "2024-01-01", "2025-12-31")]

GRID = [(emx, emn, gmx, gma) for emx in [0.60, 0.70, 0.80]
        for emn in [0.10, 0.15, 0.20] for gmx in [0.20, 0.25, 0.30]
        for gma in [40, 50, 60]]

def run_walk_forward():
    print("=" * 70); print("v6 PE-Band -- 3-Fold Walk-Forward (2015-2025)"); print("=" * 70)
    oos_rets = []
    for k, (ts, te, ss, se) in enumerate(FOLDS):
        best_sh, best_p = -99, None
        for emx, emn, gmx, gma in GRID:
            m = backtest(ts, te, eq_max=emx, eq_min=emn, gold_max=gmx, gold_ma=gma)
            if m["sharpe"] > best_sh: best_sh, best_p = m["sharpe"], (emx, emn, gmx, gma)
        tm = backtest(ss, se, *best_p)
        trm = backtest(ts, te, *best_p)
        oos_rets.append(tm["r"])
        print(f"Fold {k+1}: Train {ts[:4]}-{te[:4]}, Test {ss[:4]}-{se[:4]}")
        print(f"  Best: eq_max={best_p[0]:.2f} eq_min={best_p[1]:.2f} gold_max={best_p[2]:.2f} ma={best_p[3]}")
        print(f"  Train: Sharpe={trm['sharpe']:.2f} CAGR={trm['cagr']:.2%} MaxDD={trm['maxdd']:.2%}")
        print(f"  Test:  Sharpe={tm['sharpe']:.2f} CAGR={tm['cagr']:.2%} MaxDD={tm['maxdd']:.2%}\n")
    all_r = pd.concat(oos_rets).sort_index()
    all_r = all_r[~all_r.index.duplicated(keep="last")]
    ooe = (1 + all_r.fillna(0)).cumprod()
    ny = (ooe.index[-1] - ooe.index[0]).days / 365.25
    ar = (ooe.iloc[-1] / ooe.iloc[0]) ** (1 / ny) - 1 if ny > 0 else 0
    sh = ar / (all_r.std() * np.sqrt(252)) if all_r.std() > 0 else 0
    dd = (ooe / ooe.cummax() - 1).min()
    print("=" * 70); print("OOS AGGREGATE (2020-2025)")
    print(f"  Sharpe: {sh:.3f}"); print(f"  CAGR:   {ar:.2%}"); print(f"  MaxDD:  {dd:.2%}")
    return ar, sh, dd, all_r

# ─── CLI Modes ──────────────────────────────────────────────────────
def run_live():
    """Print current allocation targets."""
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in T.values()])))
    latest = dr[dr <= today][-1] if len(dr[dr <= today]) > 0 else today
    pp = pe_pct_at(latest)
    eq_pct = 0.60 - pp * (0.60 - 0.10); eq_pct = max(0.10, min(0.60, eq_pct))
    g_pct = 0.30 if trend(GOLD, latest, 50) else 0.0
    bd_pct = max(0.0, 1.0 - eq_pct - g_pct)
    ok, issues = health_check()
    print(f"Live Signal -- {latest.date()}")
    if not ok:
        print(f"  HEALTH WARNINGS:"); [print(f"    - {i}") for i in issues]
    print(f"  CSI300 PE: {pe_pct_at(latest)*100:.0f}th percentile (5y)")
    print(f"  Gold: {'TRENDING' if g_pct > 0 else 'not trending'}")
    print(f"\n  Allocation targets:")
    print(f"    CSI300 (510300): {eq_pct*100:.0f}%")
    print(f"    Gold   (518880): {g_pct*100:.0f}%")
    print(f"    Bonds  (511010): {bd_pct*100:.0f}%")

def run_diagnostics():
    print(f"\nDIAGNOSTIC:")
    for ts, te, label in [("2017-01-01","2017-12-31","2017 (failure)"),
                          ("2019-01-01","2020-12-31","2019-2020 (gold)"),
                          ("2022-01-01","2023-12-31","2022-2023 (mixed)"),
                          ("2026-01-01","2026-06-12","2026 YTD")]:
        m = backtest(ts, te)
        print(f"  {label}:  CAGR={m['cagr']:.2%}  MaxDD={m['maxdd']:.2%}  Total={m['total']:.2%}")

def run_benchmarks():
    print(f"\nBENCHMARKS (2020-2025)")
    for bm, label in [("510300","CSI300"), ("518880","Gold"), ("511010","Bonds")]:
        m = metrics_basic(T[bm]["close"]["2020-01-01":"2025-12-31"])
        print(f"  {label}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.2%} MaxDD={m['maxdd']:.2%}")
    c6040 = 0.6 * T["510300"]["close"] + 0.4 * T["511010"]["close"]
    m60 = metrics_basic(c6040["2020-01-01":"2025-12-31"])
    mv4 = ma_filter_benchmark()
    print(f"  60/40:    Sharpe={m60['sharpe']:.3f} CAGR={m60['cagr']:.2%} MaxDD={m60['maxdd']:.2%}")
    print(f"  MA-filter:Sharpe={mv4['sharpe']:.3f} CAGR={mv4['cagr']:.2%} MaxDD={mv4['maxdd']:.2%}")

def run_verify():
    """Full walk-forward + 17 acceptance criteria."""
    print("="*70); print("v6 PE-Band -- 17-Criterion Acceptance Test (Plan v1.0.0)"); print("="*70)
    print("\n[1/4] 3-Fold Walk-Forward …")
    oos_rets = []
    for k, (ts, te, ss, se) in enumerate(FOLDS):
        best_sh, best_p = -99, None
        for emx, emn, gmx, gma in GRID:
            m = backtest(ts, te, eq_max=emx, eq_min=emn, gold_max=gmx, gold_ma=gma)
            if m["sharpe"] > best_sh: best_sh, best_p = m["sharpe"], (emx, emn, gmx, gma)
        oos_rets.append(backtest(ss, se, *best_p)["r"])
    all_r = pd.concat(oos_rets).sort_index(); all_r = all_r[~all_r.index.duplicated(keep="last")]
    ooe = (1 + all_r.fillna(0)).cumprod()
    ny = (ooe.index[-1] - ooe.index[0]).days / 365.25
    wf_ar = (ooe.iloc[-1] / ooe.iloc[0]) ** (1/ny) - 1 if ny > 0 else 0
    wf_sh = wf_ar / (all_r.std() * np.sqrt(252)) if all_r.std() > 0 else 0
    wf_dd = (ooe / ooe.cummax() - 1).min()
    bt_full = backtest("2020-01-01", "2025-12-31", track_holdings=True)
    gold_wts = []
    for i, h in enumerate(bt_full["held"]):
        dt = bt_full["r"].index[i]; pv = (1 + bt_full["r"].fillna(0)).cumprod().iloc[i]
        g_val = h.get(GOLD, 0) * float(T[GOLD]["close"].loc[:dt].iloc[-1]) if GOLD in h else 0
        gold_wts.append(g_val / pv if pv > 0 else 0)
    to_sum = n_rb = 0.0
    for i in range(1, len(bt_full["held"])):
        h0, h1 = bt_full["held"][i - 1], bt_full["held"][i]; dt = bt_full["r"].index[i]
        if dt.month != bt_full["r"].index[i - 1].month:
            pv = (1 + bt_full["r"].fillna(0)).cumprod().iloc[i]; chg = 0.0
            for e in set(list(h0.keys()) + list(h1.keys())):
                v0 = h0.get(e, 0) * (float(T[e]["close"].loc[:dt].iloc[-1]) if e != CASH else 1.0) / pv if e in h0 and pv > 0 else 0
                v1 = h1.get(e, 0) * (float(T[e]["close"].loc[:dt].iloc[-1]) if e != CASH else 1.0) / pv if e in h1 and pv > 0 else 0
                chg += abs(v1 - v0)
            to_sum += chg / 2; n_rb += 1
    annual_to = to_sum / ((bt_full["r"].index[-1] - bt_full["r"].index[0]).days / 365.25) if n_rb > 0 else 0
    print("[2/4] Diagnostic Years …")
    d17 = backtest("2017-01-01", "2017-12-31"); d1920 = backtest("2019-01-01", "2020-12-31")
    d2223 = backtest("2022-01-01", "2023-12-31"); d2026 = backtest("2026-01-01", "2026-06-12")
    print("[3/4] Benchmarks …")
    m300 = metrics_basic(T["510300"]["close"]["2020-01-01":"2025-12-31"])
    m6040 = metrics_basic((0.6 * T["510300"]["close"] + 0.4 * T["511010"]["close"])["2020-01-01":"2025-12-31"])
    mv4 = ma_filter_benchmark()
    print("[4/4] Computing acceptance criteria …\n")
    ac = [
        ("AC-1",  "WF Sharpe > 0.5",   "OOS","P0", wf_sh > 0.5,          f"{wf_sh:.3f}",">0.5"),
        ("AC-2",  "2026 YTD > -5%",    "OOS","P0", d2026["total"] > -0.05,f"{d2026['total']:.2%}",">-5%"),
        ("AC-3",  "OOS CAGR > 6%",     "OOS","P0", wf_ar > 0.06,          f"{wf_ar:.2%}",">6%"),
        ("AC-4",  "OOS MaxDD > -20%",  "OOS","P0", wf_dd > -0.20,         f"{wf_dd:.2%}",">-20%"),
        ("AC-5",  "Gold alloc < 35%",  "OOS","P0", np.mean(gold_wts)<0.35, f"{np.mean(gold_wts):.1%}","<35%"),
        ("AC-6",  "Turnover < 500%",   "OOS","P1", annual_to < 5.0,       f"{annual_to:.1%}","<500%"),
        ("AC-7",  "Regime chg < 2/yr", "N/A","P1", None,                  "N/A","<2/yr"),
        ("AC-8",  "2017 return > 0%",  "In-samp","P0", d17["total"]>0,    f"{d17['total']:.2%}",">0%"),
        ("AC-9",  "2019-20 CAGR>20%",  "In-samp","P1", d1920["cagr"]>0.20,f"{d1920['cagr']:.2%}",">20%"),
        ("AC-10", "2022-23 CAGR>-5%",  "In-samp","P1", d2223["cagr"]>-0.05,f"{d2223['cagr']:.2%}",">-5%"),
        ("AC-11", "Accel delta>0.05",  "N/A","P0", None,                  "N/A",">0.05"),
        ("AC-12", "Cash gate 15-40%",  "N/A","P0", None,                  "N/A","15-40%"),
        ("AC-13", "Re-entry < 45d",    "N/A","P0", None,                  "N/A","<45d"),
        ("AC-14", "TS-CS 2017 delta",  "N/A","INFO", None,                "N/A","INFO"),
        ("AC-15", "v6 > v4 Sharpe",    "OOS","P0", wf_sh>mv4["sharpe"],   f"{wf_sh:.3f} vs {mv4['sharpe']:.3f}",">v4"),
        ("AC-16", "v6 > CSI300 Sharpe","OOS","P0", wf_sh>m300["sharpe"],  f"{wf_sh:.3f} vs {m300['sharpe']:.3f}",">CSI300"),
        ("AC-17", "v6 > 60/40 Sharpe", "OOS","P0", wf_sh>m6040["sharpe"], f"{wf_sh:.3f} vs {m6040['sharpe']:.3f}",">60/40"),
    ]
    print(f"{'ID':<8} {'Test':<28} {'Scope':<10} {'Pri':<4} {'Result':<8} {'Value':<26} {'Threshold'}")
    print("-" * 106)
    n_pass = n_fail = n_info = n_na = 0
    for aid, desc, scope, pri, passed, value, threshold in ac:
        if passed is None:
            status = "INFO" if "INFO" in threshold else "N/A"
            (n_info if status == "INFO" else n_na)  # noop -- just track
            if status == "INFO": n_info += 1
            else: n_na += 1
        elif passed: status = "PASS"; n_pass += 1
        else: status = "FAIL"; n_fail += 1
        print(f"{aid:<8} {desc:<28} {scope:<10} {pri:<4} {status:<8} {value:<26} {threshold}")
    print("-" * 106)
    print(f"\nSummary: {n_pass} PASS, {n_fail} FAIL (of {n_pass+n_fail} applicable), {n_info} INFO, {n_na} N/A (total 17)")
    print(f"\nOOS: Sharpe={wf_sh:.3f} CAGR={wf_ar:.2%} MaxDD={wf_dd:.2%} Gold={np.mean(gold_wts):.1%} TO={annual_to:.1%}/yr")
    print(f"vs v4={mv4['sharpe']:.3f} vs CSI300={m300['sharpe']:.3f} vs 60/40={m6040['sharpe']:.3f}")

def run_health():
    """Data freshness and integrity check."""
    ok, issues = health_check()
    print(f"Health Check -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 40)
    if ok:
        print("  ALL CLEAR -- data is fresh and intact.")
    else:
        print(f"  {len(issues)} ISSUE(S):")
        for i in issues: print(f"    [!] {i}")
    # Summary
    for e in ETFS:
        if e in T: print(f"  {e}: {len(T[e])} rows, last {T[e].index[-1].date()}")
    if pe_raw is not None: print(f"  PE: {len(pe_raw)} rows, last {pe_raw.index[-1].date()}")
    return ok

def run_report():
    """Generate markdown report with all metrics."""
    print("Generating report …")
    ar, sh, dd, _ = run_walk_forward()
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in T.values()])))
    latest = dr[dr <= today][-1] if len(dr[dr <= today]) > 0 else today
    pp = pe_pct_at(latest)
    eq_pct = 0.60 - pp * (0.60 - 0.10); eq_pct = max(0.10, min(0.60, eq_pct))
    g_pct = 0.30 if trend(GOLD, latest, 50) else 0.0
    bd_pct = max(0.0, 1.0 - eq_pct - g_pct)
    m300 = metrics_basic(T["510300"]["close"]["2020-01-01":"2025-12-31"])
    mv4 = ma_filter_benchmark()

    report_path = Path(__file__).parent / "v6_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# v6 PE-Band Report -- {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"## Live Signal ({latest.date()})\n\n")
        f.write(f"- CSI300 PE: {pp*100:.0f}th percentile (5y)\n")
        f.write(f"- Gold: {'TRENDING' if g_pct > 0 else 'not trending'}\n")
        f.write(f"- CSI300: {eq_pct*100:.0f}% | Gold: {g_pct*100:.0f}% | Bonds: {bd_pct*100:.0f}%\n\n")
        f.write(f"## Walk-Forward (2020-2025 OOS)\n\n")
        f.write(f"- Sharpe: {sh:.3f}\n- CAGR: {ar:.2%}\n- MaxDD: {dd:.2%}\n\n")
        f.write(f"## Benchmarks (2020-2025)\n\n")
        f.write(f"- v6 PE-Band: Sharpe={sh:.3f}\n")
        f.write(f"- v4 MA-filter: Sharpe={mv4['sharpe']:.3f}\n")
        f.write(f"- CSI300: Sharpe={m300['sharpe']:.3f}\n")
        f.write(f"\n## Diagnostic Years\n\n")
        for ts, te, label in [("2017","2017-12-31","2017"),("2019","2020-12-31","2019-2020"),("2022","2023-12-31","2022-2023")]:
            m = backtest(f"{ts}-01-01", te)
            f.write(f"- {label}: CAGR={m['cagr']:.2%} MaxDD={m['maxdd']:.2%}\n")
    print(f"Report saved to {report_path}")

# ─── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v6 PE-Band ETF Rotation")
    parser.add_argument("--live", action="store_true", help="Current allocation signal")
    parser.add_argument("--verify", action="store_true", help="17-criterion acceptance test")
    parser.add_argument("--fetch", action="store_true", help="Fetch latest data (ETFs+macro+PE)")
    parser.add_argument("--health", action="store_true", help="Data freshness + integrity check")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--diagnostics", action="store_true", help="Diagnostic year tests")
    parser.add_argument("--benchmarks", action="store_true", help="Benchmark comparison")
    args = parser.parse_args()

    single = sum([args.live, args.verify, args.fetch, args.health, args.report,
                  args.diagnostics, args.benchmarks])
    if single == 0:
        run_walk_forward(); run_diagnostics(); run_benchmarks()
    if args.live:     run_live()
    if args.verify:   run_verify()
    if args.fetch:    fetch_all()
    if args.health:   run_health()
    if args.report:   run_report()
    if args.diagnostics: run_diagnostics()
    if args.benchmarks:  run_walk_forward(); run_benchmarks()

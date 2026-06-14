"""
Auto-Update + Backtest Pipeline for ETF Rotation v4 (Rising MA Filter)
======================================================================
Usage:
    python scripts/auto_update.py                   # full: fetch + backtest
    python scripts/auto_update.py --skip-fetch      # only backtest (offline)
    python scripts/auto_update.py --skip-backtest   # fetch data only
    python scripts/auto_update.py --signal          # live allocation signal only

Process:
  1. Fetch latest daily bars for all 7 ETFs from AkShare (Sina source)
  2. Append new bars to data/clean/*.parquet (deduplicated by trade_date)
  3. Backtest: print v4 summary + benchmarks
  4. --signal: print live allocation target
"""

import os, sys, time, argparse, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

for k in list(os.environ.keys()):
    if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE"):
        os.environ.pop(k, None)

ALL_ETFS = {
    "510300": "CSI300",   "510500": "CSI500",   "159915": "ChiNext",
    "510880": "Dividend", "511010": "Bonds",    "518880": "Gold",
    "511880": "Cash",
}
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "clean")


# ====================== STEP 1: ETF data fetch ======================

def _sina_symbol(code):
    if code.startswith(("sh", "sz")): return code
    prefix = "sh" if code.startswith(("51", "58", "56", "60")) else "sz"
    return prefix + code

def _save_clean(code, df_new):
    import pandas as pd
    path = os.path.join(CLEAN_DIR, "{}.parquet".format(code))
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        df_new = pd.concat([existing, df_new], ignore_index=True)
        df_new = df_new.drop_duplicates(subset=["trade_date"], keep="last")
        df_new = df_new.sort_values("trade_date")
    df_new.to_parquet(path, index=False)
    return path

def fetch_latest_data():
    import akshare as ak
    import pandas as pd
    print("=" * 60)
    print("  STEP 1: FETCH LATEST ETF DATA")
    print("=" * 60)
    results = {}
    _last_call = 0.0
    for code, name in ALL_ETFS.items():
        sina_sym = _sina_symbol(code)
        elapsed = time.monotonic() - _last_call
        if elapsed < 1.5: time.sleep(1.5 - elapsed)
        _last_call = time.monotonic()
        try:
            df = ak.fund_etf_hist_sina(symbol=sina_sym)
            if df is None or df.empty:
                print("  WARN  {} ({}) -> 0 bars".format(code, name))
                results[code] = {"status": "empty", "new": 0}
                continue
            df = df.rename(columns={"date":"trade_date","open":"open","high":"high",
                                    "low":"low","close":"close","volume":"volume"})
            df["symbol"] = sina_sym
            df["amount"] = df.get("amount", 0.0)
            cols = ["symbol","trade_date","open","high","low","close","volume","amount"]
            df = df[[c for c in cols if c in df.columns]]
            existing_path = os.path.join(CLEAN_DIR, "{}.parquet".format(code))
            old_count = 0;  old_last = "N/A"
            if os.path.exists(existing_path):
                old_df = pd.read_parquet(existing_path)
                old_count = len(old_df);  old_last = old_df["trade_date"].max()
            _save_clean(code, df)
            new_df = pd.read_parquet(existing_path)
            final_count = len(new_df);  added = final_count - old_count
            new_last = str(df["trade_date"].max()) if len(df) > 0 else "N/A"
            print("  OK    {} ({:8s}) -> +{:>4} bars | {:>5} -> {:>5} | {} -> {}".format(
                code, name, added, old_count, final_count, old_last, new_last))
            results[code] = {"status":"ok","new":added,"total":final_count,"last_date":new_last}
        except Exception as e:
            print("  FAIL  {} ({}) -> {}".format(code, name, e))
    success = sum(1 for v in results.values() if v.get("status")=="ok")
    total_new = sum(v.get("new",0) for v in results.values())
    print("\n  {}/{} success, +{} total new bars".format(success, len(ALL_ETFS), total_new))
    return results


# ====================== STEP 2: Backtest ======================

def run_backtest():
    import pandas as pd
    from _funcs import load, backtest, metrics, year_bt, P, POOL_V6
    print("\n" + "=" * 60)
    print("  STEP 2: BACKTEST  (v4: 3-factor, rising MA)")
    print("=" * 60)
    data = load(POOL_V6)
    test_start = "2022-01-01"
    test_end = datetime.now().strftime("%Y-%m-%d")
    print("  Period: {} to {}".format(test_start, test_end))
    bt = backtest(data, test_start, test_end, tn=1, ef="rising",
                  th=P["th"], vt=P["vt"], wm=P["wm"], wa=P["wa"], wr=P["wr"])
    m = metrics(bt)
    print("\n  Results:")
    for k, l in [("annual_return","CAGR"),("sharpe_ratio","Sharpe"),
                 ("max_drawdown","MaxDD"),("calmar_ratio","Calmar")]:
        fmt = "{:.2%}".format(m[k]) if k in ("annual_return","max_drawdown") else "{:.2f}".format(m[k])
        print("    {}: {}".format(l, fmt))
    print("\n  Yearly:")
    for yr in range(2022, int(test_end[:4])+1):
        yb = year_bt(bt, yr)
        if yb:
            print("    {}: {:>6.1%}  MaxDD={:>5.1%}  Sharpe={:.2f}".format(
                yr, yb["return_"], yb["max_drawdown"], yb["sharpe"]))
    print("\n  Benchmarks:")
    for etf, name in [("510300","CSI300"),("518880","Gold"),("511010","Bond")]:
        c = data[etf]["close"]
        c = c[(c.index >= pd.Timestamp(test_start)) & (c.index <= pd.Timestamp(test_end))]
        bm = metrics(c)
        print("    {} B&H: CAGR={:.2%}  MaxDD={:.2%}  Sharpe={:.2f}".format(
            name, bm["annual_return"], bm["max_drawdown"], bm["sharpe_ratio"]))
    return bt, m


# ====================== Signal ======================

def run_live_signal():
    import pandas as pd
    from _funcs import load, features, P, POOL_V6
    print("\n" + "=" * 60)
    print("  LIVE ALLOCATION SIGNAL  (v4: 3-factor, rising MA)")
    print("=" * 60)
    data = load(POOL_V6)
    all_idx = set.intersection(*[set(df.index) for df in data.values()])
    latest = pd.Timestamp(sorted(all_idx)[-1])
    print("  Data through: {}".format(latest.date()))
    names = {"510300":"CSI300","510500":"CSI500","159915":"ChiNext",
             "510880":"Dividend","511010":"Bonds","518880":"Gold","511880":"Cash"}
    scores = {}
    for etf, df in data.items():
        if latest not in df.index: continue
        f = features(df.loc[:latest])
        above_120 = int(f["above_120"].iloc[-1]);  ma_rising = int(f["rising"].iloc[-1])
        adx_n = min(float(f["adx"].iloc[-1]) if not pd.isna(f["adx"].iloc[-1]) else 0, 100) / 100
        r20 = float(f["r20"].iloc[-1]) if not pd.isna(f["r20"].iloc[-1]) else 0
        mom_n = max(0, min((r20 + 0.3) / 0.6, 1.0));  px = float(f["close"].iloc[-1])
        equity = etf in ("510300","510500","159915","510880")
        if equity and not ma_rising:
            final_score = 0.0;  blocked = True
        else:
            final_score = P["wm"]*above_120 + P["wa"]*adx_n + P["wr"]*mom_n;  blocked = False
        scores[etf] = {"score":final_score,"blocked":blocked,"above_120":above_120,
                       "adx_n":adx_n,"mom_n":mom_n,"r20":r20,"ma_rising":ma_rising,"close":px}
    print()
    print("  {:8s} {:>8s} {:>7s} {:>5s} {:>7s} {:>7s} {:>6s} {}".format(
        "ETF","Score","Close",">120","ADX_n","Mom_n","Slope","Status"))
    print("  " + "-" * 70)
    for etf, s in sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True):
        if etf=="511880":  status = "(cash)"
        elif s["blocked"]:  status = "BLOCKED"
        elif s["score"]<P["th"]:  status = "no pass"
        else:  status = "READY"
        print("  {:8s} {:>7.3f} {:>8.2f} {:>5d} {:>6.2f} {:>7.2f} {:>6d} {}".format(
            etf, s["score"], s["close"], s["above_120"],
            s["adx_n"], s["mom_n"], s["ma_rising"], status))
    valid = {e:s for e,s in scores.items() if e!="511880" and s["score"]>=P["th"] and not s["blocked"]}
    print("\n  Allocation decision:")
    if not valid:
        print("    -> ALL CASH (511880)")
    else:
        best = max(valid, key=lambda e:valid[e]["score"])
        print("    -> BUY {} ({})  score={:.3f}".format(best, names.get(best,best), valid[best]["score"]))
        others = sorted(valid.items(), key=lambda x:x[1]["score"], reverse=True)
        if len(others)>1:
            print("    Runner-up: {} ({})  score={:.3f}".format(others[1][0], names.get(others[1][0],others[1][0]), others[1][1]["score"]))


# ====================== Main ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-update ETF rotation pipeline")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--signal", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Rising MA ETF Rotation -- Auto Pipeline")
    print("  {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 60)

    t0 = time.time()
    if not args.skip_fetch:
        fetch_latest_data()
    else:
        print("[SKIP] Data fetching (--skip-fetch)")
    if not args.skip_backtest:
        if args.signal:
            run_live_signal()
        else:
            run_backtest()
    else:
        print("[SKIP] Backtest (--skip-backtest)")
    print("\n  Pipeline complete: {:.1f}s".format(time.time()-t0))

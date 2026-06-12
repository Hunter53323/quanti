"""
Signal-level decomposition: isolate the effect of market state filter.
3 groups, single stock selection engine, different market regimes.
Group1: trend stocks only, always long. Group2: trend stocks + MA/ADX filter. Group3: trend stocks + PE filter.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, time
from scripts.run_complete_backtest import *

DATA_DIR = Path("C:/study/AIWorkspace/quanti/data/clean")
TEST = ("2022-01-01", "2025-12-31")
CAPITAL = 90000.0; TOP_N = 3; STOP = -0.10; MA_P = 120

# ── Shared: trend stock selection + trailing stop (EXACT same for all groups) ──
def trend_selection_engine(st, et, csi, dates, masks, scores, me, trades,
                            position_filter_fn, capital=CAPITAL, top_n=TOP_N,
                            stop=STOP, ma_p=MA_P, **kwargs):
    """
    position_filter_fn(date, csi_data, positions, cash) -> (allowed_to_hold, position_budget_pct)
    Returns True/False for market participation, and 0.0-1.0 for how much capital to deploy.
    """
    dr = []; pos = {}; cash = capital; peq = capital

    for i, d in enumerate(dates):
        if i == 0: continue
        is_me = d.month != dates[i-1].month

        # Trailing stops
        cr = [cash]
        for c in list(pos):
            trail_stop(pos, cr, st, et, c, d, stop, trades, "t")
        cash = cr[0]

        # Market state filter
        sd = csi[csi.index <= d]
        allowed, budget_pct = position_filter_fn(d, sd, pos, cash, **kwargs)

        if not allowed and is_me:
            # Liquidate all
            for c in list(pos):
                pr = sc(st, c, d)
                if pr:
                    cash += pr * pos[c]["qty"]
                    log_t(trades, d, c, "sell", pos[c]["qty"], pr,
                          (pr - pos[c]["avg"]) * pos[c]["qty"], "liq")
                    del pos[c]

        elif is_me:
            sel, _ = select_top(masks, scores, me, d)

            # Sell rotated-out
            for c in list(pos):
                if c not in sel:
                    pr = sc(st, c, d)
                    if pr:
                        cash += pr * pos[c]["qty"]
                        log_t(trades, d, c, "sell", pos[c]["qty"], pr,
                              (pr - pos[c]["avg"]) * pos[c]["qty"], "rotate")
                        del pos[c]

            # Buy new (scaled by budget_pct)
            deploy = cash * budget_pct
            nw = [s for s in sel if s not in pos]
            n_positions = max(len(sel), 1)
            if nw and deploy > 0:
                per = deploy / n_positions * 0.92
                for c in nw:
                    pr = sc(st, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= cash:
                            cash -= q * pr
                            pos[c] = {"qty": q, "avg": pr, "hwm": pr}
                            log_t(trades, d, c, "buy", q, pr, 0, "entry")

        # Equity
        eq = cash + sum(p["qty"] * (sc(st, c, d) or 0)
                       for c, p in pos.items() if sc(st, c, d))
        if peq > 0:
            dr.append((eq - peq) / peq)
        peq = eq

    return np.array(dr)


# ── Group 1: No filter, always long ──
def g1_no_filter(d, sd, pos, cash, **kwargs):
    return True, 1.0  # always allowed, full capital

# ── Group 2-a: MA bull/bear (S1 style) ──
def g2a_ma_filter(d, sd, pos, cash, **kwargs):
    if len(sd) >= MA_P:
        m120 = float(sma(sd["close"], MA_P).iloc[-1])
        if not pd.isna(m120) and m120 > 0:
            bull = float(sd["close"].iloc[-1]) > m120
            return bull, 1.0 if bull else 0.0
    return True, 1.0  # default allow

# ── Group 2-b: ADX continuous (S3 style) ──
def g2b_adx_filter(d, sd, pos, cash, **kwargs):
    if len(sd) >= 60:
        try:
            adx_v = float(adx_np(sd["high"].values, sd["low"].values, sd["close"].values, 14)[-1])
            if not np.isnan(adx_v):
                trend_st = min(max(adx_v / 40.0, 0.0), 1.0)
                allowed = adx_v >= 20  # S3 threshold
                return allowed, trend_st if allowed else 0.0
        except:
            pass
    return True, 1.0

# ── Group 3-a: PE percentile (S5 style) ──
def g3_pe_filter_raw(d, sd, pos, cash, pe_df=None, **kwargs):
    if pe_df is None:
        return True, 1.0
    pe_vals = kwargs.get("_pe_vals")
    if pe_vals is None:
        return True, 1.0
    pn = pe_vals[pe_vals.index <= d]
    if len(pn) == 0 or pd.isna(pn["pct"].iloc[-1]):
        return True, 1.0
    pctl = float(pn["pct"].iloc[-1])
    if pctl < 30:
        return True, 0.80
    elif pctl < 70:
        return True, 0.50
    else:
        return True, 0.20

# ── Group 3-b: PE x ADX grid ──
def g3_pe_adx_grid(d, sd, pos, cash, pe_df=None, **kwargs):
    if pe_df is None:
        return True, 1.0
    pe_vals = kwargs.get("_pe_vals")
    if pe_vals is None:
        return True, 1.0
    pn = pe_vals[pe_vals.index <= d]
    pctl = 50.0
    if len(pn) > 0 and not pd.isna(pn["pct"].iloc[-1]):
        pctl = float(pn["pct"].iloc[-1])
    if pctl < 30: pe_ceiling = 0.80
    elif pctl < 70: pe_ceiling = 0.50
    else: pe_ceiling = 0.20

    adx_v = 20.0
    if len(sd) >= 60:
        try:
            a = adx_np(sd["high"].values, sd["low"].values, sd["close"].values, 14)
            if not np.isnan(a[-1]): adx_v = float(a[-1])
        except: pass
    if adx_v >= 25: adx_factor = 1.00
    elif adx_v >= 20: adx_factor = 0.50
    else: adx_factor = 0.20

    return True, pe_ceiling * adx_factor


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("SIGNAL DECOMPOSITION: isolate market filter effect")
    print("=" * 80, flush=True)

    et = {c: load_etf(c) for c in ["510300", "510500", "159915"]}
    csi = load_csi300(); st = load_stocks(100)
    dates = sorted(set(pd.date_range(TEST[0], TEST[1], freq="B")).intersection(set(csi.index)))
    yrs = len(dates) / 252

    # BM
    bm_r = []
    for i in range(1, len(dates)):
        t = csi[csi.index <= dates[i]]; y = csi[csi.index <= dates[i-1]]
        bm_r.append(float(t["close"].iloc[-1]) / float(y["close"].iloc[-1]) - 1 if len(t) and len(y) else 0)
    bm = metrics(np.array(bm_r), yrs)

    # PE
    pe_df = None; pe_vals = None
    try:
        from quanti.data.index_pe import IndexPEFetcher
        fetcher = IndexPEFetcher(); raw = fetcher.fetch_history("000300.SH")
        recs = [{"date": pd.Timestamp(r["trade_date"]), "pe": r["pe"]} for r in raw if r["pe"] > 0]
        pe_df = pd.DataFrame(recs).set_index("date").sort_index()
        pe_vals = pe_df.copy()
        pe_vals["pct"] = pe_vals["pe"].rolling(250 * 10, min_periods=250).apply(
            lambda x: (x <= x.iloc[-1]).mean() * 100, raw=False)
    except Exception as e:
        print(f"  PE unavailable: {e}", flush=True)

    print(f"  CSI300 B&H: CAGR={bm['CAGR']*100:.2f}%", flush=True)

    # Precompute
    masks, sc_arr, me_arr = precompute(st, csi)

    # Run all groups
    groups = [
        ("G1: Always long (no filter)", g1_no_filter, {}),
        ("G2a: MA bull/bear (S1 style)", g2a_ma_filter, {}),
        ("G2b: ADX continuous (S3 style)", g2b_adx_filter, {}),
        ("G3a: PE percentile only", g3_pe_filter_raw,
         {"pe_df": pe_df, "_pe_vals": pe_vals} if pe_vals is not None else {}),
        ("G3b: PE x ADX grid", g3_pe_adx_grid,
         {"pe_df": pe_df, "_pe_vals": pe_vals} if pe_vals is not None else {}),
    ]

    print(f"\n{'Group':<35s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'AnnVol':>8s} {'Trades':>7s}")
    print("-" * 80)

    results = {}
    for lbl, fn, kw in groups:
        t0 = time.time(); tb = []
        rets = trend_selection_engine(st, et, csi, dates, masks, sc_arr, me_arr, tb, fn, **kw)
        m = metrics(rets, yrs)
        sells = [t for t in tb if t["side"] == "sell"]
        wins = sum(1 for t in sells if t["pnl"] > 0)
        wr = wins / max(len(sells), 1) * 100
        results[lbl] = {**m, "trades": len(sells), "wr": wr}
        print(f"{lbl:<35s} {m['CAGR']*100:>7.2f}% {m['Sharpe']:>8.3f} "
              f"{m['MaxDD']*100:>7.2f}% {m['AnnVol']*100:>7.2f}% {len(sells):>7d} "
              f"WR={wr:.0f}% ({time.time()-t0:.0f}s)", flush=True)

    # Delta vs no-filter baseline
    g1 = results["G1: Always long (no filter)"]
    print(f"\n{'='*80}")
    print(f"DELTA vs G1 (always long, no filter: CAGR={g1['CAGR']*100:.2f}%)")
    print(f"{'Group':<35s} {'dCAGR':>8s} {'dSharpe':>8s} {'dMaxDD':>8s}")
    print("-" * 70)
    for lbl, m in results.items():
        dc = m["CAGR"] - g1["CAGR"]
        ds = m["Sharpe"] - g1["Sharpe"]
        dd = (m["MaxDD"] - g1["MaxDD"]) * 100  # closer to 0 = better
        print(f"{lbl:<35s} {dc*100:>+7.2f}% {ds:>+8.3f} {dd:>+7.2f}pp")

    print("\nDone.")


if __name__ == "__main__":
    main()

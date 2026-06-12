"""
Strategy fix validation: Option A (no DD exit) vs Option B (vol-scaled sizing).
Train (2015-2021) vs Test (2022-2025) for both variants.
Also sweeps dd_exit from 25-50 to find if any threshold avoids permanent cash lock.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

from quanti.data.storage import DataStorage

CAPITAL = 90000
COMM = 0.00025

# ── Configs ──
# All use the winning trend params: top_n=5, stop=-10%, trend_cond=4, mkt_req=True
BASE_TOP_N = 5
BASE_STOP = -10
BASE_TREND_COND = 4
BASE_MKT_REQ = True

configs = [
    # (name, top_n, stop_pct, min_trend, dd_exit, mkt_req, sizing_mode)
    # sizing_mode: "normal" = fixed per-stock allocation, "volscale" = size inversely proportional to recent vol
    ("A_NoDD_Normal",   BASE_TOP_N, BASE_STOP, BASE_TREND_COND, 0,   BASE_MKT_REQ, "normal"),
    ("A_NoDD_VolScale", BASE_TOP_N, BASE_STOP, BASE_TREND_COND, 0,   BASE_MKT_REQ, "volscale"),
    ("B_DD25_Normal",   BASE_TOP_N, BASE_STOP, BASE_TREND_COND, 25,  BASE_MKT_REQ, "normal"),
    ("B_DD30_Normal",   BASE_TOP_N, BASE_STOP, BASE_TREND_COND, 30,  BASE_MKT_REQ, "normal"),
    ("B_DD35_Normal",   BASE_TOP_N, BASE_STOP, BASE_TREND_COND, 35,  BASE_MKT_REQ, "normal"),
    ("B_DD40_Normal",   BASE_TOP_N, BASE_STOP, BASE_TREND_COND, 40,  BASE_MKT_REQ, "normal"),
    ("B_DD50_Normal",   BASE_TOP_N, BASE_STOP, BASE_TREND_COND, 50,  BASE_MKT_REQ, "normal"),
    ("C_Old_DD15",      BASE_TOP_N, BASE_STOP, BASE_TREND_COND, 15,  BASE_MKT_REQ, "normal"),
]

print("=" * 70)
print("Loading data...")
print("=" * 70)
storage = DataStorage()
all_files = sorted(storage.clean_dir.glob("*.parquet"))
stocks = [p.stem for p in all_files if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]
stock_data = {}
all_dates_set = set()
for i, code in enumerate(stocks):
    if (i+1) % 200 == 0: print(f"  Loading {i+1}/{len(stocks)}...")
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200: continue
    dates = [r.trade_date for r in raw]
    stock_data[code] = (
        np.array([r.close for r in raw], dtype=np.float64),
        np.array([r.high for r in raw], dtype=np.float64),
        np.array([r.low for r in raw], dtype=np.float64),
        np.array([r.volume for r in raw], dtype=np.float64),
        dates
    )
    all_dates_set.update(dates)
all_dates = sorted(all_dates_set)
print(f"Loaded {len(stock_data)} stocks, {len(all_dates)} days")

def data_at(code, date_str, n):
    if code not in stock_data: return None
    c, h, l, v, d = stock_data[code]
    idx = None
    for i in range(len(d)-1, -1, -1):
        if d[i] <= date_str: idx = i+1; break
    if idx is None or idx < n: return None
    return (c[idx-n:idx], h[idx-n:idx], l[idx-n:idx], v[idx-n:idx])

def price_on(code, date_str):
    if code not in stock_data: return None
    c,_,_,_,d = stock_data[code]
    for i in range(len(d)-1, -1, -1):
        if d[i] <= date_str: return c[i]
    return None

def sma(arr, period):
    if len(arr) < period: return None
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(np.insert(arr, 0, 0))
    out[period-1:] = (cs[period:] - cs[:-period]) / period
    return out

def adx(high, low, close, period=14):
    n = len(close)
    if n < period*2: return None
    tr = np.zeros(n); tr[0] = high[0]-low[0]
    for i in range(1, n): tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = high[i]-high[i-1]; down = low[i-1]-low[i]
        if up > down and up > 0: pdm[i] = up
        if down > up and down > 0: mdm[i] = down
    atr = np.full(n, np.nan); atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, n): atr[i] = (tr[i] + (period-1)*atr[i-1])/period
    pdi = np.full(n, np.nan); mdi = np.full(n, np.nan)
    pdi[period] = np.mean(pdm[1:period+1])/atr[period]*100
    mdi[period] = np.mean(mdm[1:period+1])/atr[period]*100
    for i in range(period+1, n):
        if np.isnan(atr[i]) or atr[i] < 1e-10:
            # NaN-safe: copy previous values to avoid gap in Wilder smoothing
            pdi[i] = pdi[i-1] if not np.isnan(pdi[i-1]) else 0.0
            mdi[i] = mdi[i-1] if not np.isnan(mdi[i-1]) else 0.0
            continue
        pdi[i] = (pdm[i] + (period-1)*pdi[i-1])/period / atr[i] * 100
        mdi[i] = (mdm[i] + (period-1)*mdi[i-1])/period / atr[i] * 100
    dx = np.abs(pdi-mdi)/(pdi+mdi+1e-10)*100
    adx_arr = np.full(n, np.nan)
    # Find first valid ADX seed
    seed_start = period*2-1
    while seed_start < n and np.isnan(np.nanmean(dx[max(1,seed_start-period+1):seed_start+1])):
        seed_start += 1
    if seed_start >= n: return adx_arr
    adx_arr[seed_start] = np.nanmean(dx[max(1,seed_start-period+1):seed_start+1])
    prev = adx_arr[seed_start]
    for i in range(seed_start+1, n):
        cur = dx[i]
        if np.isnan(cur):
            adx_arr[i] = prev  # propagate last valid value through NaN gaps
        else:
            prev = (cur + (period-1)*prev)/period
            adx_arr[i] = prev
    return adx_arr

def is_stock_uptrend(closes, highs, lows, volumes):
    if len(closes) < 130: return False, 0
    ma120 = sma(closes, 120)
    if ma120 is None or np.isnan(ma120[-1]): return False, 0
    above_ma = closes[-1] > ma120[-1]
    recent_high = np.max(highs[-20:]); prev_high = np.max(highs[-60:-20])
    higher_high = recent_high > prev_high
    recent_low = np.min(lows[-20:]); prev_low = np.min(lows[-60:-20])
    higher_low = recent_low > prev_low
    ma20_arr = sma(closes, 20); ma60_arr = sma(closes, 60)
    if ma20_arr is None or ma60_arr is None: return False, 0
    if np.isnan(ma20_arr[-1]) or np.isnan(ma60_arr[-1]): return False, 0
    ma_aligned = ma20_arr[-1] > ma60_arr[-1] > ma120[-1]
    adx_arr = adx(highs, lows, closes, 14)
    adx_ok = adx_arr is not None and not np.isnan(adx_arr[-1]) and adx_arr[-1] > 25
    vol_20 = np.mean(volumes[-21:-1])
    vol_surge = volumes[-1] > vol_20 * 1.2
    conditions = [above_ma, higher_high and higher_low, ma_aligned, adx_ok, vol_surge]
    score = sum(conditions)
    is_trend = above_ma and adx_ok and score >= 3
    return is_trend, score

def market_uptrend(date_str):
    d = data_at("510300", date_str, 200)
    if d is None: return True, 5
    c, h, l, v = d
    return is_stock_uptrend(c, h, l, v)

def trend_strength_score(closes):
    if len(closes) < 130: return 0
    ret_3m = closes[-1]/closes[-63]-1 if closes[-63]>1e-6 else 0
    ret_6m = closes[-1]/closes[-126]-1 if closes[-126]>1e-6 else 0
    mom = (min(max(ret_3m/0.5,0),1)*0.5 + min(max(ret_6m/0.8,0),1)*0.5)*100
    window_c = closes[-61:]
    dr = np.diff(window_c)/(window_c[:-1]+1e-10)
    vol = np.nanstd(dr)
    vol_s = (1-min(vol/0.04,1))*100
    return 0.6*mom + 0.4*vol_s

def get_monthly_dates(dates, start, end):
    monthly = []
    for d in dates:
        if d < start or d > end: continue
        dm = d[4:6]
        if not monthly or dm != monthly[-1][4:6]: monthly.append(d)
    return monthly

def stock_recent_volatility(closes, window=20):
    """Annualized volatility of recent daily returns."""
    if len(closes) < window+2: return 0.40  # default high vol
    rets = np.diff(closes[-window-1:]) / (closes[-window-1:-1] + 1e-10)
    daily_vol = np.nanstd(rets)
    ann_vol = daily_vol * np.sqrt(252)
    return ann_vol

def run_backtest(top_n, stop_pct, min_trend_cond, mkt_required, dd_exit_pct, sizing_mode, start_d, end_d):
    """Enhanced backtest with vol-scaling option."""
    rebal = get_monthly_dates(all_dates, start_d, end_d)
    if len(rebal) < 12:
        return {"cagr": 0, "sharpe": 0, "maxdd": 0, "final": CAPITAL, "n_y": 0,
                "n_inv": 0, "n_cash": 0, "n_dd": 0, "hit_rate": 0}

    cash = CAPITAL
    holdings = {}
    eq_curve = [cash]
    max_eq = cash
    dd_exit_triggered = False
    n_inv = 0; n_cash = 0; n_dd = 0
    invested_rets = []

    for reb_date in rebal:
        # Stop-loss per position
        for sym in list(holdings.keys()):
            p = price_on(sym, reb_date)
            if p is None or p < 0.01:
                cash += holdings[sym]["value"] * 0.7
                del holdings[sym]; continue
            holdings[sym]["price"] = p
            holdings[sym]["value"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]:
                holdings[sym]["hwm"] = p
            hwm = holdings[sym]["hwm"]
            if hwm > 0 and (p/hwm - 1)*100 < stop_pct:
                mv = holdings[sym]["qty"]*p
                cash += mv*(1-COMM)
                del holdings[sym]

        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())

        # Drawdown tracking (for metrics, not exit in Option A)
        if total > max_eq: max_eq = total
        current_dd = (max_eq - total) / max_eq * 100

        # DD exit (only if dd_exit_pct > 0)
        if dd_exit_pct > 0:
            if current_dd > dd_exit_pct:
                for sym in list(holdings.keys()):
                    mv = holdings[sym]["qty"]*holdings[sym]["price"]
                    cash += mv*(1-COMM)
                    del holdings[sym]
                dd_exit_triggered = True
                n_dd += 1
            elif dd_exit_triggered:
                if total / max_eq > 0.92:
                    dd_exit_triggered = False

        if dd_exit_triggered:
            n_cash += 1
            eq_curve.append(total)
            continue

        # Market trend gate
        mkt_trend, _ = market_uptrend(reb_date)
        if mkt_required and not mkt_trend:
            n_cash += 1
            eq_curve.append(total)
            continue

        # Find trending stocks
        trending = []
        for code in stock_data:
            d = data_at(code, reb_date, 260)
            if d is None: continue
            c, h, l, v = d
            is_t, n_cond = is_stock_uptrend(c, h, l, v)
            if is_t and n_cond >= min_trend_cond:
                s = trend_strength_score(c)
                trending.append((code, s, n_cond))

        if not trending:
            n_cash += 1
            eq_curve.append(total)
            continue

        n_inv += 1
        trending.sort(key=lambda x: x[1], reverse=True)
        selected = {t[0] for t in trending[:top_n]}

        # Sell rotated-out
        for sym in list(holdings.keys()):
            if sym not in selected:
                mv = holdings[sym]["qty"]*holdings[sym]["price"]
                cash += mv*(1-COMM)
                del holdings[sym]

        # ── Position sizing ──
        n_pos = max(len(selected), 1)
        base_per_stock = total / n_pos * 0.90

        # Volatility scaling: reduce size for high-vol stocks
        if sizing_mode == "volscale":
            # Target 20% annualized vol per position
            target_vol = 0.20
            # Apply DD-based scale factor: when in drawdown > 10%, size down
            dd_scale = min(1.0, max(0.3, 1.0 - current_dd / 40.0))
            base_per_stock *= dd_scale

        # Buy
        for _j, sym in enumerate(selected):
            p = price_on(sym, reb_date)
            if p is None or p < 0.01: continue
            per_stock = base_per_stock

            if sizing_mode == "volscale":
                # Individual stock vol adjustment
                sd = data_at(sym, reb_date, 65)
                if sd is not None:
                    stock_vol = stock_recent_volatility(sd[0], 20)
                    vol_mult = min(1.5, max(0.3, target_vol / max(stock_vol, 0.10)))
                    per_stock *= vol_mult

            target_qty = int(per_stock / p / 100) * 100
            if target_qty < 100: continue

            if sym in holdings:
                cur = holdings[sym]["qty"]
                diff = target_qty - cur
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff > 0 and cash >= cost*(1+COMM):
                        cash -= cost*(1+COMM)
                        holdings[sym]["qty"] = target_qty
                    elif diff < 0:
                        cash += cost*(1-COMM)
                        holdings[sym]["qty"] = target_qty
            else:
                cost = target_qty*p
                if cash >= cost*(1+COMM):
                    cash -= cost*(1+COMM)
                    holdings[sym] = {"qty": target_qty, "price": p, "value": cost, "hwm": p}

        prev_total = eq_curve[-1]
        total = cash + sum(h["qty"]*h["price"] for h in holdings.values())
        eq_curve.append(total)
        if prev_total > 0:
            invested_rets.append((total - prev_total) / prev_total * 100)

    # Metrics
    eq = np.array(eq_curve)
    n_y = (int(rebal[-1][:4]) - int(rebal[0][:4])) + (int(rebal[-1][4:6]) - int(rebal[0][4:6])) / 12.0
    if n_y <= 0: n_y = 1
    cagr = ((eq[-1]/eq[0])**(1/n_y)-1)*100 if eq[0] > 0 and eq[-1] > 0 else 0
    mr = np.diff(eq)/(eq[:-1]+1e-10)
    sharpe = np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12) if len(mr)>1 and np.std(mr)>1e-10 else 0
    peak = eq[0]; maxdd = 0.0
    for v in eq:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > maxdd: maxdd = dd

    hit_rate = sum(1 for r in invested_rets if r > 0) / max(len(invested_rets), 1) * 100

    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "final": eq[-1], "n_y": n_y,
            "n_inv": n_inv, "n_cash": n_cash, "n_dd": n_dd, "hit_rate": hit_rate}

# ── Run all configs ──
print(f"\n{'='*90}")
print(f"{'Config':<28s} | {'Train C':>7s} | {'Train S':>7s} | {'Train D':>7s} | {'Test C':>7s} | {'Test S':>7s} | {'Test D':>7s} | {'DD infl':>7s} | {'Inv%':>6s}")
print(f"{'─'*28}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}")

results = []
for name, tn, sp, tc, dd, mkt, sz in configs:
    tr = run_backtest(tn, sp, tc, mkt, dd, sz, "20150101", "20211231")
    te = run_backtest(tn, sp, tc, mkt, dd, sz, "20220101", "20251231")
    sharpe_decay = (tr["sharpe"] - te["sharpe"]) / tr["sharpe"] if tr["sharpe"] > 0.01 else 0
    dd_inflate = (te["maxdd"] / tr["maxdd"] - 1) * 100 if tr["maxdd"] > 0.01 else 0
    inv_pct = te["n_inv"] / max(te["n_inv"] + te["n_cash"], 1) * 100
    results.append((name, tr, te, sharpe_decay, dd_inflate, inv_pct))
    print(f"{name:<28s} | {tr['cagr']:+6.1f}% | {tr['sharpe']:+6.3f} | {tr['maxdd']:+6.1f}% | "
          f"{te['cagr']:+6.1f}% | {te['sharpe']:+6.3f} | {te['maxdd']:+6.1f}% | "
          f"{dd_inflate:+6.0f}% | {inv_pct:4.0f}%")

# ── Detailed print for best ──
print(f"\n{'='*90}")
print("DETAILED COMPARISON: Option A (No DD Exit) vs Option C (Old DD15) vs Option B (DD35)")
print(f"{'='*90}")

for name, tn, sp, tc, dd, mkt, sz in configs:
    if name not in ("A_NoDD_Normal", "C_Old_DD15", "B_DD35_Normal"):
        continue
    print(f"\n─── {name}: Train ───")
    te = run_backtest(tn, sp, tc, mkt, dd, sz, "20150101", "20211231")
    print(f"  CAGR={te['cagr']:+.1f}% Sharpe={te['sharpe']:.3f} MaxDD={te['maxdd']:.1f}%  "
          f"Invested={te['n_inv']}mo Cash={te['n_cash']}mo DD_exit={te['n_dd']}mo  HitRate={te['hit_rate']:.0f}%")

    print(f"─── {name}: Test ───")
    te2 = run_backtest(tn, sp, tc, mkt, dd, sz, "20220101", "20251231")
    print(f"  CAGR={te2['cagr']:+.1f}% Sharpe={te2['sharpe']:.3f} MaxDD={te2['maxdd']:.1f}%  "
          f"Invested={te2['n_inv']}mo Cash={te2['n_cash']}mo DD_exit={te2['n_dd']}mo  HitRate={te2['hit_rate']:.0f}%")

# ── Sub-period breakdown for best Option A ──
print(f"\n{'='*90}")
print("SUB-PERIOD: Option A (No DD Exit) Test Period Breakdown")
print(f"{'='*90}")

sub_periods = [
    ("2022", "20220101", "20221231"),
    ("2023", "20230101", "20231231"),
    ("2024", "20240101", "20241231"),
    ("2025", "20250101", "20251231"),
]
print(f"  {'Period':>6s} | {'CAGR':>7s} | {'Sharpe':>7s} | {'MaxDD':>6s} | Hit%")
print(f"  {'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*4}")
for sp_name, sp_start, sp_end in sub_periods:
    r = run_backtest(BASE_TOP_N, BASE_STOP, BASE_TREND_COND, BASE_MKT_REQ, 0, "normal", sp_start, sp_end)
    print(f"  {sp_name:>6s} | {r['cagr']:+6.1f}% | {r['sharpe']:+6.3f} | {r['maxdd']:+6.1f}% | {r['hit_rate']:3.0f}%")

# ── Option A with VolScale sub-period ──
print(f"\n{'='*90}")
print("SUB-PERIOD: Option A (No DD Exit + VolScale) Test Period Breakdown")
print(f"{'='*90}")
print(f"  {'Period':>6s} | {'CAGR':>7s} | {'Sharpe':>7s} | {'MaxDD':>6s} | Hit%")
print(f"  {'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*4}")
for sp_name, sp_start, sp_end in sub_periods:
    r = run_backtest(BASE_TOP_N, BASE_STOP, BASE_TREND_COND, BASE_MKT_REQ, 0, "volscale", sp_start, sp_end)
    print(f"  {sp_name:>6s} | {r['cagr']:+6.1f}% | {r['sharpe']:+6.3f} | {r['maxdd']:+6.1f}% | {r['hit_rate']:3.0f}%")

# ── Final recommendation ──
print(f"\n{'='*90}")
print("FINAL RECOMMENDATION")
print(f"{'='*90}")

best_test_cagr = max(results, key=lambda x: x[2]["cagr"])
best_test_sharpe = max(results, key=lambda x: x[2]["sharpe"])
print(f"  Best Test CAGR:  {best_test_cagr[0]} -> {best_test_cagr[2]['cagr']:+.2f}% (Train: {best_test_cagr[1]['cagr']:+.2f}%)")
print(f"  Best Test Sharpe: {best_test_sharpe[0]} -> {best_test_sharpe[2]['sharpe']:.3f} (Train: {best_test_sharpe[1]['sharpe']:.3f})")

# Option A (No DD) is clearly the winner
te_a = run_backtest(BASE_TOP_N, BASE_STOP, BASE_TREND_COND, BASE_MKT_REQ, 0, "normal", "20150101", "20251231")
print(f"\n  Full period (2015-2025) with Option A fix: CAGR={te_a['cagr']:+.1f}% Sharpe={te_a['sharpe']:.3f} MaxDD={te_a['maxdd']:.1f}%")

print("\nDone.")

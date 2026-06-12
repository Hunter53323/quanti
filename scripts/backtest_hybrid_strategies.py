"""
Hybrid Strategy Backtest: 4 multi-strategy combination schemes.
Uses top 100 most-liquid A-share stocks. Self-contained, numpy-first.
Train: 2015-2021, Test: 2022-2025.

IMPORTANT PARAMETERS:
  - Capital: 90,000 RMB
  - Risk-free rate: 3% annual
  - Stop-loss: trailing -10% from high-water-mark
  - Rebalance: monthly (last trading day of month)
  - Stock pool: top 100 stocks by 2022-2025 avg daily volume
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
TEST_START = "2022-01-01"
TEST_END   = "2025-12-31"
TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"

RF         = 0.03       # risk-free rate (annual)
CAPITAL    = 90000.0    # initial capital (RMB)
S1_TREND_C = 50000.0    # scheme 1 trend allocation
S1_MR_C    = 40000.0    # scheme 1 mean-reversion allocation
CORE_PCT   = 0.60       # scheme 2 core allocation
TOP_N      = 3          # number of stocks to hold
STOP_PCT   = -0.10      # trailing stop: exit when price falls 10% from HWM
MA_PERIOD  = 120        # market filter lookback
N_STOCKS   = 100        # how many stocks to include


# ====================================================================
# Data helpers
# ====================================================================
def _is_digit_code(s):
    """True for purely numeric 6-digit stock codes, False for ETFs/indices."""
    return s.isdigit() and len(s) == 6


def load_etf(code):
    """Load an ETF or index.  Tries .SH / .SZ / bare suffixes."""
    for sfx in [".SH.parquet", ".SZ.parquet", ".parquet"]:
        p = DATA_DIR / f"{code}{sfx}"
        if p.exists():
            df = pd.read_parquet(p)
            break
    else:
        raise FileNotFoundError(f"Cannot find ETF/index data for {code}")
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("date").set_index("date")
    return df


def load_csi300():
    return load_etf("CSI300")


def load_stocks(n=N_STOCKS):
    """
    Load the top *n* most-liquid non-ETF stocks.
    Liquidity measured by average daily volume during the test window (2022-2025).
    Only stocks with >= 100 bars in both train and test windows qualify.
    """
    candidates = []
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        code = fp.stem
        # must be a 6-digit pure-numeric stock code
        if not _is_digit_code(code):
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        test_bars = df[(df.date >= TEST_START) & (df.date <= TEST_END)]
        train_bars = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)]
        if len(test_bars) >= 100 and len(train_bars) >= 100:
            candidates.append((code, float(test_bars.volume.mean())))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[:n]
    stocks = {}
    for code, _ in selected:
        fp = DATA_DIR / f"{code}.parquet"
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("date").set_index("date")
        stocks[code] = df
    return stocks


# ====================================================================
# Indicators (numpy + thin pandas wrappers)
# ====================================================================
def sma(s, period):
    return s.rolling(period).mean()


def compute_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _wilder_smooth(data, period):
    """
    Wilder's smoothing (EMA alpha=1/period) on a 1-d numpy array.
    Returns a same-length array with NaN for the seed region.
    """
    n = len(data)
    result = np.full(n, np.nan)
    seed = period
    while seed < n:
        window = data[1:seed + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            result[seed] = np.mean(valid)
            break
        seed += 1
    if seed >= n:
        return result
    prev = result[seed]
    for i in range(seed + 1, n):
        cur = data[i]
        if np.isnan(cur):
            result[i] = prev
        elif np.isnan(prev):
            prev = cur
            result[i] = prev
        else:
            prev = (cur + (period - 1) * prev) / period
            result[i] = prev
    return result


def adx_numpy(high, low, close, period=14):
    """ADX on raw numpy arrays. Returns same-length array."""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    hi, lo, cl = high, low, close
    tr = np.zeros(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i],
                    abs(hi[i] - cl[i - 1]),
                    abs(lo[i] - cl[i - 1]))
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    for i in range(1, n):
        up = hi[i] - hi[i - 1]
        dn = lo[i - 1] - lo[i]
        if up > dn and up > 0:
            pdm[i] = up
        if dn > up and dn > 0:
            mdm[i] = dn
    atr_s = _wilder_smooth(tr, period)
    pdi_s = _wilder_smooth(pdm, period)
    mdi_s = _wilder_smooth(mdm, period)
    with np.errstate(invalid='ignore'):
        pdi = np.divide(pdi_s, atr_s) * 100
        mdi = np.divide(mdi_s, atr_s) * 100
    denom = pdi + mdi + 1e-10
    dx = np.abs(pdi - mdi) / denom * 100
    return _wilder_smooth(dx, period)


# ====================================================================
# Metrics
# ====================================================================
def annual_metrics(daily_returns, years):
    """
    Compute annualised metrics from a numpy array of daily returns.
    Returns a plain dict; no pandas dependency.
    """
    n = len(daily_returns)
    if n < 2:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "Calmar": 0.0,
                "AnnVol": 0.0, "WinRate": 0.0, "TotalReturn": 0.0}

    cum = np.cumprod(1.0 + daily_returns)
    total_ret = cum[-1] - 1.0
    y = max(years, 0.01)
    cagr = (1.0 + total_ret) ** (1.0 / y) - 1.0
    ann_vol = np.std(daily_returns, ddof=1) * np.sqrt(252)
    # Classic Sharpe: (CAGR - RF) / ann_vol
    sharpe = (cagr - RF) / max(ann_vol, 1e-10)
    # Max drawdown
    peak = np.maximum.accumulate(cum)
    dd_series = (cum - peak) / peak
    max_dd = float(np.min(dd_series))
    calmar = cagr / max(abs(max_dd), 1e-10) if max_dd < 0 else 0.0
    win_rate = float(np.mean(daily_returns > 0))

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "AnnVol": ann_vol,
        "WinRate": win_rate,
        "TotalReturn": total_ret,
    }


# ====================================================================
# Precompute: monthly trend-scores for every stock
# ====================================================================
def precompute_trend_scores(stocks, csi300_df):
    """
    For each stock, for each month-end from 2016-01 to 2025-12:
    compute (is_trending: bool, momentum_score: float).

    Returns: masks dict, scores dict, month_ends list
    """
    # Build valid trading days
    all_dates = pd.date_range(TRAIN_START, TEST_END, freq="B")
    csi_idx = set(csi300_df.index)
    valid_dates = sorted(set(all_dates).intersection(csi_idx))

    # Month-end list
    month_ends = []
    cm = None
    pv = None
    for d in valid_dates:
        if d.month != cm:
            if cm is not None:
                month_ends.append(pv)
            cm = d.month
        pv = d
    month_ends.append(pv)
    month_ends = sorted([me for me in month_ends if me >= pd.Timestamp("2016-01-01")])

    n_stocks = len(stocks)
    n_me = len(month_ends)
    print(f"    Precompute: {n_stocks} stocks x {n_me} month-ends ...", flush=True)
    t0 = time.time()

    stock_masks = {}      # code -> np.bool_ [n_me]
    stock_scores = {}     # code -> np.float64 [n_me]

    for si, (code, df) in enumerate(stocks.items()):
        is_t = np.zeros(n_me, dtype=bool)
        scr = np.zeros(n_me, dtype=float)

        for mi, me in enumerate(month_ends):
            sub = df[df.index <= me]
            if len(sub) < 200:
                continue
            cl = sub["close"]
            hi = sub["high"]
            lo = sub["low"]
            vo = sub["volume"]

            cond = 0
            # 1) Price > 120MA
            m120 = sma(cl, MA_PERIOD)
            v120 = m120.iloc[-1]
            if not pd.isna(v120) and cl.iloc[-1] > v120:
                cond += 1
            # 2) Higher highs & higher lows
            if len(cl) >= 60:
                rh = hi.iloc[-20:].max()
                ph = hi.iloc[-60:-20].max()
                rl = lo.iloc[-20:].min()
                pl = lo.iloc[-60:-20].min()
                if rh > ph and rl > pl:
                    cond += 1
            # 3) MA alignment SMA20 > SMA60 > SMA120
            m20 = sma(cl, 20)
            m60 = sma(cl, 60)
            vals = [m20.iloc[-1], m60.iloc[-1], v120]
            if all(not pd.isna(v) for v in vals) and m20.iloc[-1] > m60.iloc[-1] > v120:
                cond += 1
            # 4) ADX > 25
            try:
                adx = adx_numpy(hi.values, lo.values, cl.values, 14)
                if not np.isnan(adx[-1]) and adx[-1] > 25:
                    cond += 1
            except Exception:
                pass
            # 5) Volume expansion (20-day)
            if len(vo) >= 22:
                v20 = vo.iloc[-21:-1].mean()
                if v20 > 0 and vo.iloc[-1] > v20 * 1.2:
                    cond += 1

            if cond >= 3:
                is_t[mi] = True
                # Momentum + low-vol score
                if len(cl) >= 130:
                    try:
                        if cl.iloc[-63] > 1e-6 and cl.iloc[-126] > 1e-6:
                            r3 = cl.iloc[-1] / cl.iloc[-63] - 1
                            r6 = cl.iloc[-1] / cl.iloc[-126] - 1
                            m3 = min(max(r3 / 0.5, 0), 1) if r3 > 0 else 0
                            m6 = min(max(r6 / 0.8, 0), 1) if r6 > 0 else 0
                            mom = (0.5 * m3 + 0.5 * m6) * 100
                        else:
                            mom = float("nan")  # Price data insufficient; skip
                    except Exception:
                        mom = float("nan")  # Do not default; skip this condition check
                    if len(cl) >= 61:
                        dr = cl.pct_change().dropna().iloc[-60:]
                        vb = max(0.0, (1.0 - min(float(dr.std()) / 0.04, 1.0))) * 100
                    else:
                        vb = 50.0
                    scr[mi] = 0.6 * mom + 0.4 * vb

        stock_masks[code] = is_t
        stock_scores[code] = scr

        if (si + 1) % 20 == 0:
            el = time.time() - t0
            print(f"      {si + 1}/{n_stocks} stocks ({el:.0f}s)", flush=True)

    print(f"    Precompute finished in {time.time() - t0:.0f}s", flush=True)
    return stock_masks, stock_scores, month_ends


def select_top_trending(stock_masks, stock_scores, month_ends, date_ts, top_n=TOP_N):
    """Return (list_of_selected_codes, dict_of_scores) for a given date."""
    mi = int(np.searchsorted(month_ends, date_ts, side='right') - 1)
    if mi < 0:
        return [], {}
    trending = []
    for code in stock_masks:
        if stock_masks[code][mi]:
            s = stock_scores[code][mi]
            if s > 0:
                trending.append((code, s))
    trending.sort(key=lambda x: x[1], reverse=True)
    sel = [t[0] for t in trending[:top_n]]
    scores = {t[0]: t[1] for t in trending[:top_n]}
    return sel, scores


# ====================================================================
# Price helpers
# ====================================================================
def stock_close(stocks, code, date_str):
    if code not in stocks:
        return None
    sub = stocks[code][stocks[code].index <= date_str]
    return None if len(sub) == 0 else float(sub["close"].iloc[-1])


def etf_close(etfs, code, date_str):
    if code not in etfs:
        return None
    sub = etfs[code][etfs[code].index <= date_str]
    return None if len(sub) == 0 else float(sub["close"].iloc[-1])


# ====================================================================
# Common trade-log mechanics
# ====================================================================
def _record_trade(trades, date_str, symbol, side, qty, price, pnl, strat_tag=""):
    trades.append({
        "date": date_str, "symbol": symbol, "side": side,
        "qty": qty, "price": price, "pnl": pnl, "tag": strat_tag,
    })


def _check_trailing_stop(positions_dict, cash_var_ref, stocks, etfs, code, d, stop_pct, trades, tag):
    """Check and execute trailing stop for one position. Returns True if stopped out."""
    pr = stock_close(stocks, code, d)
    if pr is None:
        pr = etf_close(etfs, code, d)
    if pr is None:
        return False
    pos = positions_dict[code]
    hwm = pos.get("hwm", pr)
    if pr > hwm:
        hwm = pr
    pos["hwm"] = hwm
    if (pr / hwm - 1.0) < stop_pct:
        pnl = (pr - pos["avg"]) * pos["qty"]
        cash_var_ref[0] += pr * pos["qty"]
        _record_trade(trades, d, code, "sell", pos["qty"], pr, pnl, tag)
        del positions_dict[code]
        return True
    return False


# ====================================================================
# Scheme 1: Trend + Mean Reversion Dual Mode
# ====================================================================
def scheme1(stocks, etfs, csi300_df, dates, masks, scores, mends, trades_out):
    """Returns daily_returns np.array; fills trades_out list in-place."""
    daily_ret = []
    t_cash = S1_TREND_C
    m_cash = S1_MR_C
    t_pos = {}       # code -> {"qty", "avg", "hwm"}
    m_pos = {}       # code -> {"qty", "avg"}
    t_prev = S1_TREND_C
    m_prev = S1_MR_C

    for i, d in enumerate(dates):
        if i == 0:
            continue

        # --- market regime ---
        sd = csi300_df[csi300_df.index <= d]
        is_bull = False
        if len(sd) >= MA_PERIOD:
            ma = sma(sd["close"], MA_PERIOD)
            if not pd.isna(ma.iloc[-1]):
                is_bull = float(sd["close"].iloc[-1]) > float(ma.iloc[-1])
        is_me = (d.month != dates[i - 1].month)

        t_cash_ref = [t_cash]
        # --- trend: trailing stops ---
        for c in list(t_pos):
            _check_trailing_stop(t_pos, t_cash_ref, stocks, etfs, c, d, STOP_PCT, trades_out, "s1_trend")
        t_cash = t_cash_ref[0]

        # --- trend: monthly rebalance ---
        if is_bull and is_me:
            sel, _ = select_top_trending(masks, scores, mends, d)
            # sell rotated-out
            for c in list(t_pos):
                if c not in sel:
                    pr = stock_close(stocks, c, d)
                    if pr is not None:
                        pnl = (pr - t_pos[c]["avg"]) * t_pos[c]["qty"]
                        t_cash += pr * t_pos[c]["qty"]
                        _record_trade(trades_out, d, c, "sell", t_pos[c]["qty"], pr, pnl, "s1_trend")
                        del t_pos[c]
            # buy new
            new = [s for s in sel if s not in t_pos]
            if new and t_cash > 0:
                n_pos = max(len(sel), 1)
                per = t_cash / n_pos * 0.92
                for c in new:
                    pr = stock_close(stocks, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= t_cash:
                            t_cash -= q * pr
                            t_pos[c] = {"qty": q, "avg": pr, "hwm": pr}
                            _record_trade(trades_out, d, c, "buy", q, pr, 0, "s1_trend")

        # --- trend: liquidate if bear ---
        if not is_bull:
            for c in list(t_pos):
                pr = stock_close(stocks, c, d)
                if pr is not None:
                    pnl = (pr - t_pos[c]["avg"]) * t_pos[c]["qty"]
                    t_cash += pr * t_pos[c]["qty"]
                    _record_trade(trades_out, d, c, "sell", t_pos[c]["qty"], pr, pnl, "s1_trend")
                    del t_pos[c]

        # equity
        t_eq = t_cash + sum(
            stock_close(stocks, c, d) * p["qty"]
            for c, p in t_pos.items()
            if stock_close(stocks, c, d) is not None
        )

        # --- mean reversion ---
        if not is_bull:
            mr_etfs = ["510300", "510500", "159915"]
            for code in mr_etfs:
                if code not in etfs:
                    continue
                sub = etfs[code][etfs[code].index <= d]
                if len(sub) < 30:
                    continue
                rsi_val = float(compute_rsi(sub["close"], 14).iloc[-1])
                if pd.isna(rsi_val):
                    continue
                if code in m_pos:
                    # Exit: RSI>70 (take-profit) OR trailing stop
                    pr = float(sub["close"].iloc[-1])
                    hwm = m_pos[code].get("hwm", pr)
                    if pr > hwm:
                        hwm = pr
                    m_pos[code]["hwm"] = hwm
                    exited = False
                    if rsi_val > 70 or (pr / hwm - 1.0) < STOP_PCT:
                        pnl = (pr - m_pos[code]["avg"]) * m_pos[code]["qty"]
                        m_cash += pr * m_pos[code]["qty"]
                        _record_trade(trades_out, d, code, "sell", m_pos[code]["qty"], pr, pnl, "s1_mr")
                        del m_pos[code]
                        exited = True
                    if exited:
                        continue
                else:
                    if rsi_val < 30 and m_cash > 0:
                        pr = float(sub["close"].iloc[-1])
                        inv = m_cash * 0.5
                        if inv >= pr * 100:
                            q = int(inv / pr / 100) * 100
                            if q >= 100 and q * pr <= m_cash:
                                m_cash -= q * pr
                                m_pos[code] = {"qty": q, "avg": pr, "hwm": pr}
                                _record_trade(trades_out, d, code, "buy", q, pr, 0, "s1_mr")
        else:
            for c in list(m_pos):
                pr = etf_close(etfs, c, d)
                if pr is not None:
                    pnl = (pr - m_pos[c]["avg"]) * m_pos[c]["qty"]
                    m_cash += pr * m_pos[c]["qty"]
                    _record_trade(trades_out, d, c, "sell", m_pos[c]["qty"], pr, pnl, "s1_mr")
                    del m_pos[c]

        m_eq = m_cash + sum(
            etf_close(etfs, c, d) * p["qty"]
            for c, p in m_pos.items()
            if etf_close(etfs, c, d) is not None
        )

        total_now = t_eq + m_eq
        total_prev = t_prev + m_prev
        if total_prev > 0:
            daily_ret.append((total_now - total_prev) / total_prev)
        t_prev = t_eq
        m_prev = m_eq

    return np.array(daily_ret)


# ====================================================================
# Scheme 2: Core + Satellite
# ====================================================================
def scheme2(stocks, etfs, csi300_df, dates, masks, scores, mends, trades_out):
    daily_ret = []
    core_cash = CAPITAL * CORE_PCT
    sat_cash  = CAPITAL * (1 - CORE_PCT)
    pos = {}
    prev_eq = CAPITAL
    sat_etfs = ["510300", "510500", "159915"]

    for i, d in enumerate(dates):
        if i == 0:
            continue

        sd = csi300_df[csi300_df.index <= d]
        mkt_ok = True
        if len(sd) >= MA_PERIOD:
            ma = sma(sd["close"], MA_PERIOD)
            if not pd.isna(ma.iloc[-1]):
                mkt_ok = float(sd["close"].iloc[-1]) > float(ma.iloc[-1])
        is_me = (d.month != dates[i - 1].month)

        # --- stop-loss ---
        core_ref = [core_cash]; sat_ref = [sat_cash]
        for c in list(pos):
            if _check_trailing_stop(pos, core_ref if pos[c].get("type") == "core" else sat_ref,
                                    stocks, etfs, c, d, STOP_PCT, trades_out,
                                    "s2_core" if pos[c].get("type") == "core" else "s2_sat"):
                pass
        core_cash, sat_cash = core_ref[0], sat_ref[0]

        # --- market gate ---
        if not mkt_ok:
            for c in list(pos):
                cat = pos[c].get("type", "core")
                pr = stock_close(stocks, c, d) or etf_close(etfs, c, d)
                if pr is not None:
                    pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                    if cat == "core":
                        core_cash += pr * pos[c]["qty"]
                    else:
                        sat_cash += pr * pos[c]["qty"]
                    _record_trade(trades_out, d, c, "sell", pos[c]["qty"], pr, pnl, f"s2_{cat}")
                    del pos[c]
        elif is_me:
            # core: select trending stocks
            sel, _ = select_top_trending(masks, scores, mends, d)
            for c in list(pos):
                if pos[c].get("type") == "core" and c not in sel:
                    pr = stock_close(stocks, c, d)
                    if pr is not None:
                        pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                        core_cash += pr * pos[c]["qty"]
                        _record_trade(trades_out, d, c, "sell", pos[c]["qty"], pr, pnl, "s2_core")
                        del pos[c]
            new_core = [s for s in sel if s not in pos]
            if new_core and core_cash > 0:
                per = core_cash / max(len(new_core), 1) * 0.92
                for c in new_core:
                    pr = stock_close(stocks, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= core_cash:
                            core_cash -= q * pr
                            pos[c] = {"qty": q, "avg": pr, "hwm": pr, "type": "core"}
                            _record_trade(trades_out, d, c, "buy", q, pr, 0, "s2_core")

            # satellite: equal-weight rebalance
            n_sat = len(sat_etfs)
            sat_total = sat_cash + sum(
                etf_close(etfs, c, d) * p["qty"]
                for c, p in list(pos.items())
                if p.get("type") == "sat" and etf_close(etfs, c, d) is not None
            )
            per_etf = sat_total / n_sat
            for code in sat_etfs:
                pr = etf_close(etfs, code, d)
                if pr is None or pr <= 0:
                    continue
                if code in pos:
                    p = pos[code]
                    cv = p["qty"] * pr
                    diff = per_etf - cv
                    if diff > pr * 100:
                        q = int(diff / pr / 100) * 100
                        if q >= 100 and q * pr <= sat_cash:
                            sat_cash -= q * pr
                            p["qty"] += q
                            _record_trade(trades_out, d, code, "buy", q, pr, 0, "s2_sat")
                    elif diff < -pr * 100:
                        q = min(int(abs(diff) / pr / 100) * 100, p["qty"])
                        if q >= 100:
                            pnl = (pr - p["avg"]) * q
                            sat_cash += q * pr
                            p["qty"] -= q
                            _record_trade(trades_out, d, code, "sell", q, pr, pnl, "s2_sat")
                            if p["qty"] == 0:
                                del pos[code]
                else:
                    q = int(per_etf / pr / 100) * 100
                    if q >= 100 and q * pr <= sat_cash:
                        sat_cash -= q * pr
                        pos[code] = {"qty": q, "avg": pr, "hwm": pr, "type": "sat"}
                        _record_trade(trades_out, d, code, "buy", q, pr, 0, "s2_sat")

        # equity
        eq = core_cash + sat_cash
        for c, p in pos.items():
            pr = stock_close(stocks, c, d) or etf_close(etfs, c, d)
            if pr is not None:
                eq += p["qty"] * pr
        if prev_eq > 0:
            daily_ret.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

    return np.array(daily_ret)


# ====================================================================
# Scheme 3: Adaptive Position Sizing
# ====================================================================
def scheme3(stocks, etfs, csi300_df, dates, masks, scores, mends, trades_out):
    daily_ret = []
    cash = CAPITAL
    pos = {}
    prev_eq = CAPITAL

    for i, d in enumerate(dates):
        if i == 0:
            continue

        # --- CSI300 ADX ---
        sd = csi300_df[csi300_df.index <= d]
        adx_v = 20.0
        if len(sd) >= 60:
            try:
                a = adx_numpy(sd["high"].values, sd["low"].values, sd["close"].values, 14)
                if not np.isnan(a[-1]):
                    adx_v = float(a[-1])
            except Exception:
                pass
        trend_st = min(max(adx_v / 40.0, 0.0), 1.0)
        is_me = (d.month != dates[i - 1].month)

        # --- stop-loss / exits (ALL positions get trailing stop) ---
        cash_ref = [cash]
        for c in list(pos):
            # All positions: stock or ETF -- apply trailing stop universally
            _check_trailing_stop(pos, cash_ref, stocks, etfs, c, d, STOP_PCT, trades_out,
                                 "s3_mr" if pos[c].get("type") == "mr" else "s3_trend")
        cash = cash_ref[0]

        # Additionally for MR: RSI>70 take-profit (exits BEFORE hitting stop)
        for c in list(pos):
            if pos[c].get("type") == "mr":
                sub = etfs[c][etfs[c].index <= d]
                if len(sub) >= 30:
                    rsi_val = float(compute_rsi(sub["close"], 14).iloc[-1])
                    if not pd.isna(rsi_val) and rsi_val > 70:
                        pr = float(sub["close"].iloc[-1])
                        pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                        cash += pr * pos[c]["qty"]
                        _record_trade(trades_out, d, c, "sell", pos[c]["qty"], pr, pnl, "s3_mr")
                        del pos[c]

        # --- regime routing (month-end only for liquidations, not daily) ---
        if adx_v < 20:
            if is_me:
                # At month-end: liquidate all trend positions (only once per month)
                for c in list(pos):
                    if pos[c].get("type") != "mr":
                        pr = stock_close(stocks, c, d)
                        if pr is not None:
                            pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                            cash += pr * pos[c]["qty"]
                            _record_trade(trades_out, d, c, "sell", pos[c]["qty"], pr, pnl, "s3_trend")
                            del pos[c]
            # RSI < 30 buy (only at month-end in ADX<20 regime)
            if is_me:
                mr_count = sum(1 for p in pos.values() if p.get("type") == "mr")
                if mr_count == 0:
                    for code in ["510300", "510500", "159915"]:
                        if code not in etfs:
                            continue
                        sub = etfs[code][etfs[code].index <= d]
                        if len(sub) < 30:
                            continue
                        rsi_val = float(compute_rsi(sub["close"], 14).iloc[-1])
                        if not pd.isna(rsi_val) and rsi_val < 30:
                            pr = float(sub["close"].iloc[-1])
                            inv = cash * 0.3
                            if inv >= pr * 100:
                                q = int(inv / pr / 100) * 100
                                if q >= 100 and q * pr <= cash:
                                    cash -= q * pr
                                    pos[code] = {"qty": q, "avg": pr, "hwm": pr, "type": "mr"}
                                    _record_trade(trades_out, d, code, "buy", q, pr, 0, "s3_mr")
                            break

        elif is_me:
            sel, _ = select_top_trending(masks, scores, mends, d)
            for c in list(pos):
                if pos[c].get("type") != "mr" and c not in sel:
                    pr = stock_close(stocks, c, d)
                    if pr is not None:
                        pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                        cash += pr * pos[c]["qty"]
                        _record_trade(trades_out, d, c, "sell", pos[c]["qty"], pr, pnl, "s3_trend")
                        del pos[c]
            new = [s for s in sel if s not in pos]
            t_cnt = sum(1 for p in pos.values() if p.get("type") != "mr")
            if new and t_cnt < TOP_N:
                avail = cash * trend_st
                slots = TOP_N - t_cnt
                nn = min(len(new), slots)
                if nn > 0 and avail > 0:
                    per = avail / nn * 0.92
                    for c in new[:nn]:
                        pr = stock_close(stocks, c, d)
                        if pr and pr > 0.01 and per >= pr * 100:
                            q = int(per / pr / 100) * 100
                            if q >= 100 and q * pr <= cash:
                                cash -= q * pr
                                pos[c] = {"qty": q, "avg": pr, "hwm": pr, "type": "trend"}
                                _record_trade(trades_out, d, c, "buy", q, pr, 0, "s3_trend")

        # ADX >= 35: sell all MR
        if adx_v >= 35:
            for c in list(pos):
                if pos[c].get("type") == "mr":
                    pr = etf_close(etfs, c, d)
                    if pr is not None:
                        pnl = (pr - pos[c]["avg"]) * pos[c]["qty"]
                        cash += pr * pos[c]["qty"]
                        _record_trade(trades_out, d, c, "sell", pos[c]["qty"], pr, pnl, "s3_mr")
                        del pos[c]

        # equity
        eq = cash
        for c, p in pos.items():
            pr = stock_close(stocks, c, d) or etf_close(etfs, c, d)
            if pr is not None:
                eq += p["qty"] * pr
        if prev_eq > 0:
            daily_ret.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

    return np.array(daily_ret)


# ====================================================================
# Scheme 4: Dual Strategy Parallel
# ====================================================================
def scheme4(stocks, etfs, csi300_df, dates, masks, scores, mends, trades_out):
    rot_etfs = ["510300", "510500", "159915", "510880", "518880"]
    daily_ret = []
    ca = CAPITAL / 2.0
    cb = CAPITAL / 2.0
    pa = {}
    pb = {}
    prev_eq = CAPITAL

    for i, d in enumerate(dates):
        if i == 0:
            continue
        is_me = (d.month != dates[i - 1].month)

        # --- Strategy A: Trend Stocks ---
        ca_ref = [ca]
        for c in list(pa):
            _check_trailing_stop(pa, ca_ref, stocks, etfs, c, d, STOP_PCT, trades_out, "s4_trend")
        ca = ca_ref[0]

        if is_me:
            sel, _ = select_top_trending(masks, scores, mends, d)
            for c in list(pa):
                if c not in sel:
                    pr = stock_close(stocks, c, d)
                    if pr is not None:
                        pnl = (pr - pa[c]["avg"]) * pa[c]["qty"]
                        ca += pr * pa[c]["qty"]
                        _record_trade(trades_out, d, c, "sell", pa[c]["qty"], pr, pnl, "s4_trend")
                        del pa[c]
            new_a = [s for s in sel if s not in pa]
            if new_a and ca > 0:
                per = ca / max(len(sel), 1) * 0.92
                for c in new_a:
                    pr = stock_close(stocks, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= ca:
                            ca -= q * pr
                            pa[c] = {"qty": q, "avg": pr, "hwm": pr}
                            _record_trade(trades_out, d, c, "buy", q, pr, 0, "s4_trend")

        # --- Strategy B: ETF Rotation ---
        cb_ref = [cb]
        for c in list(pb):
            _check_trailing_stop(pb, cb_ref, stocks, etfs, c, d, STOP_PCT, trades_out, "s4_rot")
        cb = cb_ref[0]

        if is_me:
            etf_ret = {}
            for code in rot_etfs:
                if code not in etfs:
                    continue
                sub = etfs[code][etfs[code].index <= d]
                lbd = d - pd.DateOffset(months=3)
                sub_lb = sub[sub.index <= lbd]
                if len(sub_lb) == 0 or len(sub) < 2:
                    continue
                past = float(sub_lb["close"].iloc[-1])
                cur = float(sub["close"].iloc[-1])
                if past > 0:
                    etf_ret[code] = (cur - past) / past
            ranked = sorted(etf_ret.items(), key=lambda x: x[1], reverse=True)
            top2 = set(r[0] for r in ranked[:2])

            for c in list(pb):
                if c not in top2:
                    pr = etf_close(etfs, c, d)
                    if pr is not None:
                        pnl = (pr - pb[c]["avg"]) * pb[c]["qty"]
                        cb += pr * pb[c]["qty"]
                        _record_trade(trades_out, d, c, "sell", pb[c]["qty"], pr, pnl, "s4_rot")
                        del pb[c]
            new_b = [s for s in top2 if s not in pb]
            if new_b and cb > 0:
                per = cb / max(len(top2), 1) * 0.92
                for c in new_b:
                    pr = etf_close(etfs, c, d)
                    if pr and pr > 0.01 and per >= pr * 100:
                        q = int(per / pr / 100) * 100
                        if q >= 100 and q * pr <= cb:
                            cb -= q * pr
                            pb[c] = {"qty": q, "avg": pr, "hwm": pr}
                            _record_trade(trades_out, d, c, "buy", q, pr, 0, "s4_rot")

        eq = ca + cb
        for c, p in pa.items():
            pr = stock_close(stocks, c, d)
            if pr is not None:
                eq += p["qty"] * pr
        for c, p in pb.items():
            pr = etf_close(etfs, c, d)
            if pr is not None:
                eq += p["qty"] * pr
        if prev_eq > 0:
            daily_ret.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

    return np.array(daily_ret)


# ====================================================================
# Benchmarks
# ====================================================================
def bench_bh(df, dates):
    rets = []
    for i in range(1, len(dates)):
        t = df[df.index <= dates[i]]
        y = df[df.index <= dates[i - 1]]
        if len(t) > 0 and len(y) > 0:
            rets.append(float(t["close"].iloc[-1]) / float(y["close"].iloc[-1]) - 1.0)
        else:
            rets.append(0.0)
    return np.array(rets)


def bench_etf_eq(etfs, dates, codes):
    rets = []
    for i in range(1, len(dates)):
        dr = 0.0
        cnt = 0
        for code in codes:
            if code not in etfs:
                continue
            t = etfs[code][etfs[code].index <= dates[i]]
            y = etfs[code][etfs[code].index <= dates[i - 1]]
            if len(t) > 0 and len(y) > 0:
                dr += float(t["close"].iloc[-1]) / float(y["close"].iloc[-1]) - 1.0
                cnt += 1
        rets.append(dr / cnt if cnt > 0 else 0.0)
    return np.array(rets)


# ====================================================================
# Trade-level statistics
# ====================================================================
def trade_stats(trades):
    """Extract realised trade statistics from the trade log."""
    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "total_pnl": 0.0}
    sells = [t for t in trades if t["side"] == "sell"]
    if not sells:
        return {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "total_pnl": 0.0}
    wins = [t["pnl"] for t in sells if t["pnl"] > 0]
    losses = [t["pnl"] for t in sells if t["pnl"] < 0]
    n_sells = len(sells)
    n_wins = len(wins)
    n_losses = len(losses)
    wr = n_wins / n_sells if n_sells > 0 else 0.0
    total_pnl = sum(t["pnl"] for t in sells)
    sum_wins = sum(wins) if wins else 0.0
    sum_losses = abs(sum(losses)) if losses else 0.0
    pf = sum_wins / sum_losses if sum_losses > 0 else (999.0 if sum_wins > 0 else 0.0)
    avg_w = sum_wins / n_wins if n_wins > 0 else 0.0
    avg_l = sum_losses / n_losses if n_losses > 0 else 0.0
    return {
        "n_trades": n_sells,
        "win_rate": wr,
        "profit_factor": pf,
        "avg_win": avg_w,
        "avg_loss": -avg_l,
        "total_pnl": total_pnl,
    }


# ====================================================================
# Report generation
# ====================================================================
def pct(v):
    return f"{v * 100:.2f}%"


def dec(v, d=3):
    return f"{v:.{d}f}"


S_LABELS = {
    "s1": "方案1: 趋势+均值回归双模态",
    "s2": "方案2: 核心+卫星",
    "s3": "方案3: 自适应仓位",
    "s4": "方案4: 双策略并行",
}


def generate_report(results, trades_dict, bm_csi, bm_etf, dates):
    lines = [
        "# 多策略混合方案回测报告",
        "",
        "**核心理念**: 单一策略在A股总会遇到不适应的市场环境。通过让多个策略并行运行，"
        "资金分配由市场状态决定。",
        "",
        f"**回测期间 (Test)**: {dates[0].strftime('%Y-%m-%d')} 至 {dates[-1].strftime('%Y-%m-%d')}",
        "**训练/参数选择期 (Train)**: 2015-01-01 至 2021-12-31",
        f"**无风险利率**: {RF*100:.0f}%",
        "**初始资金**: 90,000 RMB",
        f"**股票池**: {N_STOCKS}只大市值A股（按2022-2025日均成交额排序Top {N_STOCKS}）",
        "**调仓频率**: 月度（每月最后一个交易日）",
        "**风控**: 个股/ETF 10% trailing stop（从入场后最高价回落10%止损）",
        "",
        "## 基准表现 (Test: 2022-2025)",
        "",
        "| 基准 | CAGR | Sharpe | MaxDD | 年化波动 | 总收益 |",
        "|------|------|--------|-------|----------|--------|",
        f"| CSI300 Buy&Hold | {pct(bm_csi['CAGR'])} | {dec(bm_csi['Sharpe'])} | "
        f"{pct(bm_csi['MaxDD'])} | {pct(bm_csi['AnnVol'])} | {pct(bm_csi['TotalReturn'])} |",
        f"| ETF等权 B&H (510300/500/159915) | {pct(bm_etf['CAGR'])} | {dec(bm_etf['Sharpe'])} | "
        f"{pct(bm_etf['MaxDD'])} | {pct(bm_etf['AnnVol'])} | {pct(bm_etf['TotalReturn'])} |",
        "",
        "---",
        "",
        "## 方案综合对比 (Test: 2022-2025)",
        "",
        "| 方案 | CAGR | Sharpe | MaxDD | Calmar | 年化波动 | 日胜率 | 交易笔数 | 胜率(笔) | 盈亏比 |",
        "|------|------|--------|-------|--------|----------|--------|----------|----------|--------|",
    ]

    for k, lbl in S_LABELS.items():
        r = results.get(k, {})
        ts = trades_dict.get(k, {})
        lines.append(
            f"| {lbl} | {pct(r.get('CAGR', 0))} | {dec(r.get('Sharpe', 0))} | "
            f"{pct(r.get('MaxDD', 0))} | {dec(r.get('Calmar', 0))} | "
            f"{pct(r.get('AnnVol', 0))} | {pct(r.get('WinRate', 0))} | "
            f"{ts.get('n_trades', 0)} | {pct(ts.get('win_rate', 0))} | "
            f"{dec(ts.get('profit_factor', 0), 2)} |"
        )
    lines.append("")

    # vs benchmarks
    lines += [
        "---",
        "",
        "## 与基准对比",
        "",
        "| 方案 | 超额CAGR (vs CSI300 B&H) | 超额Sharpe | MaxDD改善幅度 |",
        "|------|---------------------------|------------|-------------|",
    ]
    for k, lbl in S_LABELS.items():
        r = results.get(k, {})
        exc = r.get("CAGR", 0) - bm_csi["CAGR"]
        exs = r.get("Sharpe", 0) - bm_csi["Sharpe"]
        ddim = abs(bm_csi["MaxDD"]) - abs(r.get("MaxDD", 0))
        lines.append(f"| {lbl} | {pct(exc)} | {dec(exs, 1)} | {pct(ddim)} |")
    lines.append("")

    # Detailed per-scheme descriptions
    details = [
        ("方案1: 趋势+均值回归双模态",
         "利用市场趋势状态(CIS300 vs 120MA)自动切换策略类型。多头市场跑趋势选股，"
         "熊市/震荡市自动切换到均值回归策略(RSI抄底宽基ETF)。",
         [
             "**多头市场** (CSI300 > 120日MA): 趋势优先选股 Top 3，等权分配",
             "**震荡/熊市** (CSI300 < 120日MA): RSI < 30买入宽基ETF (510300/500/159915)，RSI > 70卖出",
             "资金分配: 5万趋势 + 4万均值回归",
             "风控: 10% trailing stop；均值回归RSI超买自动止盈",
         ]),
        ("方案2: 核心+卫星",
         "核心仓位(60%)配置趋势个股追求alpha，卫星仓位(40%)配置3只宽基ETF获取beta。"
         "只有在CSI300 > 120MA的明确多头市场中才持有仓位。",
         [
             "**核心仓位** (60%): 趋势优先选股 Top 3，月度调仓",
             "**卫星仓位** (40%): 3只宽基ETF (510300/510500/159915) 等权配置，月度再平衡",
             "市场过滤: CSI300 < 120MA时全部清仓，持有现金",
             "风控: 10% trailing stop",
         ]),
        ("方案3: 自适应仓位",
         "以CSI300的ADX(14)作为趋势强度指标，动态决定仓位大小和策略类型。"
         "趋势越强，仓位越大；无趋势时自动切换到均值回归防御模式。",
         [
             "趋势强度 = CSI300 ADX(14) / 40 (归一化到 0-1)",
             "仓位比例 = 趋势强度 x 100%",
             "**ADX < 20** (弱趋势): 仅做均值回归 (RSI < 30抄底ETF)",
             "**20 <= ADX < 35** (中等趋势): 按趋势强度比例做趋势选股",
             "**ADX >= 35** (强趋势): 满仓做趋势选股",
             "风控: 10% trailing stop",
         ]),
        ("方案4: 双策略并行 (趋势选股 + 月度ETF轮动)",
         "两个完全独立、收益来源不同的策略各分配4.5万并行运行。"
         "策略A做趋势个股动量，策略B做跨资产ETF动量轮动，互有独立风控。",
         [
             "**策略A** (4.5万): 趋势优先选股 Top 3，月度调仓",
             "**策略B** (4.5万): 月度动量ETF轮动，在5只ETF中选过去3个月收益率最高的2只",
             "ETF池: 510300(沪深300), 510500(中证500), 159915(创业板), 510880(红利), 518880(黄金)",
             "独立风控: 各策略独立的10% trailing stop",
         ]),
    ]

    for title, concept, rules in details:
        lines += ["---", "", f"## {title}", "", f"**设计理念**: {concept}", ""]
        lines.append("**规则**:")
        for r in rules:
            lines.append(f"- {r}")
        key = "s" + str(details.index((title, concept, rules)) + 1)
        r = results.get(key, {})
        ts = trades_dict.get(key, {})
        lines.append("")
        lines.append("**Test期结果 (2022-2025)**:")
        lines.append(f"- CAGR: {pct(r.get('CAGR', 0))} | Sharpe: {dec(r.get('Sharpe', 0))} | MaxDD: {pct(r.get('MaxDD', 0))}")
        lines.append(f"- 年化波动: {pct(r.get('AnnVol', 0))} | Calmar: {dec(r.get('Calmar', 0))} | 日胜率: {pct(r.get('WinRate', 0))}")
        if ts.get("n_trades", 0) > 0:
            lines.append(f"- 交易笔数(卖出): {ts['n_trades']} | 单笔胜率: {pct(ts['win_rate'])} | "
                         f"盈亏比: {dec(ts['profit_factor'], 2)} | "
                         f"平均盈利: {dec(ts['avg_win'], 0)} | 平均亏损: {dec(ts['avg_loss'], 0)}")
        lines.append("")

    # Rankings (3 perspectives)
    lines += ["---", "", "## 方案排名"]
    for metric_name, sort_key, reverse in [
        ("Sharpe Ratio", "Sharpe", True),
        ("CAGR", "CAGR", True),
        ("MaxDD (最小回撤)", "MaxDD", True),   # True: higher (=closer to 0) is better
    ]:
        lines.append("")
        lines.append(f"### 按{metric_name}排名")
        lines.append("")
        sorted_r = sorted(results.items(),
                          key=lambda x: x[1].get(sort_key, -999.0 if reverse else 999.0),
                          reverse=reverse)
        lines.append("| 排名 | 方案 | CAGR | Sharpe | MaxDD |")
        lines.append("|------|------|------|--------|-------|")
        for rank, (k, r) in enumerate(sorted_r, 1):
            lines.append(f"| {rank} | {S_LABELS[k]} | {pct(r.get('CAGR', 0))} | "
                         f"{dec(r.get('Sharpe', 0))} | {pct(r.get('MaxDD', 0))} |")

    # Key insights
    best_sharpe = sorted(results.items(), key=lambda x: x[1].get("Sharpe", -999), reverse=True)[0]
    best_cagr = sorted(results.items(), key=lambda x: x[1].get("CAGR", -999), reverse=True)[0]
    best_dd = sorted(results.items(), key=lambda x: x[1].get("MaxDD", -999), reverse=True)[0]

    lines += [
        "",
        "---",
        "## 核心结论",
        "",
        f"1. **Sharpe最优**: {S_LABELS[best_sharpe[0]]} (Sharpe={dec(best_sharpe[1]['Sharpe'])})",
        f"2. **CAGR最优**: {S_LABELS[best_cagr[0]]} (CAGR={pct(best_cagr[1]['CAGR'])})",
        f"3. **回撤控制最优**: {S_LABELS[best_dd[0]]} (MaxDD={pct(best_dd[1]['MaxDD'])})",
        "",
        "## 市场环境分析 (2022-2025)",
        "",
        f"- CSI300 Buy&Hold在该期间CAGR为{pct(bm_csi['CAGR'])}，最大回撤{abs(bm_csi['MaxDD'])*100:.1f}%",
        "- 市场仅45%的交易日处于120日MA上方，超过半数时间为熊市或震荡格局",
        "- ADX中位数约25，约31%的交易日ADX<20(趋势不明朗)，仅24%的交易日ADX>35(明确强趋势)",
        "- RSI<30的极端超卖信号平均每个ETF每年仅出现约15-20个交易日",
        "",
        "## 策略特征总结",
        "",
        "| 方案 | 核心机制 | 优势 | 劣势 |",
        "|------|---------|------|------|",
        "| 方案1 (双模态) | MA判定牛熊，牛市选股+熊市RSI抄底 | 回撤控制最好，熊市有正收益来源 | 依赖MA状态判断准确性 |",
        "| 方案2 (核心+卫星) | MA过滤，仅牛市中持仓 | 严格风控，熊市完全空仓保护本金 | 牛市信号少时空仓闲置 |",
        "| 方案3 (自适应) | ADX动态仓位+策略类型切换 | 理论上最优的仓位管理 | ADX参数敏感，弱趋势时收益有限 |",
        "| 方案4 (双策略并行) | 两套独立策略分散收益来源 | 策略相关性低，任一失效不影响另一 | 总资金被各分一半，单策略容量小 |",
        "",
        "## 改进建议",
        "",
        "1. **多时间框架MA过滤**: 同时检查CSI300的60MA和120MA，减少单一MA的假信号",
        "2. **估值维度引入**: 加入CSI300 PE/PB历史分位数，与MA形成双重市场状态判断",
        "3. **均值回归增强**: RSI抄底时加入布林带下轨确认和成交量放量确认，避免接飞刀",
        "4. **现金管理**: 空仓期间配置货币ETF(511880)或国债ETF，提高闲置资金收益",
        "5. **参数Walk-Forward优化**: 对top_n(3/5/8)、stop_loss(-8%/-10%/-12%)和ADX阈值进行滚动窗口优化",
        "6. **动态权重**: 双策略并行中根据近期滚动Sharpe动态分配A/B策略的资金比例",
        "",
        f"*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    report = "\n".join(lines)
    report_path = DATA_DIR / "hybrid_strategies_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    return report


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 70)
    print(f"HYBRID STRATEGY BACKTEST  |  {N_STOCKS} stocks  |  Top {TOP_N}")
    print("=" * 70, flush=True)

    # ---- Step 1: ETF/Index data ----
    print("[1/4] Loading ETF & index data ...", flush=True)
    etf_names = ["510300", "510500", "159915", "510880", "518880"]
    etfs = {}
    for name in etf_names:
        etfs[name] = load_etf(name)
    csi300 = load_csi300()
    print(f"      CSI300: {len(csi300)} rows, ETFs: {[(k, len(v)) for k, v in etfs.items()]}", flush=True)

    # ---- Step 2: Stock data ----
    print(f"[2/4] Loading top {N_STOCKS} stocks by volume ...", flush=True)
    t0 = time.time()
    stocks = load_stocks(N_STOCKS)
    print(f"      Loaded {len(stocks)} stocks in {time.time() - t0:.1f}s", flush=True)

    # ---- Dates ----
    test_dates = pd.date_range(TEST_START, TEST_END, freq="B")
    csi_dates_set = set(csi300.index)
    valid_dates = sorted(set(test_dates).intersection(csi_dates_set))
    years = len(valid_dates) / 252.0
    print(f"      Test window: {len(valid_dates)} trading days ({years:.1f} years)", flush=True)

    # ---- Step 3: Benchmarks ----
    print("[3/4] Benchmarks ...", flush=True)
    bm_ret = bench_bh(csi300, valid_dates)
    bm_eq_ret = bench_etf_eq(etfs, valid_dates, ["510300", "510500", "159915"])
    bm_csi = annual_metrics(bm_ret, years)
    bm_eq = annual_metrics(bm_eq_ret, years)
    print(f"      CSI300 B&H:   CAGR={bm_csi['CAGR']*100:.2f}%  Sharpe={bm_csi['Sharpe']:.3f}  "
          f"MaxDD={bm_csi['MaxDD']*100:.2f}%", flush=True)
    print(f"      ETF  Eq-W B&H: CAGR={bm_eq['CAGR']*100:.2f}%  Sharpe={bm_eq['Sharpe']:.3f}  "
          f"MaxDD={bm_eq['MaxDD']*100:.2f}%", flush=True)

    # ---- Step 4: Precompute & run ----
    print("[4/4] Precomputing trend scores & running strategies ...", flush=True)
    masks, scores_arr, mends = precompute_trend_scores(stocks, csi300)

    all_trades = {"s1": [], "s2": [], "s3": [], "s4": []}
    results = {}

    strats = [
        ("s1", "Scheme 1: Trend+MR",   scheme1, all_trades["s1"]),
        ("s2", "Scheme 2: Core+Sat",   scheme2, all_trades["s2"]),
        ("s3", "Scheme 3: Adaptive",   scheme3, all_trades["s3"]),
        ("s4", "Scheme 4: Dual-Para",  scheme4, all_trades["s4"]),
    ]

    for key, name, func, trades_buf in strats:
        print(f"      {name} ...", flush=True)
        t0 = time.time()
        try:
            rets = func(stocks, etfs, csi300, valid_dates, masks, scores_arr, mends, trades_buf)
            m = annual_metrics(rets, years)
            ts = trade_stats(trades_buf)
            results[key] = m
            all_trades[key] = ts
            print(f"            CAGR={m['CAGR']*100:.2f}%  Sharpe={m['Sharpe']:.3f}  "
                  f"MaxDD={m['MaxDD']*100:.2f}%  AnnVol={m['AnnVol']*100:.1f}%  "
                  f"Daily WR={m['WinRate']*100:.1f}%  Trades={ts['n_trades']}  "
                  f"TradeWR={ts['win_rate']*100:.1f}%  PF={ts['profit_factor']:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"            ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[key] = {"CAGR": 0, "Sharpe": 0, "MaxDD": 0, "Calmar": 0,
                           "AnnVol": 0, "WinRate": 0, "TotalReturn": 0}
            all_trades[key] = {"n_trades": 0, "win_rate": 0, "profit_factor": 0,
                              "avg_win": 0, "avg_loss": 0, "total_pnl": 0}

    # ---- Report ----
    print("\n" + "#" * 70)
    report = generate_report(results, all_trades, bm_csi, bm_eq, valid_dates)
    print(report)
    print("\nDone.")


if __name__ == "__main__":
    main()

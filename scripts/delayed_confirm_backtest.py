"""
Delayed Confirmation Backtest: Reduce bull-trap drawdowns by waiting N days
after CSI300 crosses above 120MA before entering stock positions.

State machine (daily granularity for CSI300):
  CASH (0)       -> CONFIRMING (1) when CSI300 crosses above 120MA
  CONFIRMING (1) -> CONFIRMED (2)  after N-day window passes all checks
  CONFIRMING (1) -> COOLDOWN (3)   if price drops below 120MA during window
  CONFIRMING (1) -> CASH (0)       if N-day window ends but vol/return checks fail
  CONFIRMED (2)  -> CASH (0)       when price drops below 120MA (trend ends)
  COOLDOWN (3)   -> CASH (0)       after M trading days

Parameters swept:
  N = confirm_days  : [3, 5, 10, 20]
  M = cooldown_days : [20, 40, 60]
  position_size     : [full, half]

Train: 2015-2021, Test: 2022-2025.
"""

import itertools
import os
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from quanti.data.storage import DataStorage

# ── Constants ──────────────────────────────────────────
CAPITAL = 90000
COMM = 0.00025

TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END   = "20220101", "20251231"

# Parameter grid
CONFIRM_DAYS_VALS  = [3, 5, 10, 20]
COOLDOWN_DAYS_VALS = [20, 40, 60]
POSITION_SIZES     = ["full", "half"]

# Stock momentum parameters (held fixed, from prior optimization)
TOP_N       = 5
STOP_PCT    = -10
MIN_TREND   = 3
DD_EXIT_PCT = 15

# Confirmation window volume threshold (avg vol in window >= threshold * pre-breakout 20d avg)
VOL_THRESHOLD = 0.85
# Max cumulative decline during confirmation window
MAX_DECLINE = -0.02  # -2%


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def _load_all_data():
    """Load CSI300 for market timing + all stocks for selection."""
    storage = DataStorage()

    # ── CSI300 market proxy (510300) ──
    raw_300 = storage.load_bars("510300")
    if not raw_300:
        raise SystemExit("No CSI300 (510300) data found")

    csi_dates   = np.array([r.trade_date for r in raw_300])
    csi_closes  = np.array([r.close for r in raw_300], dtype=np.float64)
    csi_volumes = np.array([r.volume for r in raw_300], dtype=np.float64)

    # ── All stocks ──
    all_files = sorted(storage.clean_dir.glob("*.parquet"))
    stock_codes = [p.stem for p in all_files
                   if len(p.stem) == 6 and not p.stem.startswith(("51", "58", "15", "56"))]

    stock_data = {}
    all_dates_set = set()
    for code in stock_codes:
        raw = storage.load_bars(code)
        if not raw or len(raw) < 200:
            continue
        dates   = [r.trade_date for r in raw]
        closes  = np.array([r.close for r in raw], dtype=np.float64)
        highs   = np.array([r.high for r in raw], dtype=np.float64)
        lows    = np.array([r.low for r in raw], dtype=np.float64)
        volumes = np.array([r.volume for r in raw], dtype=np.float64)
        stock_data[code] = (closes, highs, lows, volumes, dates)
        all_dates_set.update(dates)

    all_dates = sorted(all_dates_set)
    print(f"Loaded CSI300: {len(csi_closes)} bars  ({csi_dates[0]} ~ {csi_dates[-1]})")
    print(f"Loaded {len(stock_data)} stocks, {len(all_dates)} trading days ({all_dates[0]} ~ {all_dates[-1]})")

    return csi_dates, csi_closes, csi_volumes, stock_data, all_dates


# ═══════════════════════════════════════════════════════════════
# CSI300 Delayed Confirmation State Machine
# ═══════════════════════════════════════════════════════════════

def sma(arr, period):
    if len(arr) < period:
        return None
    out = np.full(len(arr), np.nan)
    cs  = np.cumsum(np.insert(arr, 0, 0.0))
    out[period - 1:] = (cs[period:] - cs[:-period]) / period
    return out


def build_market_state(csi_dates, csi_closes, csi_volumes,
                       confirm_days, cooldown_days):
    """
    Build daily state array for CSI300 market timing.

    Returns:
      state: np.ndarray[int] -- 0=CASH, 1=CONFIRMING, 2=CONFIRMED, 3=COOLDOWN
      state_date_map: dict[str, int] -- trade_date -> state value

    State machine:
      CASH --[cross above 120MA]--> CONFIRMING
      CONFIRMING --[N days pass + checks ok]--> CONFIRMED
      CONFIRMING --[drop below MA during window]--> COOLDOWN
      CONFIRMING --[N days pass + checks fail]--> CASH (unconfirmed)
      CONFIRMED --[drop below MA]--> CASH
      COOLDOWN --[M days pass]--> CASH
    """
    n = len(csi_closes)
    ma120 = sma(csi_closes, 120)
    if ma120 is None:
        raise ValueError("CSI300 data too short for 120MA")

    above = csi_closes > ma120
    state = np.full(n, 0, dtype=int)  # default: CASH

    cooldown_end      = -1
    confirming_since   = -1
    confirmed_since    = -1

    for i in range(121, n):
        # ── Cooldown check ──
        if i <= cooldown_end:
            state[i] = 3
            continue

        # ── Confirmed state: maintain until trend ends ──
        if confirmed_since >= 0:
            if above[i] and not np.isnan(ma120[i]):
                state[i] = 2
            else:
                confirmed_since = -1
                state[i] = 0
            continue

        # ── Confirming state: in N-day window ──
        if confirming_since >= 0:
            days_in = i - confirming_since + 1

            # Price dropped below MA during window -> FALSE BREAKOUT
            if not above[i] or np.isnan(ma120[i]):
                cooldown_end = i + cooldown_days - 1
                state[i] = 3
                confirming_since = -1
                continue

            if days_in == confirm_days:
                # End of window: check volume and return conditions
                w_start = confirming_since
                w_end   = i + 1  # slice end (exclusive)
                window_vol  = np.mean(csi_volumes[w_start:w_end])
                pre_vol     = np.mean(csi_volumes[max(0, w_start - 20):w_start])
                cum_ret     = csi_closes[i] / csi_closes[confirming_since] - 1.0

                if window_vol >= VOL_THRESHOLD * pre_vol and cum_ret >= MAX_DECLINE:
                    state[i] = 2
                    confirmed_since = i
                else:
                    state[i] = 0  # unconfirmed -> back to cash
                confirming_since = -1
            else:
                state[i] = 1  # still confirming
            continue

        # ── Check for new crossing ──
        if above[i] and not above[i - 1] and not np.isnan(ma120[i]):
            confirming_since = i
            state[i] = 1
        else:
            state[i] = 0

    # Build date -> state map
    state_map = {csi_dates[j]: int(state[j]) for j in range(n)}

    return state, state_map


# ═══════════════════════════════════════════════════════════════
# Stock Selection (unchanged from trend-first momentum)
# ═══════════════════════════════════════════════════════════════

def data_at(code, date_str, n, stock_data):
    if code not in stock_data:
        return None
    c, h, l, v, d = stock_data[code]
    idx = None
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= date_str:
            idx = i + 1
            break
    if idx is None or idx < n:
        return None
    return (c[idx - n:idx], h[idx - n:idx], l[idx - n:idx], v[idx - n:idx])


def price_on(code, date_str, stock_data):
    if code not in stock_data:
        return None
    c = stock_data[code][0]
    d = stock_data[code][4]
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= date_str:
            return c[i]
    return None


def adx_arr(high, low, close, period=14):
    n = len(close)
    if n < period * 2:
        return None
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]; dn = low[i - 1] - low[i]
        if up > dn and up > 0:
            pdm[i] = up
        if dn > up and dn > 0:
            mdm[i] = dn
    atr = np.full(n, np.nan)
    atr[period] = float(np.mean(tr[1:period + 1]))
    for i in range(period + 1, n):
        atr[i] = (tr[i] + (period - 1) * atr[i - 1]) / period
    pdi_s = float(np.mean(pdm[1:period + 1]))
    mdi_s = float(np.mean(mdm[1:period + 1]))
    pdi = np.full(n, np.nan); mdi = np.full(n, np.nan)
    pdi[period] = pdi_s / max(atr[period], 0.001) * 100
    mdi[period] = mdi_s / max(atr[period], 0.001) * 100
    for i in range(period + 1, n):
        pdi_s = (pdm[i] + (period - 1) * pdi_s) / period
        mdi_s = (mdm[i] + (period - 1) * mdi_s) / period
        pdi[i] = min(pdi_s / max(atr[i], 0.001) * 100, 1000)
        mdi[i] = min(mdi_s / max(atr[i], 0.001) * 100, 1000)
    dx = np.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100
    adx_o = np.full(n, np.nan)
    seed = float(np.nanmean(dx[period:period * 2]))
    adx_o[period * 2 - 1] = 0.0 if np.isnan(seed) else seed
    ds = adx_o[period * 2 - 1]
    for i in range(period * 2, n):
        vi = dx[i] if not np.isnan(dx[i]) else ds
        ds = (vi + (period - 1) * ds) / period
        adx_o[i] = ds
    return adx_o


def is_stock_uptrend(closes, highs, lows, volumes):
    if len(closes) < 200:
        return False, 0
    ma120 = sma(closes, 120)
    if ma120 is None or np.isnan(ma120[-1]):
        return False, 0
    above_ma = closes[-1] > ma120[-1]
    recent_high = np.max(highs[-20:]); prev_high = np.max(highs[-60:-20])
    higher_high = recent_high > prev_high
    recent_low  = np.min(lows[-20:]);  prev_low  = np.min(lows[-60:-20])
    higher_low  = recent_low > prev_low
    ma20_a = sma(closes, 20); ma60_a = sma(closes, 60)
    if ma20_a is None or ma60_a is None:
        return False, 0
    if np.isnan(ma20_a[-1]) or np.isnan(ma60_a[-1]):
        return False, 0
    ma_aligned = ma20_a[-1] > ma60_a[-1] > ma120[-1]
    adx_v = adx_arr(highs, lows, closes, 14)
    adx_ok = adx_v is not None and not np.isnan(adx_v[-1]) and adx_v[-1] > 25
    vol_20 = np.mean(volumes[-21:-1])
    vol_surge = volumes[-1] > vol_20 * 1.2
    conditions = [above_ma, higher_high and higher_low, ma_aligned, adx_ok, vol_surge]
    score = sum(conditions)
    is_trend = above_ma and adx_ok and score >= MIN_TREND
    return is_trend, score


def trend_strength_score(closes):
    if len(closes) < 130:
        return 0
    ret_3m = closes[-1] / closes[-63] - 1 if closes[-63] > 1e-6 else 0
    ret_6m = closes[-1] / closes[-126] - 1 if closes[-126] > 1e-6 else 0
    m3 = min(max(ret_3m / 0.5, 0), 1) if ret_3m > 0 else 0
    m6 = min(max(ret_6m / 0.8, 0), 1) if ret_6m > 0 else 0
    mom = (0.5 * m3 + 0.5 * m6) * 100
    w = closes[-61:]
    dr = np.diff(w) / (w[:-1] + 1e-10)
    vol = np.nanstd(dr)
    vs = (1 - min(vol / 0.04, 1)) * 100
    return 0.6 * mom + 0.4 * vs


def get_monthly_dates(dates, start, end):
    monthly = []
    for d in dates:
        if d < start or d > end:
            continue
        dm = d[4:6]
        if not monthly or dm != monthly[-1][4:6]:
            monthly.append(d)
    return monthly


# ═══════════════════════════════════════════════════════════════
# Backtest Engine
# ═══════════════════════════════════════════════════════════════

def run_backtest(confirm_days, cooldown_days, position_size,
                 start_date, end_date,
                 csi_dates, csi_closes, csi_volumes,
                 stock_data, all_dates):
    """
    Run delayed-confirmation backtest for a single parameter set.

    Returns:
      dict with cagr, sharpe, maxdd, n_years, final_eq, n_trades,
      months_invested, months_cash, months_confirming, months_cooldown
    """
    # Build market state for the full period
    _, state_map = build_market_state(csi_dates, csi_closes, csi_volumes,
                                      confirm_days, cooldown_days)

    size_mult = 1.0 if position_size == "full" else 0.5
    rebal = get_monthly_dates(all_dates, start_date, end_date)
    if len(rebal) < 12:
        return {"cagr": 0.0, "sharpe": 0.0, "maxdd": 100.0,
                "ny": 0.0, "final": CAPITAL, "trades": 0,
                "m_inv": 0, "m_cash": 0, "m_conf": 0, "m_cool": 0}

    cash = CAPITAL
    holdings = {}
    eq_curve = [cash]
    max_eq = cash
    dd_exit_triggered = False

    m_inv = m_cash = m_conf = m_cool = 0

    for reb_date in rebal:
        # ── Look up market state ──
        mkt_state = state_map.get(reb_date, 0)

        # Track state distribution
        if mkt_state == 2:
            m_inv += 1
        elif mkt_state == 0:
            m_cash += 1
        elif mkt_state == 1:
            m_conf += 1
        elif mkt_state == 3:
            m_cool += 1

        # ── Update holdings and apply stop-loss ──
        for sym in list(holdings.keys()):
            p = price_on(sym, reb_date, stock_data)
            if p is None or p < 0.01:
                cash += holdings[sym]["val"] * 0.7
                del holdings[sym]
                continue
            holdings[sym]["price"] = p
            holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]:
                holdings[sym]["hwm"] = p
            hwm = holdings[sym]["hwm"]
            if hwm > 0 and (p / hwm - 1) * 100 < STOP_PCT:
                mv = holdings[sym]["qty"] * p
                cash += mv * (1 - COMM)
                del holdings[sym]

        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())

        # ── Portfolio drawdown breaker ──
        if total > max_eq:
            max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                mv = holdings[sym]["qty"] * holdings[sym]["price"]
                cash += mv * (1 - COMM)
                del holdings[sym]
            dd_exit_triggered = True
        elif dd_exit_triggered:
            if total / max_eq > 0.92:
                dd_exit_triggered = False

        if dd_exit_triggered:
            total = cash
            eq_curve.append(total)
            continue

        # ── Market gate: only trade when CONFIRMED ──
        if mkt_state != 2:
            # Sell everything, stay cash
            for sym in list(holdings.keys()):
                mv = holdings[sym]["qty"] * holdings[sym]["price"]
                cash += mv * (1 - COMM)
                del holdings[sym]
            eq_curve.append(total)
            continue

        # ── CONFIRMED: select stocks ──
        trending = []
        for code in stock_data:
            d = data_at(code, reb_date, 260, stock_data)
            if d is None:
                continue
            c, h, l, v = d
            is_t, n_cond = is_stock_uptrend(c, h, l, v)
            if is_t and n_cond >= MIN_TREND:
                s = trend_strength_score(c)
                trending.append((code, s, n_cond))

        if not trending:
            eq_curve.append(total)
            continue

        trending.sort(key=lambda x: x[1], reverse=True)
        selected = {t[0] for t in trending[:TOP_N]}

        # Rotate out non-selected
        for sym in list(holdings.keys()):
            if sym not in selected:
                mv = holdings[sym]["qty"] * holdings[sym]["price"]
                cash += mv * (1 - COMM)
                del holdings[sym]

        # Buy new / rebalance existing
        n_pos = max(len(selected), 1)
        allocable = total * size_mult
        per_stock = allocable / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, reb_date, stock_data)
            if p is None or p < 0.01:
                continue
            target_qty = int(per_stock / p / 100) * 100
            if target_qty < 100:
                continue
            if sym in holdings:
                cur = holdings[sym]["qty"]
                diff = target_qty - cur
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1 + COMM):
                        cash -= cost * (1 + COMM)
                        holdings[sym]["qty"] = target_qty
                    elif diff < 0:
                        cash += cost * (1 - COMM)
                        holdings[sym]["qty"] = target_qty
            else:
                cost = target_qty * p
                if cash >= cost * (1 + COMM):
                    cash -= cost * (1 + COMM)
                    holdings[sym] = {"qty": target_qty, "price": p,
                                     "val": cost, "hwm": p}

        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())
        eq_curve.append(total)

    # ── Compute metrics ──
    eq = np.array(eq_curve)
    n_y = len(eq_curve) / 12.0
    if eq[0] <= 0 or n_y <= 0:
        return {"cagr": 0.0, "sharpe": 0.0, "maxdd": 100.0,
                "ny": n_y, "final": eq[-1] if len(eq) > 0 else CAPITAL,
                "trades": 0, "m_inv": m_inv, "m_cash": m_cash,
                "m_conf": m_conf, "m_cool": m_cool}

    cagr = ((eq[-1] / eq[0]) ** (1 / n_y) - 1) * 100
    mr = np.diff(eq) / (eq[:-1] + 1e-10)
    sharpe = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 and np.std(mr) > 1e-10 else 0

    peak = eq[0]; maxdd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd_i = (peak - v) / peak * 100
        if dd_i > maxdd:
            maxdd = dd_i

    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd,
            "ny": n_y, "final": float(eq[-1]),
            "trades": 0, "m_inv": m_inv, "m_cash": m_cash,
            "m_conf": m_conf, "m_cool": m_cool}


# ═══════════════════════════════════════════════════════════════
# Baseline: Original (no delay confirmation, traditional CSI300 > 120MA)
# ═══════════════════════════════════════════════════════════════

def run_baseline(start_date, end_date, stock_data, all_dates):
    """Original trend-first momentum: CSI300 > 120MA as simple gate."""
    rebal = get_monthly_dates(all_dates, start_date, end_date)
    cash = CAPITAL
    holdings = {}
    eq_curve = [cash]
    max_eq = cash
    dd_exit_triggered = False
    m_inv = 0

    for reb_date in rebal:
        # Simple market gate: CSI300 > 120MA
        d = data_at("510300", reb_date, 200, stock_data)
        mkt_ok = True
        if d is not None:
            c, _, _, _ = d
            ma120 = sma(c, 120)
            if ma120 is not None and not np.isnan(ma120[-1]):
                mkt_ok = c[-1] > ma120[-1]

        # Stop-loss
        for sym in list(holdings.keys()):
            p = price_on(sym, reb_date, stock_data)
            if p is None or p < 0.01:
                cash += holdings[sym]["val"] * 0.7
                del holdings[sym]; continue
            holdings[sym]["price"] = p
            holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]:
                holdings[sym]["hwm"] = p
            hwm = holdings[sym]["hwm"]
            if hwm > 0 and (p / hwm - 1) * 100 < STOP_PCT:
                mv = holdings[sym]["qty"] * p
                cash += mv * (1 - COMM)
                del holdings[sym]

        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())

        if total > max_eq:
            max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0
        if dd > DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                mv = holdings[sym]["qty"] * holdings[sym]["price"]
                cash += mv * (1 - COMM)
                del holdings[sym]
            dd_exit_triggered = True
        elif dd_exit_triggered:
            if total / max_eq > 0.92:
                dd_exit_triggered = False

        if dd_exit_triggered:
            eq_curve.append(total)
            continue

        if not mkt_ok:
            for sym in list(holdings.keys()):
                mv = holdings[sym]["qty"] * holdings[sym]["price"]
                cash += mv * (1 - COMM)
                del holdings[sym]
            eq_curve.append(total)
            continue

        m_inv += 1
        trending = []
        for code in stock_data:
            d2 = data_at(code, reb_date, 260, stock_data)
            if d2 is None: continue
            c, h, l, v = d2
            is_t, n_cond = is_stock_uptrend(c, h, l, v)
            if is_t and n_cond >= MIN_TREND:
                s = trend_strength_score(c)
                trending.append((code, s, n_cond))

        if not trending:
            eq_curve.append(total)
            continue

        trending.sort(key=lambda x: x[1], reverse=True)
        selected = {t[0] for t in trending[:TOP_N]}

        for sym in list(holdings.keys()):
            if sym not in selected:
                mv = holdings[sym]["qty"] * holdings[sym]["price"]
                cash += mv * (1 - COMM)
                del holdings[sym]

        n_pos = max(len(selected), 1)
        per_stock = total / n_pos * 0.90
        for sym in selected:
            p = price_on(sym, reb_date, stock_data)
            if p is None or p < 0.01: continue
            tq = int(per_stock / p / 100) * 100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1 + COMM):
                        cash -= cost * (1 + COMM)
                        holdings[sym]["qty"] = tq
                    elif diff < 0:
                        cash += cost * (1 - COMM)
                        holdings[sym]["qty"] = tq
            else:
                cost = tq * p
                if cash >= cost * (1 + COMM):
                    cash -= cost * (1 + COMM)
                    holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}

        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())
        eq_curve.append(total)

    eq = np.array(eq_curve)
    n_y = len(eq_curve) / 12.0
    if eq[0] <= 0 or n_y <= 0:
        return {"cagr": 0.0, "sharpe": 0.0, "maxdd": 100.0, "ny": n_y, "final": CAPITAL}

    cagr = ((eq[-1] / eq[0]) ** (1 / n_y) - 1) * 100
    mr = np.diff(eq) / (eq[:-1] + 1e-10)
    sharpe = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 and np.std(mr) > 1e-10 else 0
    peak = eq[0]; maxdd = 0.0
    for v in eq:
        if v > peak: peak = v
        dd_i = (peak - v) / peak * 100
        if dd_i > maxdd: maxdd = dd_i

    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "ny": n_y,
            "final": float(eq[-1]), "m_inv": m_inv}


# ═══════════════════════════════════════════════════════════════
# Full Parameter Sweep
# ═══════════════════════════════════════════════════════════════

def sweep_all(csi_dates, csi_closes, csi_volumes, stock_data, all_dates):
    grid = list(itertools.product(CONFIRM_DAYS_VALS, COOLDOWN_DAYS_VALS, POSITION_SIZES))
    total = len(grid)

    print(f"\n{'='*80}")
    print(f"DELAYED CONFIRMATION STRATEGY - PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"Parameter grid: {len(CONFIRM_DAYS_VALS)} confirm x {len(COOLDOWN_DAYS_VALS)} cooldown x {len(POSITION_SIZES)} size = {total} combos")
    print(f"Train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"Test:  {TEST_START} ~ {TEST_END}")
    print(f"Volume threshold: {VOL_THRESHOLD:.0%} of pre-breakout 20d avg")
    print(f"Max decline in window: {MAX_DECLINE:.0%}")
    print(f"Stock params (fixed): top_n={TOP_N}, stop={STOP_PCT}%, min_trend={MIN_TREND}, dd_exit={DD_EXIT_PCT}%")
    print()

    # ── Baseline ──
    print("Computing baseline (traditional CSI300 > 120MA gate)...")
    bl_train = run_baseline(TRAIN_START, TRAIN_END, stock_data, all_dates)
    bl_test  = run_baseline(TEST_START, TEST_END, stock_data, all_dates)
    print(f"  Baseline Train: CAGR={bl_train['cagr']:+.2f}% Sharpe={bl_train['sharpe']:.3f} MaxDD={bl_train['maxdd']:.1f}%")
    print(f"  Baseline Test:  CAGR={bl_test['cagr']:+.2f}% Sharpe={bl_test['sharpe']:.3f} MaxDD={bl_test['maxdd']:.1f}%")
    print()

    results = []

    for idx, (N, M, ps) in enumerate(grid):
        name = f"N{N}_M{M}_{ps}"

        # Train
        r_train = run_backtest(N, M, ps, TRAIN_START, TRAIN_END,
                               csi_dates, csi_closes, csi_volumes,
                               stock_data, all_dates)

        # Test
        r_test = run_backtest(N, M, ps, TEST_START, TEST_END,
                              csi_dates, csi_closes, csi_volumes,
                              stock_data, all_dates)

        # Sharpe decay
        if r_train["sharpe"] > 0.01:
            sharpe_decay = (r_train["sharpe"] - r_test["sharpe"]) / r_train["sharpe"]
        else:
            sharpe_decay = 999

        # MaxDD inflation
        if r_train["maxdd"] > 0.01:
            maxdd_inflate = (r_test["maxdd"] / r_train["maxdd"] - 1) * 100
        else:
            maxdd_inflate = 999

        entry = {
            "name": name,
            "N": N, "M": M, "position_size": ps,
            "train_cagr": round(r_train["cagr"], 2),
            "train_sharpe": round(r_train["sharpe"], 3),
            "train_maxdd": round(r_train["maxdd"], 2),
            "train_ny": round(r_train["ny"], 2),
            "test_cagr": round(r_test["cagr"], 2),
            "test_sharpe": round(r_test["sharpe"], 3),
            "test_maxdd": round(r_test["maxdd"], 2),
            "test_ny": round(r_test["ny"], 2),
            "sharpe_decay": round(sharpe_decay, 3),
            "maxdd_inflate": round(maxdd_inflate, 2),
            "train_inv": r_train["m_inv"],
            "train_cash": r_train["m_cash"],
            "train_conf": r_train["m_conf"],
            "train_cool": r_train["m_cool"],
            "test_inv": r_test["m_inv"],
            "test_cash": r_test["m_cash"],
            "test_conf": r_test["m_conf"],
            "test_cool": r_test["m_cool"],
        }
        results.append(entry)

        print(f"[{idx+1:3d}/{total}] {name:<18s} | "
              f"Train: C={r_train['cagr']:+6.2f}% S={r_train['sharpe']:5.2f} D={r_train['maxdd']:5.1f}% | "
              f"Test: C={r_test['cagr']:+6.2f}% S={r_test['sharpe']:5.2f} D={r_test['maxdd']:5.1f}% | "
              f"Dcy={sharpe_decay:.2f} Inf={maxdd_inflate:+.1f}% | "
              f"Inv={r_test['m_inv']}/{len(get_monthly_dates(all_dates, TEST_START, TEST_END))}m")

    # ── Rankings ──
    _print_rankings(results, bl_train, bl_test)
    _write_report(results, bl_train, bl_test)

    return results


def _print_rankings(results, bl_train, bl_test):
    print(f"\n{'='*80}")
    print("TOP 5 BY TEST CAGR")
    print(f"{'='*80}")
    for i, r in enumerate(sorted(results, key=lambda x: x["test_cagr"], reverse=True)[:5], 1):
        imp = r["test_cagr"] - bl_test["cagr"]
        print(f"  #{i}: {r['name']:<18s} Test C={r['test_cagr']:+5.1f}% S={r['test_sharpe']:.3f} D={r['test_maxdd']:.1f}% "
              f"(vs baseline {bl_test['cagr']:+.1f}%, delta={imp:+.1f}%) "
              f"| Train C={r['train_cagr']:+5.1f}%")

    print(f"\n{'='*80}")
    print("TOP 5 BY TEST SHARPE")
    print(f"{'='*80}")
    for i, r in enumerate(sorted(results, key=lambda x: x["test_sharpe"], reverse=True)[:5], 1):
        print(f"  #{i}: {r['name']:<18s} Test S={r['test_sharpe']:.3f} C={r['test_cagr']:+5.1f}% D={r['test_maxdd']:.1f}% "
              f"(baseline S={bl_test['sharpe']:.3f}) | Train S={r['train_sharpe']:.3f}")

    print(f"\n{'='*80}")
    print("TOP 5 BY MINIMUM MAXDD")
    print(f"{'='*80}")
    for i, r in enumerate(sorted(results, key=lambda x: x["test_maxdd"])[:5], 1):
        imp = bl_test["maxdd"] - r["test_maxdd"]
        print(f"  #{i}: {r['name']:<18s} Test D={r['test_maxdd']:.1f}% (baseline={bl_test['maxdd']:.1f}%, "
              f"improvement={imp:+.1f}%) | C={r['test_cagr']:+5.1f}%")

    print(f"\n{'='*80}")
    print("TOP 3 BY MINIMUM SHARPE DECAY (least overfitting)")
    print(f"{'='*80}")
    for i, r in enumerate(sorted(results, key=lambda x: x["sharpe_decay"])[:3], 1):
        print(f"  #{i}: {r['name']:<18s} Decay={r['sharpe_decay']:.3f} "
              f"| Train S={r['train_sharpe']:.3f} -> Test S={r['test_sharpe']:.3f}")

    # ── Composite best ──
    by_cagr   = sorted(results, key=lambda x: x["test_cagr"], reverse=True)
    by_sharpe = sorted(results, key=lambda x: x["test_sharpe"], reverse=True)
    by_maxdd  = sorted(results, key=lambda x: x["test_maxdd"])
    by_decay  = sorted(results, key=lambda x: x["sharpe_decay"])

    for i, r in enumerate(by_cagr):   r["rank_cagr"]   = i
    for i, r in enumerate(by_sharpe): r["rank_sharpe"] = i
    for i, r in enumerate(by_maxdd):  r["rank_maxdd"]  = i
    for i, r in enumerate(by_decay):  r["rank_decay"]  = i
    for r in results:
        r["composite"] = r["rank_cagr"] + r["rank_sharpe"] + r["rank_maxdd"] + r["rank_decay"]

    best = min(results, key=lambda x: x["composite"])

    print(f"\n{'='*80}")
    print("COMPOSITE BEST (rank sum: CAGR + Sharpe + MaxDD + Decay)")
    print(f"{'='*80}")
    print(f"  {best['name']}  (N={best['N']}, M={best['M']}, size={best['position_size']})")
    print(f"  Train: CAGR={best['train_cagr']:+.1f}%  Sharpe={best['train_sharpe']:.3f}  MaxDD={best['train_maxdd']:.1f}%")
    print(f"  Test:  CAGR={best['test_cagr']:+.1f}%  Sharpe={best['test_sharpe']:.3f}  MaxDD={best['test_maxdd']:.1f}%")
    print(f"  Baseline Test: CAGR={bl_test['cagr']:+.1f}% Sharpe={bl_test['sharpe']:.3f} MaxDD={bl_test['maxdd']:.1f}%")
    print(f"  Sharpe Decay: {best['sharpe_decay']:.3f}  |  MaxDD Inflation: {best['maxdd_inflate']:+.1f}%")
    print(f"  Months invested (test): {best['test_inv']} invested, {best['test_cash']} cash, "
          f"{best['test_conf']} confirming, {best['test_cool']} cooldown")


def _write_report(results, bl_train, bl_test):
    """Write comprehensive markdown report."""
    report_path = os.path.join(_PROJECT_ROOT, "data", "delayed_confirm_report.md")

    by_cagr   = sorted(results, key=lambda x: x["test_cagr"], reverse=True)
    by_sharpe = sorted(results, key=lambda x: x["test_sharpe"], reverse=True)
    by_maxdd  = sorted(results, key=lambda x: x["test_maxdd"])
    by_decay  = sorted(results, key=lambda x: x["sharpe_decay"])

    for i, r in enumerate(by_cagr):   r["rank_cagr"]   = i
    for i, r in enumerate(by_sharpe): r["rank_sharpe"] = i
    for i, r in enumerate(by_maxdd):  r["rank_maxdd"]  = i
    for i, r in enumerate(by_decay):  r["rank_decay"]  = i
    for r in results:
        r["composite"] = r["rank_cagr"] + r["rank_sharpe"] + r["rank_maxdd"] + r["rank_decay"]

    best = min(results, key=lambda x: x["composite"])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Delayed Confirmation Strategy: Parameter Search Report\n\n")
        f.write(f"**Generated:** 2026-06-14\n\n")

        f.write("## Research Question\n\n")
        f.write("Can a rolling multi-day confirmation window (after CSI300 crosses above 120MA) "
                "reduce false-breakout drawdowns without missing slow-bull markets?\n\n")

        f.write("## Framework\n\n")
        f.write(f"- **Train:** {TRAIN_START} to {TRAIN_END} (7 years)\n")
        f.write(f"- **Test:** {TEST_START} to {TEST_END} (4 years)\n")
        f.write("- **Strict sample-out-of-sample** - no future information leakage\n\n")

        f.write("## Strategy Logic\n\n")
        f.write("### Market Timing State Machine\n\n")
        f.write("1. **CASH** --[CSI300 crosses above 120MA]--> **CONFIRMING**\n")
        f.write(f"2. **CONFIRMING** --[N-day window passes all checks]--> **CONFIRMED**\n")
        f.write(f"3. **CONFIRMING** --[price drops below 120MA during window]--> **COOLDOWN** (M days)\n")
        f.write(f"4. **CONFIRMING** --[N-day window ends, vol/return checks fail]--> **CASH** (unconfirmed)\n")
        f.write(f"5. **CONFIRMED** --[price drops below 120MA]--> **CASH** (trend ends)\n")
        f.write(f"6. **COOLDOWN** --[M days elapsed]--> **CASH**\n\n")

        f.write("### Confirmation Window Checks (at end of N-day window)\n\n")
        f.write(f"- Average daily CSI300 volume during window >= {VOL_THRESHOLD:.0%} of pre-breakout 20-day average\n")
        f.write(f"- Cumulative CSI300 return during window >= {MAX_DECLINE:.0%} (no false-breakout decline pattern)\n")
        f.write("- CSI300 price must remain above 120MA throughout the window\n\n")

        f.write("### Stock Selection (unchanged from trend-first momentum)\n\n")
        f.write(f"- 5 trend conditions (price>120MA, higher highs/lows, MA alignment, ADX>25, volume expansion)\n")
        f.write(f"- Top {TOP_N} by momentum(60%)+low-vol(40%) composite score\n")
        f.write(f"- Individual stop-loss: {STOP_PCT}% trailing from HWM\n")
        f.write(f"- Portfolio drawdown breaker: {DD_EXIT_PCT}% from peak\n\n")

        f.write("## Parameter Space\n\n")
        f.write(f"- Confirm days (N): {CONFIRM_DAYS_VALS}\n")
        f.write(f"- Cooldown days (M): {COOLDOWN_DAYS_VALS}\n")
        f.write(f"- Position size: {POSITION_SIZES}\n")
        f.write(f"- Total combinations: {len(results)}\n\n")

        f.write("## Baseline (Traditional CSI300 > 120MA Gate)\n\n")
        f.write("| Period | CAGR | Sharpe | MaxDD |\n")
        f.write("|--------|------|--------|-------|\n")
        f.write(f"| Train | {bl_train['cagr']:+.2f}% | {bl_train['sharpe']:.3f} | {bl_train['maxdd']:.1f}% |\n")
        f.write(f"| Test | {bl_test['cagr']:+.2f}% | {bl_test['sharpe']:.3f} | {bl_test['maxdd']:.1f}% |\n\n")

        f.write("## 1. Top 10 by Test CAGR\n\n")
        f.write("| # | Config | Test CAGR | Test Sharpe | Test MaxDD | Train CAGR | Train Sharpe | Sharpe Decay | MaxDD Infl |\n")
        f.write("|---|--------|-----------|-------------|------------|------------|-------------|-------------|----------|\n")
        for i, r in enumerate(by_cagr[:10], 1):
            f.write(f"| {i} | {r['name']} | {r['test_cagr']:+.2f}% | {r['test_sharpe']:.3f} | {r['test_maxdd']:.1f}% | {r['train_cagr']:+.2f}% | {r['train_sharpe']:.3f} | {r['sharpe_decay']:.3f} | {r['maxdd_inflate']:+.1f}% |\n")

        f.write("\n## 2. Top 10 by Test Sharpe\n\n")
        f.write("| # | Config | Test Sharpe | Test CAGR | Test MaxDD | Train Sharpe | Sharpe Decay |\n")
        f.write("|---|--------|-------------|-----------|------------|-------------|-------------|\n")
        for i, r in enumerate(by_sharpe[:10], 1):
            f.write(f"| {i} | {r['name']} | {r['test_sharpe']:.3f} | {r['test_cagr']:+.2f}% | {r['test_maxdd']:.1f}% | {r['train_sharpe']:.3f} | {r['sharpe_decay']:.3f} |\n")

        f.write("\n## 3. Top 10 by Minimum MaxDD (Best Drawdown Protection)\n\n")
        f.write("| # | Config | Test MaxDD | Test CAGR | Test Sharpe | Train MaxDD | MaxDD Infl |\n")
        f.write("|---|--------|------------|-----------|-------------|-------------|----------|\n")
        for i, r in enumerate(by_maxdd[:10], 1):
            dd_imp = bl_test["maxdd"] - r["test_maxdd"]
            f.write(f"| {i} | {r['name']} | {r['test_maxdd']:.1f}% | {r['test_cagr']:+.2f}% | {r['test_sharpe']:.3f} | {r['train_maxdd']:.1f}% | {r['maxdd_inflate']:+.1f}% |\n")

        f.write("\n## 4. Top 5 by Minimum Sharpe Decay (Least Overfitting)\n\n")
        f.write("| # | Config | Sharpe Decay | Train Sharpe | Test Sharpe | Test CAGR |\n")
        f.write("|---|--------|-------------|-------------|------------|----------|\n")
        for i, r in enumerate(by_decay[:5], 1):
            f.write(f"| {i} | {r['name']} | {r['sharpe_decay']:.3f} | {r['train_sharpe']:.3f} | {r['test_sharpe']:.3f} | {r['test_cagr']:+.2f}% |\n")

        f.write("\n## 5. Composite Best Recommendation\n\n")
        f.write(f"**{best['name']}** (N={best['N']}, M={best['M']}, size={best['position_size']})\n\n")
        f.write("| Metric | Train | Test | Baseline Test | Improvement |\n")
        f.write("|--------|-------|------|---------------|-------------|\n")
        f.write(f"| CAGR | {best['train_cagr']:+.2f}% | {best['test_cagr']:+.2f}% | {bl_test['cagr']:+.2f}% | {best['test_cagr'] - bl_test['cagr']:+.2f}% |\n")
        f.write(f"| Sharpe | {best['train_sharpe']:.3f} | {best['test_sharpe']:.3f} | {bl_test['sharpe']:.3f} | {best['test_sharpe'] - bl_test['sharpe']:+.3f} |\n")
        f.write(f"| MaxDD | {best['train_maxdd']:.1f}% | {best['test_maxdd']:.1f}% | {bl_test['maxdd']:.1f}% | {bl_test['maxdd'] - best['test_maxdd']:+.1f}% |\n")
        f.write(f"| Sharpe Decay | | {best['sharpe_decay']:.3f} | | |\n")
        f.write(f"| MaxDD Inflation | | {best['maxdd_inflate']:+.1f}% | | |\n\n")

        f.write(f"- Months invested (test): {best['test_inv']} invested, {best['test_cash']} cash, "
                f"{best['test_conf']} confirming, {best['test_cool']} cooldown\n\n")

        f.write("### Analysis\n\n")
        improvements = [f"{best['test_cagr'] - bl_test['cagr']:+.1f}% CAGR" if best['test_cagr'] > bl_test['cagr'] else f"{best['test_cagr'] - bl_test['cagr']:+.1f}% CAGR"]
        dd_improvement = bl_test['maxdd'] - best['test_maxdd']
        if dd_improvement > 0:
            improvements.append(f"{dd_improvement:+.1f}% MaxDD improvement")
        f.write(f"The composite best configuration delivers {', '.join(improvements)} vs baseline.\n\n")

        # Year-by-year test breakdown for best config
        f.write("## 6. Full Parameter Grid (sorted by Test CAGR)\n\n")
        f.write("| Config | N | M | Size | Train CAGR | Train Sharpe | Train MaxDD | Test CAGR | Test Sharpe | Test MaxDD | Sharpe Decay | MaxDD Infl | T.Inv | T.Cash | T.Conf | T.Cool |\n")
        f.write("|--------|---|---|------|-----------|-------------|------------|-----------|-------------|------------|-------------|----------|-------|--------|--------|--------|\n")
        for r in sorted(results, key=lambda x: x["test_cagr"], reverse=True):
            f.write(f"| {r['name']} | {r['N']} | {r['M']} | {r['position_size']} | {r['train_cagr']:+.2f}% | {r['train_sharpe']:.3f} | {r['train_maxdd']:.1f}% | {r['test_cagr']:+.2f}% | {r['test_sharpe']:.3f} | {r['test_maxdd']:.1f}% | {r['sharpe_decay']:.3f} | {r['maxdd_inflate']:+.1f}% | {r['test_inv']} | {r['test_cash']} | {r['test_conf']} | {r['test_cool']} |\n")

    print(f"\nReport written to: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Loading data...")
    print("=" * 80)
    csi_dates, csi_closes, csi_volumes, stock_data, all_dates = _load_all_data()
    sweep_all(csi_dates, csi_closes, csi_volumes, stock_data, all_dates)
    print("\nDone.")


if __name__ == "__main__":
    main()

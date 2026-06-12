"""
Shared technical indicator functions.

All indicator computation lives here exactly once. Strategy classes
(ETFTrendStrategy, StockMomentumStrategy), RiskChecker, and any future
consumer import from this module. Each indicator is a pure function
operating on numpy arrays or Bar lists -- zero side effects, zero
dependencies on settings or global state.

Exported functions:
  sma, ema, wilder_smooth
  adx, adx_with_di
  bollinger_bands
  compute_atr, compute_rsi
"""

import numpy as np

# ---- Moving Averages ----

def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average. Returns array with NaN for insufficient data."""
    result = np.full_like(data, np.nan, dtype=np.float64)
    if len(data) >= period:
        cumsum = np.cumsum(np.insert(data, 0, 0))
        result[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return result


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    result = np.full_like(data, np.nan, dtype=np.float64)
    if len(data) >= period:
        result[period - 1] = np.mean(data[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def wilder_smooth(data: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothing (EMA with alpha = 1/period).

    Carries last valid value through NaN inputs to prevent signal gaps.
    """
    result = np.full_like(data, np.nan, dtype=np.float64)
    start = period
    while start < len(data):
        window = data[1:start + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            result[start] = np.mean(valid)
            break
        start += 1
    if start >= len(data):
        return result
    for i in range(start + 1, len(data)):
        prev = result[i - 1]
        cur = data[i]
        if np.isnan(cur):
            result[i] = prev
        elif np.isnan(prev):
            result[i] = cur
        else:
            result[i] = (cur + (period - 1) * prev) / period
    return result


# ---- ADX / Directional Movement ----

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray | None:
    """
    Average Directional Index (ADX).
    Returns array of ADX values or None if insufficient data.
    """
    result = adx_with_di(high, low, close, period)
    if result is None:
        return None
    return result[0]


def adx_with_di(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    ADX with Plus/Minus DI.
    Returns (adx, plus_di, minus_di) arrays or None if insufficient data.
    """
    n = len(close)
    if n < period * 2:
        return None

    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    atr_arr = wilder_smooth(tr, period)
    plus_di = wilder_smooth(plus_dm, period) / atr_arr * 100
    minus_di = wilder_smooth(minus_dm, period) / atr_arr * 100

    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx_arr = wilder_smooth(dx, period)

    return (adx_arr, plus_di, minus_di)


# ---- Bollinger Bands ----

def bollinger_bands(
    closes: np.ndarray, period: int, std_mult: float
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Compute Bollinger Bands.
    Returns (middle, upper, lower) arrays or (None, None, None) if insufficient data.
    middle = SMA(period)
    upper  = middle + rolling_std * std_mult
    lower  = middle - rolling_std * std_mult
    """
    n = len(closes)
    if n < period:
        return (None, None, None)

    middle = sma(closes, period)
    upper = np.full_like(closes, np.nan, dtype=np.float64)
    lower = np.full_like(closes, np.nan, dtype=np.float64)

    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        std = np.std(window, ddof=1)
        upper[i] = middle[i] + std * std_mult
        lower[i] = middle[i] - std * std_mult

    return (middle, upper, lower)


# ---- ATR (from Bar list) ----

def compute_atr(bars: list, period: int = 14) -> float | None:
    """
    Compute ATR (Average True Range) from a list of Bar objects.
    Uses Wilder's smoothing.
    """
    n = len(bars)
    if n < period + 1:
        return None

    tr_values = np.zeros(n - 1, dtype=np.float64)
    for i in range(1, n):
        tr_values[i - 1] = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )

    atr_val = np.mean(tr_values[:period])
    for i in range(period, len(tr_values)):
        atr_val = (tr_values[i] + (period - 1) * atr_val) / period

    return float(atr_val)


# ---- RSI ----

def compute_rsi(closes: np.ndarray, period: int = 14) -> float | None:
    """
    Compute RSI (Relative Strength Index) using Wilder's smoothing.
    Returns the latest RSI value or None.
    """
    n = len(closes)
    if n < period + 1:
        return None

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0.0:
        return 100.0

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return float(rsi)

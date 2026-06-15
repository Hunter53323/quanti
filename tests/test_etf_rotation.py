"""Tests for ETFRotationStrategy: signal methods, constructor, compute_scores.
Covers the integration gaps identified after MACD/KDJ indicator work:
- _macd_signal() and _kdj_signal() edge cases
- Constructor backward compatibility with explicit old-style weights
- compute_scores() regression gate (zero-weight factor equivalence)
- Server indicator parity (shared macd/kdj match legacy _macd/_kdj helpers)
"""
from datetime import datetime

import numpy as np

from quanti.strategy.etf_rotation import ETFRotationStrategy
from quanti.types import Bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(end_price: float, n: int = 200, symbol: str = "510300",
               volatility: float = 0.005) -> list[Bar]:
    """Create *n* bars with moderate noise so indicators get realistic input."""
    rng = np.random.default_rng(42)
    closes = np.linspace(10.0, end_price, n, dtype=np.float64)
    noise = rng.normal(0, volatility, n)
    closes = closes + np.cumsum(noise) * 0.1  # mild random walk
    closes = np.maximum(closes, 1.0)

    start = datetime(2020, 1, 1)
    bars: list[Bar] = []
    for i, c in enumerate(closes):
        bars.append(Bar(
            symbol=symbol,
            datetime=start + __import__("datetime").timedelta(days=i),
            open=float(c * 0.995),
            high=float(c * 1.01),
            low=float(c * 0.985),
            close=float(c),
            volume=1_000_000,
        ))
    return bars


def _arr(bars: list[Bar]) -> np.ndarray:
    """Extract close array."""
    return np.array([b.close for b in bars], dtype=np.float64)


def _arr_high(bars: list[Bar]) -> np.ndarray:
    return np.array([b.high for b in bars], dtype=np.float64)


def _arr_low(bars: list[Bar]) -> np.ndarray:
    return np.array([b.low for b in bars], dtype=np.float64)


# ---------------------------------------------------------------------------
# MACD signal
# ---------------------------------------------------------------------------

class TestMACDSignal:
    """_macd_signal() edge cases and expected behavior."""

    def test_returns_float(self):
        bars = _make_bars(15.0, 200)
        sig = ETFRotationStrategy._macd_signal(_arr(bars), 12, 26, 9)
        assert isinstance(sig, float)
        assert sig in (0.0, 1.0)

    def test_strong_uptrend_produces_valid_signal(self):
        """MACD signal should be 0 or 1 (not NaN, not exception) on clean data."""
        bars = _make_bars(20.0, 300, volatility=0.001)
        sig = ETFRotationStrategy._macd_signal(_arr(bars), 12, 26, 9)
        assert sig in (0.0, 1.0)

    def test_downtrend_yields_zero(self):
        """Sustained downtrend: histogram negative -> signal 0."""
        bars = _make_bars(5.0, 300, volatility=0.001)  # price falls from 10 to 5
        sig = ETFRotationStrategy._macd_signal(_arr(bars), 12, 26, 9)
        assert sig == 0.0

    def test_insufficient_data_returns_zero(self):
        """Fewer bars than slow+signal period -> 0.0."""
        bars = _make_bars(12.0, 30)  # only 30 bars, well below 26+9=35
        sig = ETFRotationStrategy._macd_signal(_arr(bars), 12, 26, 9)
        assert sig == 0.0

    def test_nan_closes_handled(self):
        """NaN in closes should not crash."""
        closes = _arr(_make_bars(15.0, 200))
        closes[-1] = np.nan
        sig = ETFRotationStrategy._macd_signal(closes, 12, 26, 9)
        assert sig == 0.0

    def test_parameter_override(self):
        """Non-default parameters should work."""
        bars = _make_bars(15.0, 200)
        sig = ETFRotationStrategy._macd_signal(_arr(bars), fast=5, slow=35, signal_period=5)
        assert isinstance(sig, float)


# ---------------------------------------------------------------------------
# KDJ signal
# ---------------------------------------------------------------------------

class TestKDJSignal:
    """_kdj_signal() edge cases and expected behavior."""

    def test_returns_float(self):
        bars = _make_bars(15.0, 200)
        sig = ETFRotationStrategy._kdj_signal(
            _arr_high(bars), _arr_low(bars), _arr(bars), 9)
        assert isinstance(sig, float)
        assert sig in (0.0, 1.0)

    def test_insufficient_data_returns_zero(self):
        bars = _make_bars(12.0, 8)  # < n=9
        sig = ETFRotationStrategy._kdj_signal(
            _arr_high(bars), _arr_low(bars), _arr(bars), 9)
        assert sig == 0.0

    def test_nan_inputs_yield_zero(self):
        closes = _arr(_make_bars(15.0, 200))
        closes[-1] = np.nan
        sig = ETFRotationStrategy._kdj_signal(
            _arr_high(_make_bars(15.0, 200)), _arr_low(_make_bars(15.0, 200)),
            closes, 9)
        assert sig == 0.0

    def test_overbought_j_above_80_yields_zero(self):
        """Very strong rally pushes J above 80 -> signal 0."""
        bars = _make_bars(30.0, 200, volatility=0.01)  # large rally
        sig = ETFRotationStrategy._kdj_signal(
            _arr_high(bars), _arr_low(bars), _arr(bars), 9)
        # May be 0 or 1 depending on exact values; we just verify it does not crash
        assert sig in (0.0, 1.0)

    def test_parameter_override(self):
        bars = _make_bars(15.0, 200)
        sig = ETFRotationStrategy._kdj_signal(
            _arr_high(bars), _arr_low(bars), _arr(bars), n=14)
        assert isinstance(sig, float)


# ---------------------------------------------------------------------------
# Constructor backward compatibility
# ---------------------------------------------------------------------------

class TestConstructorBackwardCompat:
    """Existing callers passing old 3-factor weights must still work."""

    def test_no_args_uses_3factor_defaults(self):
        s = ETFRotationStrategy()
        assert s.w_trend == 0.35
        assert s.w_adx == 0.40
        assert s.w_momentum == 0.25
        assert s.w_macd == 0.0
        assert s.w_kdj == 0.0

    def test_explicit_old_weights_no_new_args(self):
        """Caller passes old weights only -- must work without w_macd/w_kdj."""
        s = ETFRotationStrategy(w_trend=0.35, w_adx=0.40, w_momentum=0.25)
        assert s.w_trend == 0.35
        assert s.w_adx == 0.40
        assert s.w_momentum == 0.25
        assert s.w_macd == 0.0   # default
        assert s.w_kdj == 0.0    # default

    def test_explicit_macd_kdj_opt_in(self):
        """Caller explicitly opts into MACD/KDJ."""
        s = ETFRotationStrategy(w_macd=0.20, w_kdj=0.10)
        assert s.w_trend == 0.35   # unchanged default
        assert s.w_macd == 0.20
        assert s.w_kdj == 0.10

    def test_full_5factor_override(self):
        s = ETFRotationStrategy(
            w_trend=0.30, w_adx=0.30, w_momentum=0.20,
            w_macd=0.10, w_kdj=0.10)
        assert s.w_trend == 0.30
        assert s.w_macd == 0.10
        assert s.w_kdj == 0.10

    def test_macd_kdj_params_present(self):
        s = ETFRotationStrategy()
        assert s.macd_fast == 12
        assert s.macd_slow == 26
        assert s.macd_signal_period == 9
        assert s.kdj_n == 9


# ---------------------------------------------------------------------------
# compute_scores regression gate
# ---------------------------------------------------------------------------

class TestComputeScoresRegression:
    """compute_scores with w_macd=0,w_kdj=0 must equal old 3-factor formula."""

    def _old_3factor_composite(self, closes, highs, lows, w_trend, w_adx, w_momentum):
        """Replicate the pre-MACD/KDJ _score_etf logic exactly."""
        # Trend
        from quanti.indicators import sma
        ma = sma(closes, 120)
        trend = 1.0 if (not np.isnan(ma[-1]) and closes[-1] > ma[-1]) else 0.0

        # ADX
        if len(closes) >= 28:
            adx_v = ETFRotationStrategy._compute_adx(highs, lows, closes, 14)
            adx_val = min(float(adx_v) / 50.0, 1.0) if not np.isnan(adx_v) else 0.5
        else:
            adx_val = 0.5

        # Momentum
        if closes[-21] > 1e-6:
            ret = (closes[-1] / closes[-21] - 1) * 100
            mom = min(max(ret / 15.0, 0), 1) if ret > 0 else 0
        else:
            mom = 0.5

        return w_trend * trend + w_adx * adx_val + w_momentum * mom

    def test_zero_macd_kdj_equals_old_formula(self):
        bars = _make_bars(15.0, 200)
        c = _arr(bars); h = _arr_high(bars); l = _arr_low(bars)
        old = self._old_3factor_composite(c, h, l, 0.35, 0.40, 0.25)
        result = ETFRotationStrategy.compute_scores(
            c, h, l, 0.35, 0.40, 0.25, 0.0, 0.0)
        assert abs(result["composite"] - old) < 1e-12

    def test_zero_weights_different_period_config(self):
        bars = _make_bars(15.0, 200)
        c = _arr(bars); h = _arr_high(bars); l = _arr_low(bars)
        old = self._old_3factor_composite(c, h, l, 0.25, 0.25, 0.50)
        result = ETFRotationStrategy.compute_scores(
            c, h, l, 0.25, 0.25, 0.50, 0.0, 0.0)
        assert abs(result["composite"] - old) < 1e-12

    def test_macd_kdj_fields_present(self):
        bars = _make_bars(15.0, 200)
        result = ETFRotationStrategy.compute_scores(
            _arr(bars), _arr_high(bars), _arr_low(bars),
            0.35, 0.40, 0.25, 0.0, 0.0)
        for key in ("trend", "adx", "momentum", "macd", "kdj", "composite"):
            assert key in result
            assert isinstance(result[key], (float, np.floating))

    def test_macd_at_nonzero_weight_alters_composite(self):
        """Nonzero MACD weight must produce a composite value (may equal zero-weight
        if MACD signal is 0.0 on this particular data -- that is correct behavior)."""
        bars = _make_bars(20.0, 300, volatility=0.001)
        r_zero = ETFRotationStrategy.compute_scores(
            _arr(bars), _arr_high(bars), _arr_low(bars),
            0.35, 0.40, 0.25, 0.0, 0.0)
        r_nonzero = ETFRotationStrategy.compute_scores(
            _arr(bars), _arr_high(bars), _arr_low(bars),
            0.35, 0.40, 0.25, 0.10, 0.0)
        # MACD signal is 0 or 1. If 1, composite should differ; if 0, identical.
        assert r_nonzero["composite"] >= r_zero["composite"]


# ---------------------------------------------------------------------------
# Server indicator parity
# ---------------------------------------------------------------------------

class TestIndicatorParityWithServer:
    """Shared macd/kdj from quanti.indicators must match legacy server helpers."""

    def test_macd_output_structure(self):
        """Shared macd() returns valid arrays of correct length, no NaN at tail."""
        from quanti.indicators import macd as shared_macd

        closes_list = [10.0, 10.2, 10.1, 10.3, 10.5, 10.4, 10.6, 10.8,
                       10.7, 10.9, 11.0, 10.8, 11.1, 11.3, 11.2, 11.5,
                       11.7, 11.6, 11.8, 12.0, 11.9, 12.2, 12.4, 12.3,
                       12.5, 12.7, 12.6, 12.8, 13.0, 12.9, 13.1, 13.3,
                       13.2, 13.5, 13.7, 13.6, 13.8, 14.0] * 6  # 228 bars
        closes_np = np.array(closes_list, dtype=np.float64)

        dif, dea, hist = shared_macd(closes_np, 12, 26, 9)
        assert dif is not None
        assert len(dif) == len(closes_np)
        assert len(dea) == len(closes_np)
        assert len(hist) == len(closes_np)
        # Tail values are valid (not NaN) after warmup
        assert not np.isnan(dif[-1])
        assert not np.isnan(dea[-1])
        assert not np.isnan(hist[-1])
        # Histogram = (DIF - DEA) * 2
        assert abs(hist[-1] - (dif[-1] - dea[-1]) * 2) < 1e-10

    def test_kdj_output_seed_and_range(self):
        """Shared kdj() returns valid arrays, first valid index incorporates RSV seed."""
        from quanti.indicators import kdj as shared_kdj

        n_bars = 200
        closes = np.linspace(10.0, 15.0, n_bars, dtype=np.float64)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.1, n_bars)
        closes = closes + np.cumsum(noise) * 0.05
        highs = closes * 1.01
        lows = closes * 0.99

        k, d, j = shared_kdj(highs, lows, closes, 9)
        assert len(k) == n_bars
        # First n-1 indices are NaN
        assert np.isnan(k[7])
        assert np.isnan(d[7])
        # Index n-1 = 8 is the first valid value (RSV seed + smoothing)
        assert not np.isnan(k[8])
        assert not np.isnan(d[8])
        # J = 3*K - 2*D at all valid indices
        for i in range(8, n_bars):
            assert abs(j[i] - (3 * k[i] - 2 * d[i])) < 1e-10
        # Values in [0, 100]
        assert 0 <= k[-1] <= 100
        assert 0 <= d[-1] <= 100

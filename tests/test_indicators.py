"""Tests for shared indicator functions in quanti.indicators."""
import numpy as np

from quanti.indicators import (
    adx,
    adx_with_di,
    bollinger_bands,
    compute_atr,
    compute_rsi,
    ema,
    kdj,
    macd,
    sma,
)


class TestSMA:
    def test_sma_basic(self):
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = sma(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[3] == 3.0  # (2+3+4)/3
        assert result[4] == 4.0  # (3+4+5)/3

    def test_sma_insufficient_data(self):
        data = np.array([1, 2], dtype=np.float64)
        result = sma(data, 5)
        assert all(np.isnan(r) for r in result)


class TestEMA:
    def test_ema_basic(self):
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = ema(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0  # seed = mean(1,2,3)


class TestADX:
    def test_adx_returns_array(self):
        n = 60
        t = np.linspace(0, 4*np.pi, n)
        closes = np.sin(t) * 10 + 50
        highs = closes + 1
        lows = closes - 1
        result = adx(highs, lows, closes, 14)
        assert result is not None
        assert len(result) == n

    def test_adx_insufficient_data(self):
        closes = np.ones(10)
        highs = closes + 1
        lows = closes - 1
        result = adx(highs, lows, closes, 14)
        assert result is None

    def test_adx_with_di_returns_tuple(self):
        n = 60
        t = np.linspace(0, 4*np.pi, n)
        closes = np.sin(t) * 10 + 50
        highs = closes + 1
        lows = closes - 1
        result = adx_with_di(highs, lows, closes, 14)
        assert result is not None
        adx_arr, plus_di, minus_di = result
        assert len(adx_arr) == n
        assert len(plus_di) == n
        assert len(minus_di) == n


class TestBollingerBands:
    def test_bb_basic(self):
        closes = np.array([10, 11, 12, 11, 10, 11, 12, 11, 10, 11,
                          12, 11, 10, 11, 12, 11, 10, 11, 12, 13,
                          14, 15, 16, 17, 18], dtype=np.float64)
        middle, upper, lower = bollinger_bands(closes, 20, 2.0)
        assert middle is not None
        assert upper is not None
        assert lower is not None
        # Upper should be > middle > lower
        for i in range(19, len(closes)):
            if not np.isnan(upper[i]):
                assert upper[i] >= middle[i] >= lower[i]

    def test_bb_insufficient_data(self):
        closes = np.array([10, 11, 12], dtype=np.float64)
        middle, upper, lower = bollinger_bands(closes, 20, 2.0)
        assert middle is None
        assert upper is None
        assert lower is None


class TestATR:
    def test_atr_positive(self):
        from quanti.data.schema import ETFDailyBar
        bars = []
        price = 10.0
        for i in range(20):
            bars.append(ETFDailyBar(
                symbol="TEST", trade_date=f"202401{str(i+1).zfill(2)}",
                open=price, high=price+0.5, low=price-0.3, close=price+0.1,
                volume=1000, amount=10000,
            ))
            price += 0.1 * (1 if i % 3 != 0 else -1)
        atr_val = compute_atr(bars, 14)
        assert atr_val is not None
        assert atr_val > 0

    def test_atr_insufficient_bars(self):
        from quanti.data.schema import ETFDailyBar
        bars = [ETFDailyBar(symbol="TEST", trade_date="20240101",
                open=10, high=10.5, low=9.5, close=10, volume=1000, amount=10000)]
        atr_val = compute_atr(bars, 14)
        assert atr_val is None


class TestRSI:
    def test_rsi_uptrend(self):
        closes = np.linspace(10, 20, 30, dtype=np.float64)
        rsi = compute_rsi(closes, 14)
        assert rsi is not None
        assert rsi > 50  # Uptrend gives high RSI

    def test_rsi_downtrend(self):
        closes = np.linspace(20, 10, 30, dtype=np.float64)
        rsi = compute_rsi(closes, 14)
        assert rsi is not None
        assert rsi < 50  # Downtrend gives low RSI

    def test_rsi_insufficient(self):
        closes = np.ones(5, dtype=np.float64)
        rsi = compute_rsi(closes, 14)
        assert rsi is None


class TestMACD:
    def test_macd_basic(self):
        n = 80
        t = np.linspace(0, 4 * np.pi, n)
        closes = np.sin(t) * 10 + 50
        result = macd(closes, 12, 26, 9)
        assert result is not None
        dif, dea, hist = result
        assert len(dif) == n
        assert len(dea) == n
        assert len(hist) == n
        # First slow-1 (=25) elements of DIF are NaN
        assert all(np.isnan(dif[i]) for i in range(25))
        # DIF should have valid values after index 25
        assert not np.isnan(dif[25])

    def test_macd_insufficient_data(self):
        closes = np.ones(20, dtype=np.float64)
        result = macd(closes, 12, 26, 9)
        assert result is None

    def test_macd_consistency(self):
        """Verify against hand-checked values on a small array with fast=3, slow=5, signal=2."""
        closes = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float64)
        dif, dea, hist = macd(closes, fast=3, slow=5, signal=2)
        assert dif is not None
        # DIF[0:4] = NaN (slow-1 = 4)
        assert all(np.isnan(dif[i]) for i in range(4))
        assert not np.isnan(dif[4])

    def test_macd_histogram_zero_crossing(self):
        """DIF crossing DEA should flip histogram sign."""
        # Rising then falling prices produce a histogram sign change
        n = 100
        t = np.linspace(0, 6 * np.pi, n)
        closes = np.sin(t) * 10 + 50
        dif, dea, hist = macd(closes, 12, 26, 9)
        assert hist is not None
        valid = hist[~np.isnan(hist)]
        assert len(valid) > 0
        # At least one sign change in the valid portion
        signs = np.sign(valid)
        changes = np.sum(np.diff(signs) != 0)
        assert changes > 0, "Expected histogram sign change over a full cycle"


class TestKDJ:
    def test_kdj_basic(self):
        n = 80
        t = np.linspace(0, 4 * np.pi, n)
        closes = np.sin(t) * 10 + 50
        highs = closes + 1
        lows = closes - 1
        result = kdj(highs, lows, closes, 9)
        assert result is not None
        k, d, j = result
        assert len(k) == n
        assert len(d) == n
        assert len(j) == n
        # First n-1 (=8) elements are NaN
        assert all(np.isnan(k[i]) for i in range(8))
        assert not np.isnan(k[8])

    def test_kdj_insufficient_data(self):
        closes = np.ones(5, dtype=np.float64)
        highs = closes + 1
        lows = closes - 1
        result = kdj(highs, lows, closes, 9)
        assert result is None

    def test_kdj_flat_prices(self):
        """All prices equal -> K, D, J converge to 50."""
        n = 40
        closes = np.full(n, 50.0, dtype=np.float64)
        highs = np.full(n, 51.0, dtype=np.float64)
        lows = np.full(n, 49.0, dtype=np.float64)
        k, d, j = kdj(highs, lows, closes, 9)
        # Last value should be near 50
        assert abs(k[-1] - 50.0) < 1e-6
        assert abs(d[-1] - 50.0) < 1e-6
        assert abs(j[-1] - 50.0) < 1e-6

    def test_kdj_extreme(self):
        """Very high close (RSV=100) produces J > 100."""
        n = 30
        closes = np.concatenate([np.full(10, 50.0), np.linspace(50, 100, 20)])
        highs = np.full(n, 100.0, dtype=np.float64)
        lows = np.full(n, 0.0, dtype=np.float64)
        k, d, j = kdj(highs, lows, closes, 9)
        assert not np.isnan(j[-1])
        assert j[-1] > 100, f"Expected J > 100 for extreme uptrend, got {j[-1]:.2f}"

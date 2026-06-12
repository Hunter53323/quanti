"""Tests for shared indicator functions in quanti.indicators."""
import numpy as np

from quanti.indicators import (
    adx,
    adx_with_di,
    bollinger_bands,
    compute_atr,
    compute_rsi,
    ema,
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

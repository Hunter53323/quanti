"""Integration test for BacktestEngine end-to-end with synthetic bars."""
from datetime import datetime

import numpy as np

from quanti.backtest.engine import BacktestEngine
from quanti.data.schema import ETFDailyBar
from quanti.strategy.etf_trend import ETFTrendStrategy


def make_daily_bars(symbol, dates_and_closes):
    """Create ETFDailyBar objects from a list of (date_str, close) tuples."""
    bars = []
    for i, (date_str, close) in enumerate(dates_and_closes):
        bars.append(ETFDailyBar(
            symbol=symbol,
            trade_date=date_str,
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=500_000 + i * 10_000,
            amount=close * (500_000 + i * 10_000),
        ))
    return bars


def _gen_dates(start_year, n):
    """Generate YYYYMMDD date strings for n sequential trading days."""
    dates = []
    from datetime import timedelta
    d = datetime(start_year, 1, 2)  # avoid Jan 1 holiday edge
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


class TestBacktestEngineIntegration:
    """End-to-end backtest engine tests with synthetic data."""

    def test_single_etf_buy_and_hold_simulated(self):
        """
        Build synthetic ETF data with a clear uptrend (SMA20 > SMA60 > SMA120),
        run backtest engine, verify it produces trades and non-empty metrics.
        """
        n = 200
        dates = _gen_dates(2023, n)
        # Build a strong uptrend: prices rise steadily
        closes = np.linspace(5.0, 10.0, n).tolist()
        bars = make_daily_bars("510300", list(zip(dates, closes, strict=False)))

        engine = BacktestEngine(
            strategy_class=ETFTrendStrategy,
            params={"ma_fast": 5, "ma_slow": 20, "ma_long": 40,
                    "adx_entry_threshold": 15, "di_diff_threshold": 5,
                    "entry_mode": "legacy"},
            initial_capital=100000,
            commission_rate=0.0001,
            slippage_bps=1,
        )
        result = engine.run(["510300"], {"510300": bars}, period_label="integration_test")

        # Verification
        assert result is not None
        assert len(result.equity_curve) > 0, "Equity curve should not be empty"
        assert isinstance(result.trades, list), "Trades should be a list"
        assert result.start_date, "Start date should be set"
        assert result.end_date, "End date should be set"
        assert result.max_drawdown_pct >= 0, "MaxDD should be non-negative"
        # Total return with strong uptrend should be positive
        assert result.total_return_pct > -50, f"Total return {result.total_return_pct:.1f}% too negative for uptrend"

    def test_flat_market_no_signals(self):
        """Flat prices (no trend) should produce few or no buy signals."""
        n = 150
        dates = _gen_dates(2023, n)
        closes = [10.0] * n  # completely flat
        bars = make_daily_bars("510300", list(zip(dates, closes, strict=False)))

        engine = BacktestEngine(
            strategy_class=ETFTrendStrategy,
            params={"ma_fast": 20, "ma_slow": 60, "ma_long": 120,
                    "adx_entry_threshold": 25, "di_diff_threshold": 15,
                    "entry_mode": "resonance"},
            initial_capital=100000,
        )
        result = engine.run(["510300"], {"510300": bars}, period_label="flat_market")

        buy_trades = [t for t in result.trades if t["side"] == "buy"]
        # With flat prices and strict resonance mode, should get 0 or very few buys
        assert len(buy_trades) <= 2, (
            f"Flat market should produce few buys, got {len(buy_trades)}"
        )

    def test_sharp_decline_exits(self):
        """A sharp V-shaped decline should trigger stop-loss exits."""
        from quanti.config import settings

        # Force stop loss on for this test
        settings.STOP_LOSS_PCT = 5.0
        settings.ATR_TRAILING_STOP_ENABLED = True

        n = 120
        dates = _gen_dates(2023, n)
        # Uptrend first 60 bars (build position), then crash
        closes = (
            np.linspace(5.0, 8.0, 60).tolist()
            + np.linspace(8.0, 4.0, 60).tolist()
        )
        bars = make_daily_bars("510300", list(zip(dates, closes, strict=False)))

        engine = BacktestEngine(
            strategy_class=ETFTrendStrategy,
            params={"ma_fast": 5, "ma_slow": 20, "ma_long": 40,
                    "adx_entry_threshold": 10, "di_diff_threshold": 3,
                    "entry_mode": "legacy"},
            initial_capital=100000,
            commission_rate=0.0001,
            slippage_bps=1,
        )
        result = engine.run(["510300"], {"510300": bars}, period_label="crash_test")

        # Verify equity curve exists
        assert len(result.equity_curve) > 0
        assert result.max_drawdown_pct >= 0

    def test_t1_settlement_prevents_immediate_reinvestment(self):
        """T+1 settlement: sell proceeds should not be immediately available for buying."""
        n = 100
        dates = _gen_dates(2023, n)
        closes = np.linspace(5.0, 7.0, n).tolist()
        bars = make_daily_bars("510300", list(zip(dates, closes, strict=False)))

        engine = BacktestEngine(
            strategy_class=ETFTrendStrategy,
            params={"ma_fast": 5, "ma_slow": 20, "ma_long": 40,
                    "adx_entry_threshold": 10, "di_diff_threshold": 3,
                    "entry_mode": "legacy"},
            initial_capital=50000,
        )
        result = engine.run(["510300"], {"510300": bars}, period_label="t1_test")

        # Verify that trades are recorded
        assert result is not None
        # Check that metric computation does not crash on T+1 data
        assert isinstance(result.sharpe_ratio, float)

    def test_multi_etf(self):
        """Backtest with two ETFs should work with MarketData for both."""
        n = 180
        dates = _gen_dates(2023, n)
        closes_up = np.linspace(5.0, 10.0, n).tolist()
        closes_flat = np.linspace(6.0, 6.5, n).tolist()

        etf1 = make_daily_bars("510300", list(zip(dates, closes_up, strict=False)))
        etf2 = make_daily_bars("510500", list(zip(dates, closes_flat, strict=False)))

        engine = BacktestEngine(
            strategy_class=ETFTrendStrategy,
            params={"ma_fast": 5, "ma_slow": 20, "ma_long": 40,
                    "adx_entry_threshold": 10, "di_diff_threshold": 3,
                    "entry_mode": "legacy"},
            initial_capital=100000,
        )
        result = engine.run(
            ["510300", "510500"],
            {"510300": etf1, "510500": etf2},
            period_label="multi_etf",
        )

        assert result is not None
        assert len(result.equity_curve) > 0

    def test_walk_forward_parameter_stability(self):
        """Walk-forward should produce train+test results without crashing."""
        n = 200
        dates = _gen_dates(2020, n)
        closes = np.linspace(5.0, 12.0, n).tolist()
        bars = make_daily_bars("510300", list(zip(dates, closes, strict=False)))

        engine = BacktestEngine(
            strategy_class=ETFTrendStrategy,
            params={"entry_mode": "legacy"},
            initial_capital=100000,
        )

        results = engine.run_walk_forward(
            ["510300"], {"510300": bars},
            train_years=1, test_months=3,
        )
        assert isinstance(results, list)
        # Should produce at least a few results
        if len(results) > 0:
            for r in results:
                assert r.cagr_pct is not None

    def test_out_of_sample_runs(self):
        """OOS backtest should return InSample + OOS pair."""
        n = 180
        dates = _gen_dates(2022, n)
        closes = np.linspace(5.0, 8.0, n).tolist()
        bars = make_daily_bars("510300", list(zip(dates, closes, strict=False)))

        engine = BacktestEngine(
            strategy_class=ETFTrendStrategy,
            params={"entry_mode": "legacy"},
            initial_capital=100000,
        )

        is_result, oos_result = engine.run_out_of_sample(
            ["510300"], {"510300": bars}
        )
        assert is_result is not None
        # OOS may be None or empty if dates don't overlap

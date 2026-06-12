"""Tests for circuit breakers in circuit_breaker.py."""

from quanti.execution.circuit_breaker import (
    BreakerManager,
    CircuitBreaker,
    ConsecutiveLossBreaker,
    MonthlyDrawdownBreaker,
)


class TestCircuitBreaker:
    """Original CircuitBreaker: consecutive failures, daily drawdown, data freshness."""

    def test_starts_active(self):
        cb = CircuitBreaker()
        assert cb.is_active is True
        assert cb.check() is True

    def test_trip_on_consecutive_failures(self):
        cb = CircuitBreaker()
        cb.record_execution_failure("test error")
        cb.record_execution_failure("test error")
        cb.record_execution_failure("test error")  # trips at MAX_CONSECUTIVE_FAILURES=3
        assert cb.is_active is False
        assert cb.check() is False
        assert "3 consecutive" in cb.trip_reason

    def test_success_resets_consecutive(self):
        cb = CircuitBreaker()
        cb.record_execution_failure("err1")
        cb.record_execution_failure("err2")
        cb.record_execution_success()  # reset
        cb.record_execution_failure("err3")
        assert cb.is_active is True  # only 1 failure after reset

    def test_manual_reset(self):
        cb = CircuitBreaker()
        cb.trip("test trip")
        assert cb.is_active is False
        cb.reset()
        assert cb.is_active is True
        assert cb.trip_reason == ""

    def test_trip_is_idempotent(self):
        cb = CircuitBreaker()
        cb.trip("first")
        reason1 = cb.trip_reason
        cb.trip("second")  # should not change
        assert cb.trip_reason == reason1


class TestMonthlyDrawdownBreaker:
    """Monthly drawdown tracking and auto-reset."""

    def test_starts_active(self):
        mb = MonthlyDrawdownBreaker(total_capital=100000.0, max_monthly_drawdown_pct=0.05)
        assert mb.is_active is True

    def test_small_loss_no_trip(self):
        mb = MonthlyDrawdownBreaker(total_capital=100000.0, max_monthly_drawdown_pct=0.05)
        mb.record_trade_pnl(-2000.0)  # 2% of capital, under 5% limit
        assert mb.is_active is True
        assert mb.monthly_pnl == -2000.0

    def test_large_loss_trips(self):
        mb = MonthlyDrawdownBreaker(total_capital=100000.0, max_monthly_drawdown_pct=0.05)
        mb.record_trade_pnl(-6000.0)  # 6% > 5% limit
        assert mb.is_active is False
        assert "Monthly loss" in mb.trip_reason

    def test_cumulative_loss_trips(self):
        mb = MonthlyDrawdownBreaker(total_capital=100000.0, max_monthly_drawdown_pct=0.05)
        mb.record_trade_pnl(-3000.0)
        assert mb.is_active is True
        mb.record_trade_pnl(-3000.0)  # total -6000 > 5000 limit
        assert mb.is_active is False

    def test_manual_reset(self):
        mb = MonthlyDrawdownBreaker(total_capital=100000.0, max_monthly_drawdown_pct=0.05)
        mb.record_trade_pnl(-6000.0)
        assert mb.is_active is False
        mb.reset()
        assert mb.is_active is True
        assert mb.monthly_pnl == 0.0

    def test_check_method(self):
        mb = MonthlyDrawdownBreaker(total_capital=100000.0, max_monthly_drawdown_pct=0.05)
        assert mb.check() is True
        mb.record_trade_pnl(-6000.0)
        assert mb.check() is False

    def test_custom_parameters(self):
        mb = MonthlyDrawdownBreaker(total_capital=50000.0, max_monthly_drawdown_pct=0.10)
        mb.record_trade_pnl(-4000.0)  # 8% < 10%
        assert mb.is_active is True
        mb.record_trade_pnl(-2000.0)  # 12% > 10%
        assert mb.is_active is False


class TestConsecutiveLossBreaker:
    """Consecutive stop-loss tracking with cooldown."""

    def test_starts_active(self):
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=3)
        assert cl.is_active is True

    def test_profits_do_not_count(self):
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=3)
        cl.record_profit('TEST', 100.0)
        assert cl.consecutive_losses == 0

    def test_losses_accumulate(self):
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=3)
        cl.record_loss('A', -100.0)
        assert cl.consecutive_losses == 1
        cl.record_loss('B', -200.0)
        assert cl.consecutive_losses == 2
        assert cl.is_active is True

    def test_loss_limit_trips(self):
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=3)
        cl.record_loss('A', -100.0)
        cl.record_loss('B', -200.0)
        cl.record_loss('C', -150.0)
        assert cl.is_active is False
        assert "3 consecutive stop-losses" in cl.trip_reason

    def test_profit_resets_counter(self):
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=3)
        cl.record_loss('A', -100.0)
        cl.record_loss('B', -200.0)
        cl.record_profit('C', 50.0)  # profit resets counter
        assert cl.consecutive_losses == 0

    def test_record_profit_with_negative_pnl(self):
        """record_profit with negative PnL should call record_loss."""
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=3)
        cl.record_profit('A', -100.0)  # negative = loss
        assert cl.consecutive_losses == 1

    def test_manual_reset(self):
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=3)
        cl.record_loss('A', -100.0)
        cl.record_loss('B', -200.0)
        cl.record_loss('C', -150.0)
        assert cl.is_active is False
        cl.reset()
        assert cl.is_active is True
        assert cl.consecutive_losses == 0

    def test_check_still_blocked_in_cooldown(self):
        """After tripping, check() returns False during cooldown."""
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=3, cooldown_trading_days=3)
        cl.record_loss('A', -100.0)
        cl.record_loss('B', -200.0)
        cl.record_loss('C', -150.0)
        assert cl.check() is False  # still in cooldown

    def test_custom_limit(self):
        cl = ConsecutiveLossBreaker(consecutive_loss_limit=5)
        for i in range(4):
            cl.record_loss(f'SYM{i}', -100.0)
        assert cl.is_active is True
        cl.record_loss('SYM5', -100.0)
        assert cl.is_active is False


class TestBreakerManager:
    """Composite breaker manager wrapping all breakers."""

    def test_starts_all_active(self):
        bm = BreakerManager()
        assert bm.check_all() is True

    def test_monthly_trip_halts_all(self):
        bm = BreakerManager()
        bm.monthly_drawdown.record_trade_pnl(-100000.0)  # massive loss
        assert bm.check_all() is False

    def test_consecutive_loss_trip_halts_all(self):
        bm = BreakerManager()
        for i in range(5):
            bm.consecutive_loss.record_loss(f'SYM{i}', -100.0)
        assert bm.check_all() is False

    def test_general_trip_halts_all(self):
        bm = BreakerManager()
        bm.general.trip("test")
        assert bm.check_all() is False

    def test_status_contains_all(self):
        bm = BreakerManager()
        s = bm.status()
        assert 'general' in s
        assert 'monthly_drawdown' in s
        assert 'consecutive_loss' in s

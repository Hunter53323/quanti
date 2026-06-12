"""
Circuit breakers that halt trading on adverse conditions.

Breaker types:
1. ConsecutiveFailureBreaker: 3+ consecutive execution failures -> halt
2. DailyDrawdownBreaker: Single-day realized loss > 2% -> halt
3. DataFeedBreaker: Data feed gap > 5 minutes -> halt
4. MonthlyDrawdownBreaker: Monthly loss > MONTHLY_MAX_DRAWDOWN_PCT -> halt
5. ConsecutiveLossBreaker: N consecutive stop-losses -> halt with cooldown

All breakers require manual re-enable after trip (except MonthlyDrawdownBreaker
which auto-resets on the 1st of the next month).
"""
from datetime import datetime, timedelta
from enum import Enum

from quanti.config import settings
from quanti.monitor.alerts import AlertLevel, get_alerter


class BreakerState(Enum):
    ACTIVE = "active"      # Trading allowed
    TRIPPED = "tripped"    # Trading halted


class CircuitBreaker:
    """
    Monitors trading conditions and trips breakers when thresholds are breached.
    Once tripped, a breaker requires manual reset.
    """

    def __init__(self):
        self._state = BreakerState.ACTIVE
        self._trip_reason = ""
        self._trip_time = None
        self._consecutive_failures = 0
        self._daily_pnl = 0.0
        self._last_data_ts = datetime.now()
        self._alerter = get_alerter()

    @property
    def is_active(self):
        return self._state == BreakerState.ACTIVE

    @property
    def trip_reason(self):
        return self._trip_reason

    def check(self):
        """Check all rules. Returns True if trading is allowed."""
        return self._state != BreakerState.TRIPPED

    def record_execution_failure(self, error: str = ""):
        self._consecutive_failures += 1
        if self._consecutive_failures >= settings.MAX_CONSECUTIVE_FAILURES:
            self.trip(f"{self._consecutive_failures} consecutive execution failures. Last: {error}")

    def record_execution_success(self):
        self._consecutive_failures = 0

    def update_pnl(self, daily_pnl: float):
        self._daily_pnl = daily_pnl
        loss_pct = abs(daily_pnl) / settings.TOTAL_CAPITAL
        if daily_pnl < 0 and loss_pct > settings.MAX_SINGLE_DAY_DRAWDOWN_PCT:
            self.trip(f"Single-day loss {loss_pct*100:.2f}% exceeds {settings.MAX_SINGLE_DAY_DRAWDOWN_PCT*100:.1f}% limit")

    def record_data_feed(self, timestamp: datetime):
        self._last_data_ts = timestamp

    def check_data_freshness(self):
        gap_sec = (datetime.now() - self._last_data_ts).total_seconds()
        if gap_sec > 300:  # 5 minutes
            self.trip(f"Data feed gap: {gap_sec:.0f}s since last data")

    def trip(self, reason: str):
        if self._state == BreakerState.TRIPPED:
            return  # Already tripped
        self._state = BreakerState.TRIPPED
        self._trip_reason = reason
        self._trip_time = datetime.now()
        self._alerter.send(AlertLevel.CRITICAL, "Circuit Breaker TRIPPED", reason)

    def reset(self):
        self._state = BreakerState.ACTIVE
        self._trip_reason = ""
        self._consecutive_failures = 0
        print(f"[CIRCUIT BREAKER] Reset at {datetime.now().isoformat()}")

    def status(self) -> dict:
        return dict(
            state=self._state.value,
            trip_reason=self._trip_reason,
            trip_time=self._trip_time.isoformat() if self._trip_time else None,
            consecutive_failures=self._consecutive_failures,
            daily_pnl=self._daily_pnl,
        )


class MonthlyDrawdownBreaker:
    """
    Tracks realized P&L per calendar month.
    Trips when monthly realized loss exceeds MONTHLY_MAX_DRAWDOWN_PCT * TOTAL_CAPITAL.
    Auto-resets on the 1st of the next month.
    """

    def __init__(
        self,
        total_capital: float | None = None,
        max_monthly_drawdown_pct: float | None = None,
    ):
        self.total_capital = total_capital or settings.TOTAL_CAPITAL
        self.max_drawdown_pct = max_monthly_drawdown_pct or settings.MONTHLY_MAX_DRAWDOWN_PCT
        self._state = BreakerState.ACTIVE
        self._trip_reason = ""
        self._trip_time: datetime | None = None
        self._current_month: int | None = None
        self._monthly_pnl: float = 0.0
        self._alerter = get_alerter()

    @property
    def is_active(self) -> bool:
        return self._state == BreakerState.ACTIVE

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    @property
    def monthly_pnl(self) -> float:
        return self._monthly_pnl

    def check(self) -> bool:
        """Check auto-reset and tripped state. Returns True if trading allowed."""
        self._check_month_rollover()
        return self._state != BreakerState.TRIPPED

    def record_trade_pnl(self, pnl: float) -> None:
        """
        Record realized P&L from a closed trade.
        Trips the breaker if monthly loss exceeds threshold.

        Args:
            pnl: Realized profit/loss from the trade (positive = profit).
        """
        self._check_month_rollover()
        self._monthly_pnl += pnl

        loss_limit = self.total_capital * self.max_drawdown_pct
        if self._monthly_pnl < -loss_limit:
            self.trip(
                f"Monthly loss {-self._monthly_pnl:.2f} exceeds limit {loss_limit:.2f} "
                f"({self.max_drawdown_pct*100:.1f}% of capital)"
            )

    def trip(self, reason: str) -> None:
        if self._state == BreakerState.TRIPPED:
            return
        self._state = BreakerState.TRIPPED
        self._trip_reason = reason
        self._trip_time = datetime.now()
        self._alerter.send(AlertLevel.CRITICAL, "Monthly Drawdown Breaker TRIPPED", reason)

    def reset(self) -> None:
        """Manual reset (also auto-resets on month rollover)."""
        self._state = BreakerState.ACTIVE
        self._trip_reason = ""
        self._monthly_pnl = 0.0
        print(f"[MONTHLY DRAWDOWN BREAKER] Reset at {datetime.now().isoformat()}")

    def save_state(self) -> dict:
        """Serialize breaker state for journal persistence."""
        return dict(
            state=self._state.value,
            trip_reason=self._trip_reason,
            trip_time=self._trip_time.isoformat() if self._trip_time else None,
            monthly_pnl=self._monthly_pnl,
            current_month=self._current_month,
        )

    @classmethod
    def restore_state(cls, state: dict, total_capital: float | None = None,
                      max_monthly_drawdown_pct: float | None = None) -> "MonthlyDrawdownBreaker":
        """Restore breaker from persisted state."""
        obj = cls(total_capital=total_capital, max_monthly_drawdown_pct=max_monthly_drawdown_pct)
        obj._state = BreakerState(state["state"])
        obj._trip_reason = state.get("trip_reason", "")
        trip_time = state.get("trip_time")
        obj._trip_time = datetime.fromisoformat(trip_time) if trip_time else None
        obj._monthly_pnl = state.get("monthly_pnl", 0.0)
        obj._current_month = state.get("current_month")
        return obj

    def _check_month_rollover(self) -> None:
        """Auto-reset when calendar month changes."""
        now = datetime.now()
        if self._current_month is not None and now.month != self._current_month:
            self._state = BreakerState.ACTIVE
            self._trip_reason = ""
            self._monthly_pnl = 0.0
        self._current_month = now.month

    def status(self) -> dict:
        return dict(
            state=self._state.value,
            trip_reason=self._trip_reason,
            trip_time=self._trip_time.isoformat() if self._trip_time else None,
            monthly_pnl=self._monthly_pnl,
            current_month=self._current_month,
            loss_limit=self.total_capital * self.max_drawdown_pct,
        )


class ConsecutiveLossBreaker:
    """
    Tracks consecutive stop-loss exits.
    Trips after CONSECUTIVE_LOSS_LIMIT consecutive losses.
    Has a cooldown period of 3 trading days after trip.

    A "loss" is defined as a position exit where the realized P&L was negative.
    """

    def __init__(
        self,
        consecutive_loss_limit: int | None = None,
        cooldown_trading_days: int = 3,
    ):
        self.loss_limit = consecutive_loss_limit or settings.CONSECUTIVE_LOSS_LIMIT
        self.cooldown_days = cooldown_trading_days
        self._state = BreakerState.ACTIVE
        self._trip_reason = ""
        self._trip_time: datetime | None = None
        self._consecutive_losses: int = 0
        self._loss_history: list[tuple[datetime, str, float]] = []
        self._cooldown_end: datetime | None = None
        self._alerter = get_alerter()

    @property
    def is_active(self) -> bool:
        return self._state == BreakerState.ACTIVE

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def check(self) -> bool:
        """
        Check if trading is allowed. Returns True if active.
        Handles cooldown expiry (auto-reset after cooldown_days).
        """
        if self._state == BreakerState.ACTIVE:
            return True

        # Check if cooldown has expired
        if self._cooldown_end and datetime.now() >= self._cooldown_end:
            self._state = BreakerState.ACTIVE
            self._trip_reason = ""
            self._consecutive_losses = 0
            print(f"[CONSECUTIVE LOSS BREAKER] Cooldown expired, auto-reset at {datetime.now().isoformat()}")
            return True

        return False

    def record_loss(self, symbol: str, pnl: float) -> None:
        """
        Record a losing trade. Trips the breaker if consecutive loss limit
        is reached.

        Args:
            symbol: The symbol that was exited at a loss.
            pnl: Realized P&L (negative for a loss).
        """
        if pnl >= 0:
            # Winning trade resets the counter
            self._consecutive_losses = 0
            return

        self._consecutive_losses += 1
        self._loss_history.append((datetime.now(), symbol, pnl))

        # Trim history to reasonable size
        if len(self._loss_history) > 100:
            self._loss_history = self._loss_history[-50:]

        if self._consecutive_losses >= self.loss_limit:
            self.trip(
                f"{self._consecutive_losses} consecutive stop-losses (limit: {self.loss_limit}). "
                f"Last: {symbol} {pnl:.2f}"
            )

    def record_profit(self, symbol: str, pnl: float) -> None:
        """
        Record a winning trade. Resets the consecutive loss counter.

        Args:
            symbol: The symbol that was exited at a profit.
            pnl: Realized P&L (positive for a profit).
        """
        if pnl > 0:
            self._consecutive_losses = 0
        else:
            self.record_loss(symbol, pnl)

    def trip(self, reason: str) -> None:
        if self._state == BreakerState.TRIPPED:
            return
        self._state = BreakerState.TRIPPED
        self._trip_reason = reason
        self._trip_time = datetime.now()
        self._cooldown_end = datetime.now() + timedelta(days=self.cooldown_days)
        self._alerter.send(AlertLevel.WARNING, "Consecutive Loss Breaker TRIPPED", reason)

    def reset(self) -> None:
        """Manual reset (bypass cooldown)."""
        self._state = BreakerState.ACTIVE
        self._trip_reason = ""
        self._consecutive_losses = 0
        self._cooldown_end = None
        print(f"[CONSECUTIVE LOSS BREAKER] Manual reset at {datetime.now().isoformat()}")

    def save_state(self) -> dict:
        """Serialize breaker state for journal persistence."""
        return dict(
            state=self._state.value,
            trip_reason=self._trip_reason,
            trip_time=self._trip_time.isoformat() if self._trip_time else None,
            consecutive_losses=self._consecutive_losses,
            cooldown_end=self._cooldown_end.isoformat() if self._cooldown_end else None,
        )

    @classmethod
    def restore_state(cls, state: dict, consecutive_loss_limit: int | None = None,
                      cooldown_trading_days: int = 3) -> "ConsecutiveLossBreaker":
        """Restore breaker from persisted state."""
        obj = cls(consecutive_loss_limit=consecutive_loss_limit, cooldown_trading_days=cooldown_trading_days)
        obj._state = BreakerState(state["state"])
        obj._trip_reason = state.get("trip_reason", "")
        trip_time = state.get("trip_time")
        obj._trip_time = datetime.fromisoformat(trip_time) if trip_time else None
        obj._consecutive_losses = state.get("consecutive_losses", 0)
        cooldown = state.get("cooldown_end")
        obj._cooldown_end = datetime.fromisoformat(cooldown) if cooldown else None
        return obj

    def status(self) -> dict:
        return dict(
            state=self._state.value,
            trip_reason=self._trip_reason,
            trip_time=self._trip_time.isoformat() if self._trip_time else None,
            consecutive_losses=self._consecutive_losses,
            cooldown_end=self._cooldown_end.isoformat() if self._cooldown_end else None,
            recent_losses=[
                {"time": t.isoformat(), "symbol": s, "pnl": p}
                for t, s, p in self._loss_history[-5:]
            ],
        )


# ------------------------------------------------------------------
# Composite breaker manager (optional: ties all breakers together)
# ------------------------------------------------------------------

class BreakerManager:
    """
    Composite manager that aggregates all individual circuit breakers.
    Any tripped breaker halts trading.
    """

    def __init__(self):
        self.general = CircuitBreaker()
        self.monthly_drawdown = MonthlyDrawdownBreaker()
        self.consecutive_loss = ConsecutiveLossBreaker()

    def check_all(self) -> bool:
        """Returns True only if all breakers are active."""
        return (
            self.general.check()
            and self.monthly_drawdown.check()
            and self.consecutive_loss.check()
        )

    def status(self) -> dict:
        return {
            "general": self.general.status(),
            "monthly_drawdown": self.monthly_drawdown.status(),
            "consecutive_loss": self.consecutive_loss.status(),
        }

    def save_state(self) -> dict:
        """Serialize all breaker states for journal persistence."""
        return {
            "general": self.general.status(),
            "monthly_drawdown": self.monthly_drawdown.save_state(),
            "consecutive_loss": self.consecutive_loss.save_state(),
        }

    @classmethod
    def restore_state(cls, state: dict) -> "BreakerManager":
        """Restore all breakers from persisted state dict."""
        obj = cls()
        if "monthly_drawdown" in state:
            obj.monthly_drawdown = MonthlyDrawdownBreaker.restore_state(state["monthly_drawdown"])
        if "consecutive_loss" in state:
            obj.consecutive_loss = ConsecutiveLossBreaker.restore_state(state["consecutive_loss"])
        return obj

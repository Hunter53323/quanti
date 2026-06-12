"""
Base strategy interface. All concrete strategies inherit from BaseStrategy.
The execution engine depends only on this abstract base class -- never on
concrete strategy implementations.
"""

from abc import ABC, abstractmethod

from quanti.types import MarketData, Order, Portfolio, Signal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str

    @abstractmethod
    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        """Produce trading signals from market data. Pure computation, no side effects."""
        ...

    @abstractmethod
    def size_positions(
        self, signals: list[Signal], capital: float, portfolio: Portfolio
    ) -> list[Order]:
        """Convert signals to sized orders respecting capital constraints and position limits."""
        ...

    @abstractmethod
    def risk_check(
        self, orders: list[Order], portfolio: Portfolio,
        market_data: "MarketData | None" = None,
        risk_checker: object = None,
    ) -> list[Order]:
        """Filter or modify orders based on risk rules. May drop or reduce orders entirely.
        Accepts optional market_data and risk_checker for unified exit path."""
        ...

"""
Shared types: MarketData, Portfolio, Position, Bar, OrderSide, Signal, Order.

These types are defined ONCE and imported by strategy/, state/, execution/, and data/ modules.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Bar:
    """Single price bar for any instrument (ETF, stock, index)."""
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketData:
    """Container passed to strategy.generate_signals()."""
    bars: dict[str, list[Bar]]          # symbol -> latest N bars
    index_bars: dict[str, list[Bar]]    # index data for regime detection
    timestamp: datetime
    index_fundamentals: dict | None = None  # PE/PB percentiles for allocation strategies


@dataclass
class Position:
    """Single position held in the portfolio."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float


@dataclass
class Portfolio:
    """Snapshot of current portfolio state."""
    positions: dict[str, Position]       # symbol -> position
    cash: float                           # total cash (settled + unsettled)
    total_capital: float                  # cash + market value of all positions
    timestamp: datetime
    settled_cash: float = 0.0             # cash available for new buys (T+1 adjusted)
    pending_settlement: float = 0.0       # sell proceeds not yet settled
    settlement_lag_days: int = 1          # days until sell proceeds become available


@dataclass
class Signal:
    """Output from strategy.generate_signals()."""
    symbol: str
    side: OrderSide
    strength: float                      # 0.0 to 1.0, for position sizing priority
    reason: str                          # human-readable, logged for audit trail


@dataclass
class Order:
    """Output from strategy.size_positions(), input to execution layer."""
    symbol: str
    side: OrderSide
    quantity: int                        # in shares/lots
    price: float | None                  # None = market order
    order_type: str                      # "limit" | "market"
    signal_ref: str                      # back-reference to signal for audit trail

"""
Data schemas for incoming market data. All data ingestion modules produce
instances of these dataclasses, and all downstream consumers read through them.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ETFDailyBar:
    """Daily OHLCV bar for an ETF."""
    symbol: str
    trade_date: str               # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float                 # Turnover in RMB


@dataclass(frozen=True)
class ETFMinuteBar:
    """Minute-level bar for intraday use (future)."""
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class IndexDailyBar:
    """Daily OHLCV bar for a market index (CSI 300, CSI 500, etc.)."""
    symbol: str
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class BondDailyBar:
    """Daily data for convertible bonds (Phase 6+)."""
    symbol: str
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    convert_price: float | None    # Conversion price (changes on adjustment)
    convert_value: float | None    # = stock_price * 100 / convert_price
    premium_rt: float | None       # Premium rate (%)
    maturity_date: str | None      # Bond maturity date

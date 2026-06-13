"""
Delayed Confirmation Strategy (simplified, tested equivalently to 5-state original).

Market timing on daily CSI300 bars. Internal state machine emits only:
  0 = not-invested (CASH / confirming / cooldown all emit 0)
  2 = invested full-position (120MA confirmed for N days)
  4 = invested half-position (above 60MA but not 120MA-confirmed)

The cooldown after a false breakout is managed privately by _skip_until_bar.
No VT/MD parameters -- both proven degenerate/redundant (retrospective VT=0.85
never bound because CSI300 volume around breakouts always exceeds 60% of its
20-day average; MD is redundant because any -1% decline crosses below 120MA and
triggers cooldown directly).

When invested: stock momentum selection (top N by trend score).
When not-invested: park cash in 511880 bond/money-market ETF.
Monthly rebalance, stop-loss + DD breaker on rebalance dates.

Decay (default A43): position size decays with months_in_trend_cycle.
  months 1-4: 100%  | months 5-8: 75%  | months 9+: 50%
"""

import numpy as np

from quanti.config import settings
from quanti.indicators import sma as _sma_indicator, adx
from quanti.strategy.base import BaseStrategy
from quanti.types import Bar, MarketData, Order, OrderSide, Portfolio, Position, Signal

DECAY_SCHEDULES = {
    "none": lambda m: 1.0,
    "A43":  lambda m: 1.0 if m <= 4 else (0.75 if m <= 8 else 0.50),
}


class DelayedConfirmStrategy(BaseStrategy):
    """Delayed confirmation market timing with stock momentum + bond rotation + decay."""

    name = "delayed_confirm"

    def __init__(
        self,
        confirm_days: int = 5,
        cooldown_days: int = 40,
        top_n: int = 5,
        stop_loss_pct: float = -10.0,
        min_trend_score: int = 3,
        dd_exit_pct: float = 15.0,
        stock_universe: list | None = None,
        decay_schedule: str = "A43",
    ):
        self.confirm_days = confirm_days
        self.cooldown_days = cooldown_days
        self.top_n = top_n
        self.stop_loss_pct = stop_loss_pct
        self.min_trend_score = min_trend_score
        self.dd_exit_pct = dd_exit_pct
        self.decay_schedule = decay_schedule
        self._decay_fn = DECAY_SCHEDULES.get(decay_schedule, DECAY_SCHEDULES["A43"])

        # Internal state
        self._hwm: dict[str, float] = {}
        self._max_equity: float = 0.0
        self._dd_exit_active: bool = False
        self._dd_exit_days: int = 0

        # Market state
        self._market_state: dict[str, int] = {}  # date_str -> 0/2/4
        self._csi300_bars_loaded: bool = False
        self._skip_until_bar: int = -1  # cooldown bar-index endpoint (private)

        # Monthly rebalance
        self._last_rebalance_month: str = ""

        # Decay timing
        self._months_in_cycle: int = 0
        self._prev_entry_state: int = -1

        # Stock universe filter
        self._stock_universe: set | None = set(stock_universe) if stock_universe else None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        signals: list[Signal] = []
        if not market_data.bars:
            return signals

        if not self._csi300_bars_loaded:
            self._build_market_state(market_data)

        td_str = market_data.timestamp.strftime("%Y%m%d")
        mst = self._market_state.get(td_str, 0)

        # Monthly rebalance gate
        current_month = td_str[:6]
        if current_month == self._last_rebalance_month:
            return signals
        self._last_rebalance_month = current_month

        # Update decay cycle counter
        if mst in (2, 4):
            self._months_in_cycle = 1 if self._prev_entry_state not in (2, 4) else self._months_in_cycle + 1
        else:
            self._months_in_cycle = 0
        self._prev_entry_state = mst

        base_mult = 1.0 if mst == 2 else (0.5 if mst == 4 else 0.0)
        size_mult = base_mult * self._decay_fn(self._months_in_cycle)

        if size_mult > 0.02:
            trending = self._score_stocks(market_data)
            if not trending:
                return signals
            trending.sort(key=lambda x: x[1], reverse=True)
            for sym, score, _ in trending[:self.top_n]:
                signals.append(Signal(
                    symbol=sym, side=OrderSide.BUY,
                    strength=round(min(score / 100.0 * size_mult, 1.0), 4),
                    reason=f"DELAYED_CONFIRM: state={mst} score={score:.1f} decay={self._decay_fn(self._months_in_cycle):.0%} mo={self._months_in_cycle}",
                ))
            signals.append(Signal(symbol="511880", side=OrderSide.SELL, strength=1.0,
                                  reason="DELAYED_CONFIRM: entering market, sell bonds"))
        else:
            signals.append(Signal(symbol="511880", side=OrderSide.BUY, strength=1.0,
                                  reason=f"DELAYED_CONFIRM: state={mst}, park in bonds"))
        return signals

    def size_positions(
        self, signals: list[Signal], capital: float, portfolio: Portfolio,
        market_data: MarketData | None = None,
    ) -> list[Order]:
        orders: list[Order] = []
        buy_signals = [s for s in signals if s.side == OrderSide.BUY]
        sell_signals = [s for s in signals if s.side == OrderSide.SELL]

        if market_data:
            td_str = market_data.timestamp.strftime("%Y%m%d")
            mst = self._market_state.get(td_str, 0)
        else:
            mst = 0

        base_mult = 1.0 if mst == 2 else (0.5 if mst == 4 else 0.0)
        decay_mult = self._decay_fn(self._months_in_cycle) if self._months_in_cycle > 0 else 1.0
        size_mult = base_mult * decay_mult

        stock_buys = [s for s in buy_signals if s.symbol != "511880"]
        bond_buys = [s for s in buy_signals if s.symbol == "511880"]
        bond_sells = [s for s in sell_signals if s.symbol == "511880"]
        selected_syms = {s.symbol for s in stock_buys}

        for sym, pos in portfolio.positions.items():
            if sym == "511880" or pos.quantity <= 0:
                continue
            sell = False
            reason = ""
            if sym not in selected_syms:
                sell = True
                reason = f"DELAYED_CONFIRM: rotated out (state={mst})"
            elif sym in self._hwm and pos.current_price > 0:
                loss = (pos.current_price / self._hwm[sym] - 1) * 100
                if loss < self.stop_loss_pct:
                    sell = True
                    reason = f"Stop-loss: {loss:.1f}% from HWM"
            if sell:
                orders.append(Order(symbol=sym, side=OrderSide.SELL, quantity=pos.quantity,
                                    price=None, order_type="market", signal_ref=reason))

        new_entries = [s for s in stock_buys if s.symbol not in portfolio.positions]
        if new_entries and size_mult > 0.02:
            total_cap = capital + sum(p.quantity * p.current_price for p in portfolio.positions.values())
            n_positions = max(len(selected_syms), 1)
            per_stock = total_cap * size_mult / n_positions * 0.92
            for sig in new_entries:
                price = self._get_price(sig.symbol, portfolio, market_data)
                if price and price > 0.01:
                    qty = int(per_stock / price / 100) * 100
                    if qty >= 100:
                        orders.append(Order(symbol=sig.symbol, side=OrderSide.BUY,
                                            quantity=qty, price=price, order_type="limit",
                                            signal_ref=sig.reason))

        bond_pos = portfolio.positions.get("511880")
        bond_units = bond_pos.quantity if bond_pos else 0

        for sig in bond_sells:
            if bond_units > 0:
                orders.append(Order(symbol="511880", side=OrderSide.SELL, quantity=bond_units,
                                    price=None, order_type="market", signal_ref=sig.reason))

        for sig in bond_buys:
            bond_price = self._get_price("511880", portfolio, market_data)
            if bond_price and bond_price > 0.01 and capital > 1000:
                qty = int(capital * 0.99 / bond_price)
                if qty > 0:
                    orders.append(Order(symbol="511880", side=OrderSide.BUY, quantity=qty,
                                        price=bond_price, order_type="limit", signal_ref=sig.reason))
        return orders

    def risk_check(
        self, orders: list[Order], portfolio: Portfolio,
        market_data: MarketData | None = None, risk_checker: object = None,
    ) -> list[Order]:
        approved: list[Order] = []
        total_value = portfolio.cash + sum(p.quantity * p.current_price for p in portfolio.positions.values())
        if total_value > self._max_equity:
            self._max_equity = total_value
            self._dd_exit_active = False

        if self._max_equity > 0 and self.dd_exit_pct > 0:
            dd = (self._max_equity - total_value) / self._max_equity * 100
            if dd > self.dd_exit_pct:
                self._dd_exit_active = True
                self._dd_exit_days = 0
                self._months_in_cycle = 0
                self._prev_entry_state = -1
            elif self._dd_exit_active:
                self._dd_exit_days += 1
                if (total_value / self._max_equity > 0.92) or (self._dd_exit_days > 60 and dd <= self.dd_exit_pct):
                    self._dd_exit_active = False
                    self._max_equity = total_value
                    self._dd_exit_days = 0

        if self._dd_exit_active:
            for sym, pos in portfolio.positions.items():
                if pos.quantity > 0:
                    approved.append(Order(symbol=sym, side=OrderSide.SELL, quantity=pos.quantity,
                                          price=None, order_type="market",
                                          signal_ref=f"DD breaker: -{dd:.1f}% from peak"))
            return approved

        for sym, pos in portfolio.positions.items():
            if sym == "511880":
                continue
            if sym not in self._hwm or pos.current_price > self._hwm[sym]:
                self._hwm[sym] = pos.current_price
            if pos.current_price > 0 and self._hwm.get(sym, 0) > 0:
                loss_pct = (pos.current_price / self._hwm[sym] - 1) * 100
                if loss_pct < self.stop_loss_pct:
                    approved.append(Order(symbol=sym, side=OrderSide.SELL, quantity=pos.quantity,
                                          price=None, order_type="market",
                                          signal_ref=f"HWM stop: {loss_pct:.1f}% from peak"))

        for o in orders:
            cost = o.quantity * (o.price or 0)
            trading_cap = getattr(settings, "TRADING_CAPITAL", 90000)
            if cost > trading_cap * 0.25:
                continue
            approved.append(o)
        return approved

    def get_decay_info(self) -> dict:
        return {
            "schedule": self.decay_schedule,
            "months_in_cycle": self._months_in_cycle,
            "current_multiplier": self._decay_fn(self._months_in_cycle),
            "prev_entry_state": self._prev_entry_state,
        }

    # -----------------------------------------------------------------
    # Simplified market state machine: emits only 0/2/4
    # -----------------------------------------------------------------

    def _build_market_state(self, market_data: MarketData) -> None:
        csi300_bars = None
        for sym in ("510300", "159919"):
            if market_data.index_bars and sym in market_data.index_bars:
                csi300_bars = market_data.index_bars[sym]
                break
            if sym in market_data.bars:
                csi300_bars = market_data.bars[sym]
                break
        if not csi300_bars or len(csi300_bars) < 200:
            self._csi300_bars_loaded = True
            return

        closes = np.array([b.close for b in csi300_bars], dtype=np.float64)
        dates  = [b.datetime.strftime("%Y%m%d") for b in csi300_bars]
        n      = len(closes)

        ma120 = self._sma(closes, 120)
        ma60  = self._sma(closes, 60)
        if ma120 is None or ma60 is None:
            self._csi300_bars_loaded = True
            return

        a120 = (closes > ma120) & (~np.isnan(ma120))
        a60  = (closes > ma60)  & (~np.isnan(ma60))

        N = self.confirm_days
        M = self.cooldown_days

        cd_end   = -1  # cooldown end bar-index
        cf_start = -1  # confirming start bar-index
        fd_start = -1  # full (invested) start bar-index

        for i in range(121, n):
            if i <= cd_end:
                self._market_state[dates[i]] = 0
                continue

            if fd_start >= 0:
                if a120[i]:
                    self._market_state[dates[i]] = 2
                    continue
                else:
                    fd_start = -1

            if cf_start >= 0:
                if not a120[i]:
                    cd_end = i + M - 1
                    self._market_state[dates[i]] = 0
                    cf_start = -1
                    continue
                if i - cf_start + 1 == N:
                    self._market_state[dates[i]] = 2
                    fd_start = i
                    cf_start = -1
                else:
                    self._market_state[dates[i]] = 0
                continue

            if a120[i] and not a120[i - 1]:
                cf_start = i
                self._market_state[dates[i]] = 0
            elif a60[i]:
                self._market_state[dates[i]] = 4
            else:
                self._market_state[dates[i]] = 0

        self._csi300_bars_loaded = True

    # -----------------------------------------------------------------
    # Stock scoring (unchanged)
    # -----------------------------------------------------------------

    @staticmethod
    def _sma(arr: np.ndarray, period: int) -> np.ndarray | None:
        if len(arr) < period:
            return None
        return _sma_indicator(arr, period)

    @staticmethod
    def _adx(high, low, close, period=14):
        return adx(high, low, close, period)

    def _score_stocks(self, market_data: MarketData) -> list[tuple[str, float, int]]:
        trending: list[tuple[str, float, int]] = []
        for symbol, bars in market_data.bars.items():
            if symbol == "511880":
                continue
            if self._stock_universe and symbol not in self._stock_universe:
                continue
            if len(bars) < 200:
                continue
            is_t, cond_count = self._is_stock_trending(bars)
            if not is_t or cond_count < self.min_trend_score:
                continue
            score = self._trend_strength_score(bars)
            if score > 0:
                trending.append((symbol, score, cond_count))
        return trending

    def _is_stock_trending(self, bars: list[Bar]) -> tuple[bool, int]:
        if len(bars) < 200:
            return False, 0
        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs  = np.array([b.high for b in bars], dtype=np.float64)
        lows   = np.array([b.low for b in bars], dtype=np.float64)
        vols   = np.array([b.volume for b in bars], dtype=np.float64)

        count = 0
        ma120 = self._sma(closes, 120)
        if ma120 is not None and not np.isnan(ma120[-1]) and closes[-1] > ma120[-1]:
            count += 1

        rh = np.max(highs[-20:]); ph = np.max(highs[-60:-20])
        rl = np.min(lows[-20:]);  pl = np.min(lows[-60:-20])
        if rh > ph and rl > pl:
            count += 1

        ma20_a = self._sma(closes, 20); ma60_a = self._sma(closes, 60)
        if (ma20_a is not None and ma60_a is not None and ma120 is not None and
            not np.isnan(ma20_a[-1]) and not np.isnan(ma60_a[-1]) and
            not np.isnan(ma120[-1]) and ma20_a[-1] > ma60_a[-1] > ma120[-1]):
            count += 1

        adx_arr = self._adx(highs, lows, closes, 14)
        if adx_arr is not None and not np.isnan(adx_arr[-1]) and adx_arr[-1] > 25:
            count += 1

        vol_20 = np.mean(vols[-21:-1])
        if vol_20 > 0 and vols[-1] > vol_20 * 1.2:
            count += 1

        return (count >= self.min_trend_score and count >= 1), count

    def _trend_strength_score(self, bars: list[Bar]) -> float:
        closes = np.array([b.close for b in bars], dtype=np.float64)
        if len(closes) < 130:
            return 0.0
        if closes[-63] < 1e-6 or closes[-126] < 1e-6:
            mom_score = 0.0
        else:
            ret_3m = closes[-1] / closes[-63] - 1
            ret_6m = closes[-1] / closes[-126] - 1
            m3 = min(max(ret_3m / 0.5, 0), 1) if ret_3m > 0 else 0
            m6 = min(max(ret_6m / 0.8, 0), 1) if ret_6m > 0 else 0
            mom_score = (0.5 * m3 + 0.5 * m6) * 100
        if len(closes) >= 61:
            w = closes[-61:]
            dr = np.diff(w) / (w[:-1] + 1e-10)
            vol_score = max(0, (1 - min(np.nanstd(dr) / 0.04, 1))) * 100
        else:
            vol_score = 50.0
        return 0.6 * mom_score + 0.4 * vol_score

    @staticmethod
    def _get_price(symbol: str, portfolio: Portfolio, market_data: MarketData | None) -> float | None:
        pos = portfolio.positions.get(symbol)
        if pos and pos.current_price > 0:
            return pos.current_price
        if market_data and symbol in market_data.bars:
            bars = market_data.bars[symbol]
            if bars:
                return bars[-1].close
        return None

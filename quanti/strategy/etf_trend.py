"""
ETF Trend Following Strategy.
Implements multi-indicator AND-resonance entry, market filter, and sector rotation.

Entry logic (ALL 5 conditions must be True):
1. MA Alignment: SMA20 > SMA60 > SMA120 on both today AND yesterday
2. ADX Trend: ADX > entry_threshold AND +DI > -DI
3. BB Expansion: Bollinger Band width expanding AND price breaks above upper band
4. Volume Surge: Current volume > 20-day average * volume_surge_mult
5. Market Filter: At least one major index has ADX > 20

Sell logic (baseline exit):
- Fast MA crosses below slow MA (trend reversal)

Enhanced exits (ATR trailing stop, time stop, RSI overbought, volatility stop) are
integrated into the risk_check() method for unified exit logic.
"""
from datetime import datetime

import numpy as np

from quanti.config import settings
from quanti.execution.risk import RiskChecker
from quanti.indicators import (
    adx,
    adx_with_di,
    bollinger_bands,
    compute_atr,
    compute_rsi,
    ema,
    sma,
    wilder_smooth,
)
from quanti.strategy.base import BaseStrategy
from quanti.strategy.signal_filters import MarketEnvironmentFilter
from quanti.types import MarketData, Order, OrderSide, Portfolio, Position, Signal


class ETFTrendStrategy(BaseStrategy):
    """
    Multi-ETF trend following strategy with multi-indicator AND-resonance entry.

    Entry requires all 5 conditions to fire simultaneously:
    1. MA Alignment (short > medium > long, confirmed across 2 days)
    2. ADX Trend confirmation (ADX > threshold, +DI > -DI)
    3. Bollinger Band expansion with breakout
    4. Volume surge confirmation
    5. Market filter (broad market trending)

    Parameters:
    - ma_fast: Fast SMA period (default: 20)
    - ma_medium: Medium SMA period (default: 60)
    - ma_long: Long SMA period (default: 120)
    - bb_period: Bollinger Band period (default: 20)
    - bb_std: Bollinger Band standard deviation multiplier (default: 2.0)
    - volume_surge_mult: Volume surge threshold multiplier (default: 1.5)
    - adx_entry_threshold: ADX minimum for entry (default: 25)
    - adx_threshold: ADX minimum for regime check (default: 20)
    - di_diff_threshold: DI difference threshold (default: 15)
    - entry_mode: 'resonance' (5-condition) | 'legacy' (MA cross) (default: 'resonance')
    """

    name = "etf_trend"

    def __init__(
        self,
        ma_fast: int | None = None,
        ma_slow: int | None = None,
        ma_long: int | None = None,
        bb_period: int | None = None,
        bb_std: float | None = None,
        volume_surge_mult: float | None = None,
        adx_threshold: int | None = None,
        adx_entry_threshold: int | None = None,
        di_diff_threshold: int | None = None,
        entry_mode: str = "resonance",
    ):
        # MA params
        self.ma_fast = ma_fast or settings.MA_FAST
        self.ma_slow = ma_slow or settings.MA_SLOW
        self.ma_long = ma_long or getattr(settings, 'MA_LONG', 120)

        # Bollinger Band params
        self.bb_period = bb_period or getattr(settings, 'BB_PERIOD', 20)
        self.bb_std = bb_std or getattr(settings, 'BB_STD', 2.0)

        # Volume params
        self.volume_surge_mult = volume_surge_mult or getattr(settings, 'VOLUME_SURGE_MULTIPLIER', 1.5)

        # ADX params
        self.adx_threshold = adx_threshold or settings.ADX_THRESHOLD
        self.adx_entry_threshold = adx_entry_threshold or getattr(settings, 'ADX_ENTRY_THRESHOLD', 25)
        self.di_diff_threshold = di_diff_threshold or getattr(settings, 'DI_DIFF_THRESHOLD', 15)

        # Entry mode: 'resonance' (5-condition AND) or 'legacy' (simple MA cross)
        self.entry_mode = entry_mode

        # Exit tracking state (per-position)
        self._hwm_tracker: dict[str, float] = {}        # high water mark per symbol
        self._entry_time_tracker: dict[str, datetime] = {}  # entry datetime per symbol
        self._entry_atr_tracker: dict[str, float] = {}   # entry ATR per symbol

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        """
        Generate trading signals using weighted composite scoring.

        For each symbol, compute 5 indicator scores (0-100 each):
        1. MA Alignment (0-100): SMA20 > 60 > 120 with separation bonus
        2. ADX Trend (0-100): ADX strength + DI directional bias
        3. BB Expansion (0-100): bandwidth expansion + price position vs bands
        4. Volume Surge (0-100): volume ratio vs 20d average
        5. Market Filter (0-100): broad market trend strength

        Weighted sum >= entry_score_threshold → BUY
        SELL follows baseline MA crossover reversal logic.
        """
        signals: list[Signal] = []

        # Market filter score (computed once per call)
        market_score = self._check_market_filter_with_score(market_data)

        for symbol, bars in market_data.bars.items():
            if len(bars) < max(self.ma_long, self.bb_period + 5, 30):
                continue

            closes = np.array([b.close for b in bars], dtype=np.float64)
            highs = np.array([b.high for b in bars], dtype=np.float64)
            lows = np.array([b.low for b in bars], dtype=np.float64)
            volumes = np.array([b.volume for b in bars], dtype=np.float64)

            # MAs
            sma20 = self._sma(closes, self.ma_fast)
            sma60 = self._sma(closes, self.ma_slow)
            sma120 = self._sma(closes, self.ma_long)

            if sma20[-1] is None or np.isnan(sma20[-1]) or sma60[-1] is None or np.isnan(sma60[-1]):
                continue

            # ── Legacy Mode ──
            if self.entry_mode == "legacy":
                adx_result = self._adx_with_di(highs, lows, closes, period=14)
                adx_val = adx_result[0][-1] if adx_result is not None else 0
                has_trend = not np.isnan(adx_val) and adx_val > self.adx_threshold
                trend_up = sma20[-1] > sma60[-1]
                trend_down = sma20[-1] < sma60[-1]
                if trend_up and has_trend:
                    signals.append(Signal(symbol=symbol, side=OrderSide.BUY, strength=0.85,
                        reason=f"LEGACY: fast_ma({self.ma_fast})={sma20[-1]:.4f} > slow_ma({self.ma_slow})={sma60[-1]:.4f}, ADX={adx_val:.1f}"))
                elif trend_down:
                    signals.append(Signal(symbol=symbol, side=OrderSide.SELL, strength=1.0,
                        reason=f"LEGACY: fast_ma({self.ma_fast})={sma20[-1]:.4f} < slow_ma({self.ma_slow})={sma60[-1]:.4f}"))
                continue

            # ── Scoring Mode: weighted composite ──
            entry_threshold = getattr(settings, 'ENTRY_SCORE_THRESHOLD', 55)

            # ADX + DI
            adx_result = self._adx_with_di(highs, lows, closes, period=14)
            if adx_result is None:
                continue
            adx, plus_di, minus_di = adx_result

            # Bollinger Bands (can fail gracefully)
            bb_middle, bb_upper, bb_lower = self._bollinger_bands(closes, self.bb_period, self.bb_std)

            # ── Compute individual scores (0-100 each) ──
            scores: dict[str, float] = {}

            # 1. MA Alignment (25% weight)
            _, ma_subscore = self._check_ma_alignment_with_score(sma20, sma60, sma120)
            scores["ma"] = ma_subscore * 100  # already 0-1

            # 2. ADX Trend (25% weight)
            _, adx_subscore = self._check_adx_trend_with_score(adx, plus_di, minus_di)
            scores["adx"] = adx_subscore * 100

            # 3. BB Expansion (20% weight): even if BB computation failed, give partial score
            if bb_middle is not None:
                _, bb_subscore = self._check_bb_expansion_with_score(closes, bb_middle, bb_upper, bb_lower)
                scores["bb"] = bb_subscore * 100
            else:
                scores["bb"] = 30.0  # neutral when BB unavailable

            # 4. Volume Surge (20% weight)
            _, vol_subscore = self._check_volume_surge_with_score(bars, volumes)
            scores["vol"] = vol_subscore * 100

            # 5. Market Filter (10% weight)
            scores["mkt"] = market_score * 100

            # ── Weighted composite ──
            w_ma  = getattr(settings, 'ENTRY_WEIGHT_MA', 0.25)
            w_adx = getattr(settings, 'ENTRY_WEIGHT_ADX', 0.25)
            w_bb  = getattr(settings, 'ENTRY_WEIGHT_BB', 0.20)
            w_vol = getattr(settings, 'ENTRY_WEIGHT_VOL', 0.20)
            w_mkt = getattr(settings, 'ENTRY_WEIGHT_MKT', 0.10)

            composite = (
                w_ma  * scores["ma"]
                + w_adx * scores["adx"]
                + w_bb  * scores["bb"]
                + w_vol * scores["vol"]
                + w_mkt * scores["mkt"]
            )

            # ── Signal decision ──
            if composite >= entry_threshold:
                # Normalize composite to 0-1 for signal strength
                strength = round(composite / 100.0, 4)

                parts = []
                parts.append(f"score={composite:.1f}/100")
                parts.append(f"MA={scores['ma']:.0f}")
                parts.append(f"ADX={scores['adx']:.0f}")
                parts.append(f"BB={scores['bb']:.0f}")
                parts.append(f"VOL={scores['vol']:.0f}")
                parts.append(f"MKT={scores['mkt']:.0f}")

                signals.append(Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=strength,
                    reason=" | ".join(parts),
                ))

            # SELL: MA crossover reversal (baseline exit)
            elif sma20[-1] < sma60[-1]:
                signals.append(Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=1.0,
                    reason=f"MA cross: fast({self.ma_fast})={sma20[-1]:.4f} < slow({self.ma_slow})={sma60[-1]:.4f}",
                ))

        return signals

    # ------------------------------------------------------------------
    # Check methods (return bool; with_score variants return (bool, float))
    # ------------------------------------------------------------------

    def _check_ma_alignment(self, closes: np.ndarray) -> bool:
        """
        Check if SMA20 > SMA60 > SMA120 on both today and yesterday.
        Prevents single-day false breakouts.
        """
        sma20 = self._sma(closes, self.ma_fast)
        sma60 = self._sma(closes, self.ma_slow)
        sma120 = self._sma(closes, self.ma_long)
        return self._check_ma_alignment_with_score(sma20, sma60, sma120)[0]

    @staticmethod
    def _check_ma_alignment_with_score(
        sma20: np.ndarray, sma60: np.ndarray, sma120: np.ndarray
    ) -> tuple[bool, float]:
        """Check MA alignment and compute score. Returns (passed, score)."""
        # Need at least 2 valid bars
        if len(sma20) < 2 or len(sma60) < 2 or len(sma120) < 2:
            return (False, 0.0)

        today = -1
        yesterday = -2

        vals = [
            sma20[today], sma60[today], sma120[today],
            sma20[yesterday], sma60[yesterday], sma120[yesterday],
        ]
        if any(v is None or np.isnan(v) for v in vals):
            return (False, 0.0)

        # Today: SMA20 > SMA60 > SMA120
        today_ok = sma20[today] > sma60[today] > sma120[today]
        # Yesterday: SMA20 > SMA60 > SMA120
        yesterday_ok = sma20[yesterday] > sma60[yesterday] > sma120[yesterday]

        passed = today_ok and yesterday_ok

        if passed:
            # ma_score = min(abs(SMA20-SMA60)/SMA60 * 100/3.0, 1.0)
            separation_pct = abs(sma20[today] - sma60[today]) / sma60[today] * 100
            ma_score = min(separation_pct / 3.0, 1.0)
        else:
            ma_score = 0.0

        return (passed, ma_score)

    def _check_adx_trend(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> bool:
        """
        Check if ADX > entry_threshold AND +DI > -DI.
        Ensures strong directional trend.
        """
        result = self._adx_with_di(highs, lows, closes, period=14)
        if result is None:
            return False
        adx, plus_di, minus_di = result
        return self._check_adx_trend_with_score(adx, plus_di, minus_di)[0]

    def _check_adx_trend_with_score(
        self, adx: np.ndarray, plus_di: np.ndarray, minus_di: np.ndarray
    ) -> tuple[bool, float]:
        """Check ADX trend and compute score. Returns (passed, score)."""
        if len(adx) == 0:
            return (False, 0.0)

        adx_now = adx[-1]
        plus_now = plus_di[-1]
        minus_now = minus_di[-1]

        if any(np.isnan(v) for v in [adx_now, plus_now, minus_now]):
            return (False, 0.0)

        passed = (
            adx_now > self.adx_entry_threshold
            and plus_now > minus_now
            and (plus_now - minus_now) > self.di_diff_threshold
        )

        adx_score = min(adx_now / 40.0, 1.0) if passed else 0.0

        return (passed, adx_score)

    def _check_bb_expansion(self, closes: np.ndarray) -> bool:
        """
        Check Bollinger Band expansion and breakout.
        Current bandwidth > avg of last 5 bandwidths * 1.2 AND close > upper band.
        """
        bb_middle, bb_upper, bb_lower = self._bollinger_bands(closes, self.bb_period, self.bb_std)
        if bb_middle is None:
            return False
        return self._check_bb_expansion_with_score(closes, bb_middle, bb_upper, bb_lower)[0]

    @staticmethod
    def _check_bb_expansion_with_score(
        closes: np.ndarray,
        bb_middle: np.ndarray,
        bb_upper: np.ndarray,
        bb_lower: np.ndarray,
    ) -> tuple[bool, float]:
        """Check BB expansion and compute score. Returns (passed, score)."""
        n = len(bb_middle)
        if n < 6:  # Need at least current + 5 past bandwidth values
            return (False, 0.0)

        # Compute bandwidth array
        bandwidths = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if (not np.isnan(bb_upper[i]) and not np.isnan(bb_lower[i])
                    and not np.isnan(bb_middle[i]) and bb_middle[i] > 0):
                bandwidths[i] = (bb_upper[i] - bb_lower[i]) / bb_middle[i]

        current_bw = bandwidths[-1]
        if np.isnan(current_bw):
            return (False, 0.0)

        # Average of last 5 bandwidths (indices -6 to -2, excluding current)
        past_bws = bandwidths[-6:-1]
        past_bws = past_bws[~np.isnan(past_bws)]
        if len(past_bws) == 0:
            return (False, 0.0)

        avg_past_bw = np.mean(past_bws)
        if avg_past_bw <= 0:
            return (False, 0.0)

        # Expansion: current bandwidth > avg past * 1.2
        expansion_ratio = current_bw / avg_past_bw
        expanding = expansion_ratio > 1.2

        # Breakout: close above upper band
        close_now = closes[-1]
        upper_now = bb_upper[-1]
        breakout = (not np.isnan(close_now) and not np.isnan(upper_now)
                    and close_now > upper_now)

        passed = expanding and breakout

        bb_score = min((expansion_ratio - 1.0) / 0.5, 1.0) if passed else 0.0

        return (passed, bb_score)

    def _check_volume_surge(self, bars: list, volumes: np.ndarray | None = None) -> bool:
        """
        Check if current volume exceeds 20-day average * volume_surge_mult.
        """
        if volumes is None:
            volumes = np.array([b.volume for b in bars], dtype=np.float64)
        return self._check_volume_surge_with_score(bars, volumes)[0]

    def _check_volume_surge_with_score(
        self, bars: list, volumes: np.ndarray
    ) -> tuple[bool, float]:
        """Check volume surge and compute score. Returns (passed, score)."""
        n = len(volumes)
        if n < 22:  # Need 20 past + 1 current
            return (False, 0.0)

        current_vol = volumes[-1]
        if current_vol <= 0:
            return (False, 0.0)

        # 20-day average volume (excluding current day)
        avg_vol_20 = np.mean(volumes[-21:-1])
        if avg_vol_20 <= 0:
            return (False, 0.0)

        volume_ratio = current_vol / avg_vol_20
        passed = volume_ratio > self.volume_surge_mult

        vol_score = min(volume_ratio / 3.0, 1.0) if passed else 0.0

        return (passed, vol_score)

    def _check_market_filter(self, market_data: MarketData) -> bool:
        """
        Check if broad market is trending (at least one index has ADX > 20).
        Returns True if no index data available (permissive).
        Returns False only if ALL indices have ADX <= 20.
        """
        return self._check_market_filter_with_score(market_data) >= 0.5

    def _check_market_filter_with_score(self, market_data: MarketData) -> float:
        """
        Broad market trend score (0.0-1.0).

        - 1.0: index ADX > 40 (strong trend)
        - 0.7: index ADX 30-40
        - 0.5: index ADX 20-30
        - 0.3: index ADX < 20 (weak/no trend)
        - 1.0: no index data (permissive — don't block on missing data)
        """
        if not market_data.index_bars:
            return 1.0  # Permissive

        best_adx = 0.0
        for _index_name, bars in market_data.index_bars.items():
            if len(bars) < 30:
                continue
            closes = np.array([b.close for b in bars], dtype=np.float64)
            highs = np.array([b.high for b in bars], dtype=np.float64)
            lows = np.array([b.low for b in bars], dtype=np.float64)
            adx = self._adx(highs, lows, closes, period=14)
            if adx is not None and len(adx) > 0 and not np.isnan(adx[-1]):
                best_adx = max(best_adx, adx[-1])

        if best_adx <= 0:
            return 1.0  # No indicator data = permissive
        if best_adx >= 40:
            return 1.0
        if best_adx >= 30:
            return 0.7
        if best_adx >= 20:
            return 0.5
        return 0.3

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def size_positions(
        self, signals: list[Signal], capital: float, portfolio: Portfolio,
        market_data: "MarketData | None" = None,
    ) -> list[Order]:
        """
        Equal-weight position sizing among active buy signals.
        Sells liquidate the entire position.

        Capital allocation:
        - Each position gets TRADING_CAPITAL / min(num_signals, MAX_POSITIONS)
        - Minimum 1 position, maximum MAX_POSITIONS
        """
        buy_signals = [s for s in signals if s.side == OrderSide.BUY]
        sell_signals = [s for s in signals if s.side == OrderSide.SELL]

        orders: list[Order] = []

        # Sell orders: liquidate existing positions that have sell signals
        for sig in sell_signals:
            pos = portfolio.positions.get(sig.symbol)
            if pos and pos.quantity > 0:
                orders.append(Order(
                    symbol=sig.symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    price=None,  # market order for exit
                    order_type="market",
                    signal_ref=sig.reason,
                ))

        # Buy orders: equal-weight among ranked buy signals
        if buy_signals and capital > 0:
            # Sort by strength, take top N
            ranked = sorted(buy_signals, key=lambda s: s.strength, reverse=True)
            active = ranked[:settings.MAX_POSITIONS]

            per_position_capital = capital / len(active)
            existing_symbols = set(portfolio.positions.keys())

            for sig in active:
                # Skip if we already hold this
                if sig.symbol in existing_symbols:
                    continue

                # Get price: portfolio first, then market data
                price = self._get_last_price(sig.symbol, portfolio)
                if price is None and market_data is not None:
                    bars = market_data.bars.get(sig.symbol, [])
                    if bars:
                        price = bars[-1].close

                if price and price > 0:
                    qty = int(per_position_capital / price / 100) * 100  # round to lots
                    if qty > 0:
                        orders.append(Order(
                            symbol=sig.symbol,
                            side=OrderSide.BUY,
                            quantity=qty,
                            price=price,
                            order_type="limit",
                            signal_ref=sig.reason,
                        ))

        return orders

    # ------------------------------------------------------------------
    # Risk check
    # ------------------------------------------------------------------

    def risk_check(
        self,
        orders: list[Order],
        portfolio: Portfolio,
        market_data: "MarketData | None" = None,
        risk_checker: "RiskChecker | None" = None,
    ) -> list[Order]:
        """
        Unified risk & exit check for both backtest and live paths.

        Exit conditions (delegated to individual testable methods):
        1. _flat_stop_loss: exit when loss from avg_cost > STOP_LOSS_PCT
        2. _atr_trailing_stop: exit when price drops below HWM - atr_mult * ATR
        3. _time_stop: reduce 50% at time_stop_reduce days, full exit at time_stop_exit
        4. _rsi_exit: tighten ATR multiplier from 2x to 1.5x when RSI > 80
        5. _volatility_stop: exit when ATR_current > expansion_mult * ATR_entry

        Risk filters (delegated to RiskChecker.check_all when risk_checker provided):
        1. Capital sufficiency
        2. Position limits
        3. Duplicate detection
        4. Stop-loss / ATR checks (unified with live/paper path)

        In-strategy filters (ordering / sizing):
        1. Single position cost <= TRADING_CAPITAL
        2. Total exposure <= TRADING_CAPITAL
        3. No duplicate symbol+side orders
        """
        stop_loss_pct = float(getattr(settings, 'STOP_LOSS_PCT', 0.0) or 0.0)
        atr_trailing_enabled = getattr(settings, 'ATR_TRAILING_STOP_ENABLED', False)
        time_stop_enabled = getattr(settings, 'TIME_STOP_ENABLED', False)
        vol_stop_enabled = getattr(settings, 'VOLATILITY_STOP_ENABLED', False)
        rsi_exit_enabled = getattr(settings, 'RSI_EXIT_ENABLED', False)

        # Mutable tracking: start from incoming orders, add exit SELLs
        exit_orders: list[Order] = []

        # Compute policy intervention score (shared across all exit methods)
        nt_detection_enabled = getattr(settings, 'NT_INTERVENTION_DETECTION_ENABLED', True)
        policy_score = 0.0
        if nt_detection_enabled and market_data is not None:
            try:
                nt_filter = MarketEnvironmentFilter()
                policy_score = nt_filter.get_policy_intervention_score(market_data.bars)
            except Exception:
                pass  # Fail silently: no intervention data = score 0.0

        # ------------------------------------------------------------------
        # Exit checks: call individual testable methods
        # ------------------------------------------------------------------

        for symbol, pos in portfolio.positions.items():
            if pos.quantity <= 0:
                continue

            bars = market_data.bars.get(symbol) if market_data else None
            if bars is None or len(bars) < 16:
                continue

            # 0. Gap risk: detect T+1 lock-in danger (precedes all exit checks)
            gap_risk_enabled = getattr(settings, 'GAP_RISK_CHECK_ENABLED', True)
            if gap_risk_enabled and self._check_gap_risk(symbol, pos, bars):
                # Reduce position 50% preemptively when gap risk is elevated
                reduce_qty = max(pos.quantity // 2, 100)
                exit_orders.append(Order(
                    symbol=symbol, side=OrderSide.SELL,
                    quantity=reduce_qty, price=None, order_type="market",
                    signal_ref='gap risk: intraday range wide, T+1 lock-in danger',
                ))

            # 1. Flat stop-loss
            exit_order = self._flat_stop_loss(symbol, portfolio.positions, stop_loss_pct)
            if exit_order is not None:
                exit_orders.append(exit_order)
                continue  # skip other exits if already liquidating

            # 5. Volatility stop
            if vol_stop_enabled:
                exit_order = self._volatility_stop(symbol, bars)
                if exit_order is not None:
                    exit_order = Order(
                        symbol=exit_order.symbol, side=exit_order.side,
                        quantity=pos.quantity, price=exit_order.price,
                        order_type=exit_order.order_type,
                        signal_ref=exit_order.signal_ref,
                    )
                    exit_orders.append(exit_order)
                    continue

            # 2. ATR trailing stop (includes RSI tightening internally)
            if atr_trailing_enabled:
                exit_order = self._atr_trailing_stop(symbol, bars, policy_intervention_score=policy_score)
                if exit_order is not None:
                    exit_order = Order(
                        symbol=exit_order.symbol, side=exit_order.side,
                        quantity=pos.quantity, price=exit_order.price,
                        order_type=exit_order.order_type,
                        signal_ref=exit_order.signal_ref,
                    )
                    exit_orders.append(exit_order)
                    continue

            # 4. RSI exit (tightened ATR stop) -- only if not already covered by ATR check
            # The _atr_trailing_stop already applies RSI tightening internally.
            # _rsi_exit is a standalone check for when ATR stop is disabled
            # but RSI-based tightening is desired separately.
            if rsi_exit_enabled and not atr_trailing_enabled:
                exit_order = self._rsi_exit(symbol, bars)
                if exit_order is not None:
                    exit_order = Order(
                        symbol=exit_order.symbol, side=exit_order.side,
                        quantity=pos.quantity, price=exit_order.price,
                        order_type=exit_order.order_type,
                        signal_ref=exit_order.signal_ref,
                    )
                    exit_orders.append(exit_order)
                    continue

            # 3. Time stop
            if time_stop_enabled:
                exit_order = self._time_stop(symbol, bars)
                if exit_order is not None:
                    if exit_order.quantity == 0:
                        # Time stop returns qty=0 to indicate caller fills in actual
                        # Check if it's a full exit or partial reduction
                        entry_dt = self._entry_time_tracker.get(symbol)
                        if entry_dt and bars:
                            days_held = (bars[-1].datetime - entry_dt).days
                            time_stop_exit_days = getattr(settings, 'TIME_STOP_DAYS_EXIT', 60)
                            getattr(settings, 'TIME_STOP_DAYS_REDUCE', 40)
                            if days_held >= time_stop_exit_days:
                                exit_order = Order(
                                    symbol=exit_order.symbol, side=exit_order.side,
                                    quantity=pos.quantity, price=exit_order.price,
                                    order_type=exit_order.order_type,
                                    signal_ref=exit_order.signal_ref,
                                )
                            else:
                                reduce_qty = max(pos.quantity // 2, 100)
                                exit_order = Order(
                                    symbol=exit_order.symbol, side=exit_order.side,
                                    quantity=reduce_qty, price=exit_order.price,
                                    order_type=exit_order.order_type,
                                    signal_ref=exit_order.signal_ref,
                                )
                    exit_orders.append(exit_order)

        # ------------------------------------------------------------------
        # Delegate to RiskChecker for unified pre-trade checks
        # ------------------------------------------------------------------
        # Only skip RiskChecker exits when market_data was provided and the
        # strategy already generated its own exit orders. If market_data is
        # None (test mode or degenerate case), let RiskChecker handle exits.
        skip_exits = market_data is not None
        all_orders = list(orders) + exit_orders
        if risk_checker is not None:
            # Use the shared risk_checker (engine/main passes theirs)
            risk_checker._high_water_marks.update(self._hwm_tracker)
            risk_checker._entry_atr.update(self._entry_atr_tracker)
            all_orders, _ = risk_checker.check_all(all_orders, portfolio, market_data,
                                                      skip_exits=skip_exits)
            for sym, hwm in risk_checker._high_water_marks.items():
                self._hwm_tracker[sym] = hwm
            for sym, atr in risk_checker._entry_atr.items():
                self._entry_atr_tracker[sym] = atr
        else:
            # Fallback: create temporary checker (for standalone tests)
            temp_checker = RiskChecker()
            temp_checker._high_water_marks.update(self._hwm_tracker)
            temp_checker._entry_atr.update(self._entry_atr_tracker)
            all_orders, _ = temp_checker.check_all(all_orders, portfolio, market_data,
                                                     skip_exits=skip_exits)
            for sym, hwm in temp_checker._high_water_marks.items():
                self._hwm_tracker[sym] = hwm
            for sym, atr in temp_checker._entry_atr.items():
                self._entry_atr_tracker[sym] = atr

        # ------------------------------------------------------------------
        # In-strategy filters: exposure limits, duplicate detection
        # ------------------------------------------------------------------
        final: list[Order] = []
        seen: set[str] = set()
        total_exposure = sum(
            p.quantity * p.current_price for p in portfolio.positions.values()
        )

        for order in all_orders:
            cost = order.quantity * (order.price or 0)
            if cost > settings.TRADING_CAPITAL:
                continue
            key = f"{order.symbol}:{order.side.value}"
            if key in seen:
                continue
            seen.add(key)
            new_exposure = total_exposure + cost
            if new_exposure > settings.TRADING_CAPITAL:
                continue
            final.append(order)

            # Track entry on new BUY
            if order.side == OrderSide.BUY and time_stop_enabled and market_data is not None:
                bars = market_data.bars.get(order.symbol)
                if bars:
                    self._hwm_tracker[order.symbol] = bars[-1].close
                    self._entry_time_tracker[order.symbol] = bars[-1].datetime

            total_exposure += cost

        return final


    # ------------------------------------------------------------------
    # Indicator methods (delegating to shared quanti.indicators)
    # ------------------------------------------------------------------

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        return sma(data, period)

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        return ema(data, period)

    @staticmethod
    def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> np.ndarray | None:
        return adx(high, low, close, period)

    @staticmethod
    def _adx_with_di(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return adx_with_di(high, low, close, period)

    @staticmethod
    def _bollinger_bands(closes: np.ndarray, period: int, std_mult: float):
        return bollinger_bands(closes, period, std_mult)

    @staticmethod
    def _wilder_smooth(data: np.ndarray, period: int) -> np.ndarray:
        return wilder_smooth(data, period)

    @staticmethod
    def _calc_strength(fast_ma_val: float, slow_ma_val: float, adx_val: float) -> float:
        """
        Calculate signal strength (0.0 to 1.0).
        (Legacy method -- used by tests and backward compat.)
        """
        if slow_ma_val <= 0:
            return 0.0
        ma_sep = (fast_ma_val - slow_ma_val) / slow_ma_val * 100
        ma_score = min(abs(ma_sep) / 3.0, 1.0)
        adx_score = min(adx_val / 40.0, 1.0)
        return 0.5 * ma_score + 0.5 * adx_score

    # ------------------------------------------------------------------
    # Exit helper methods (delegating to shared quanti.indicators)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(bars: list, period: int = 14) -> float | None:
        """Compute ATR via shared indicator module."""
        return compute_atr(bars, period)

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float | None:
        """Compute RSI via shared indicator module."""
        return compute_rsi(closes, period)

    # ------------------------------------------------------------------
    # Exit conditions (called from risk_check; also directly testable)
    # ------------------------------------------------------------------

    def _check_gap_risk(self, symbol: str, pos: "Position", bars: list) -> bool:
        """
        Pre-exit gap risk check: detect positions at risk of gapping
        through their stop-loss levels due to T+1 settlement lock-in.

        Checks two conditions:
        1. Intraday range (high-low) exceeds GAP_RISK_THRESHOLD_PCT of price
        2. Current stop distance < 2x average true range

        When true, the position is at risk of an intraday gap that cannot
        be exited due to T+1 settlement. The strategy should reduce position
        size or tighten stops preemptively.

        Args:
            symbol: ETF symbol
            pos: Position object (has avg_cost, current_price)
            bars: List of Bar objects (need high/low for intraday range)

        Returns:
            True if gap risk is elevated, False otherwise.
        """
        gap_enabled = getattr(settings, 'GAP_RISK_CHECK_ENABLED', True)
        if not gap_enabled:
            return False

        gap_threshold_pct = getattr(settings, 'GAP_RISK_THRESHOLD_PCT', 5.0)
        if len(bars) < 15 or pos.quantity <= 0 or pos.current_price <= 0:
            return False

        # 1. Intraday range check: today's range exceeds threshold
        current_bar = bars[-1]
        intraday_range_pct = (current_bar.high - current_bar.low) / current_bar.close * 100
        if intraday_range_pct < gap_threshold_pct:
            return False

        # 2. Stop distance check: stop level is within 2x ATR
        atr = self._compute_atr(bars, period=getattr(settings, 'ATR_PERIOD', 14))
        if atr is None or atr <= 0:
            return False

        # Estimate stop distance (% of price from current to stop level)
        stop_loss_pct = float(getattr(settings, 'STOP_LOSS_PCT', 0.0) or 0.0)
        if stop_loss_pct > 0:
            # Flat stop: distance from current price to stop level (e.g., 8% -> 0.08)
            stop_distance = stop_loss_pct / 100.0
        else:
            # ATR-based: distance from HWM to stop level = 2*ATR, as fraction of price
            atr_dist = 2 * atr
            stop_distance = atr_dist / pos.current_price if pos.current_price > 0 else 1.0

        atr_pct = atr / pos.current_price * 100
        return stop_distance < 0.02 * atr_pct  # stop within ~2% of ATR% range

    def _flat_stop_loss(self, symbol: str, positions: dict,
                        stop_loss_pct: float | None = None) -> Order | None:
        """
        Flat percentage stop-loss: exit when loss from avg_cost > stop_loss_pct.

        Args:
            symbol: ETF symbol
            positions: portfolio position dict (symbol -> Position)
            stop_loss_pct: stop-loss threshold percentage. Reads STOP_LOSS_PCT if None.

        Returns:
            SELL Order if stop triggered, None otherwise.
        """
        if stop_loss_pct is None:
            stop_loss_pct = float(getattr(settings, 'STOP_LOSS_PCT', 0.0) or 0.0)

        pos = positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            return None
        if stop_loss_pct <= 0 or pos.current_price <= 0 or pos.avg_cost <= 0:
            return None

        loss_pct = (pos.avg_cost - pos.current_price) / pos.avg_cost * 100
        if loss_pct > stop_loss_pct:
            return Order(
                symbol=symbol, side=OrderSide.SELL,
                quantity=pos.quantity, price=None, order_type="market",
                signal_ref=f'stop-loss: {loss_pct:.1f}% > {stop_loss_pct:.1f}%',
            )
        return None

    def _atr_trailing_stop(
        self, symbol: str, bars: list, atr_period: int | None = None, atr_mult: float | None = None,
        policy_intervention_score: float = 0.0,
    ) -> Order | None:
        """
        ATR trailing stop: track high water mark per position.
        Exit when price drops below HWM - atr_mult * ATR(14).

        Args:
            symbol: ETF symbol
            bars: List of Bar objects (most recent first)
            atr_period: ATR period
            atr_mult: ATR multiplier for stop distance

        Returns:
            SELL Order if stop triggered, None otherwise.
        """
        if atr_period is None:
            atr_period = getattr(settings, 'ATR_PERIOD', 14)
        if atr_mult is None:
            atr_mult = getattr(settings, 'ATR_TRAILING_MULTIPLIER', 2.0)

        if len(bars) < atr_period + 1:
            return None

        atr_val = self._compute_atr(bars, atr_period)
        if atr_val is None or atr_val <= 0:
            return None

        current_close = bars[-1].close
        hwm = self._hwm_tracker.get(symbol, current_close)
        if current_close > hwm:
            hwm = current_close
        self._hwm_tracker[symbol] = hwm

        # RSI tightening
        closes = np.array([b.close for b in bars], dtype=np.float64)
        rsi_period_setting = getattr(settings, 'RSI_PERIOD', 14)
        rsi = self._compute_rsi(closes, rsi_period_setting)
        rsi_overbought_setting = getattr(settings, 'RSI_OVERBOUGHT', 80)
        effective_mult = atr_mult
        if rsi is not None and rsi > rsi_overbought_setting:
            atr_tight_val = getattr(settings, 'ATR_TIGHTEN_MULTIPLIER', 1.5)
            effective_mult = atr_tight_val

        # Policy intervention tightening: when NT is active, tighten stops
        nt_exit_tighten = getattr(settings, 'NT_POLICY_EXIT_TIGHTEN', 0.7)
        if policy_intervention_score > 0.3:
            effective_mult *= nt_exit_tighten

        stop_level = hwm - effective_mult * atr_val
        if current_close < stop_level:
            self._hwm_tracker.pop(symbol, None)
            return Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=0,  # caller fills in actual qty
                price=None,
                order_type="market",
                signal_ref=f'ATR trailing stop: close={current_close:.4f} < {stop_level:.4f} (HWM={hwm:.4f}, ATR={atr_val:.4f}, mult={effective_mult})',
            )

        return None

    def _time_stop(
        self, symbol: str, bars: list, time_limit_days: int = 60, reduce_days: int = 40
    ) -> Order | None:
        """
        Time-based exit: reduce 50% at reduce_days, full exit at time_limit_days.

        The clock is measured from the entry datetime stored in _entry_time_tracker.
        Only triggers at reduce_days if no new high was made (price below HWM).

        Args:
            symbol: ETF symbol
            bars: List of Bar objects (to check current price vs HWM)
            time_limit_days: Full exit after this many days (default 60)
            reduce_days: 50% reduction after this many days (default 40)

        Returns:
            SELL Order if time stop triggered, None otherwise.
        """
        entry_dt = self._entry_time_tracker.get(symbol)
        if entry_dt is None:
            return None

        # Use bar datetime if available, else now
        now = bars[-1].datetime if bars and hasattr(bars[-1], 'datetime') else datetime.now()

        days_held = (now - entry_dt).days

        # Full exit at time limit
        if days_held >= time_limit_days:
            self._hwm_tracker.pop(symbol, None)
            self._entry_time_tracker.pop(symbol, None)
            return Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=0,  # caller fills in qty (full position)
                price=None,
                order_type="market",
                signal_ref=f'time stop: held {days_held}d >= {time_limit_days}d full exit',
            )

        # 50% reduction at reduce threshold (only if below HWM = no new high)
        if days_held >= reduce_days:
            hwm = self._hwm_tracker.get(symbol)
            current_close = bars[-1].close if bars else 0
            if hwm is not None and current_close < hwm:
                return Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=0,  # caller fills in qty (50% of position)
                    price=None,
                    order_type="market",
                    signal_ref=f'time stop: held {days_held}d >= {reduce_days}d, reduce 50%',
                )

        return None

    def _rsi_exit(
        self, symbol: str, bars: list,
        rsi_period: int | None = None,
        rsi_threshold: float | None = None,
        atr_mult_normal: float | None = None,
        atr_mult_tight: float | None = None,
    ) -> Order | None:
        """
        RSI-based exit: tighten ATR trailing stop multiplier when RSI > threshold.

        When RSI > 80 (overbought), the ATR trailing stop uses a tighter multiplier
        (1.5x instead of 2.0x), making exits more sensitive to pullbacks.

        Args:
            symbol: ETF symbol
            bars: List of Bar objects
            rsi_period: RSI lookback period (default 14)
            rsi_threshold: RSI overbought level (default 80)
            atr_mult_normal: Normal ATR multiplier (default 2.0)
            atr_mult_tight: Tightened multiplier when RSI > threshold (default 1.5)

        Returns:
            SELL Order if tightened stop triggered, None otherwise.
        """
        if rsi_period is None:
            rsi_period = getattr(settings, 'RSI_PERIOD', 14)
        if rsi_threshold is None:
            rsi_threshold = getattr(settings, 'RSI_OVERBOUGHT', 80.0)
        if atr_mult_normal is None:
            atr_mult_normal = getattr(settings, 'ATR_TRAILING_MULTIPLIER', 2.0)
        if atr_mult_tight is None:
            atr_mult_tight = getattr(settings, 'ATR_TIGHTEN_MULTIPLIER', 1.5)

        if len(bars) < rsi_period + 2:
            return None

        closes = np.array([b.close for b in bars], dtype=np.float64)
        rsi = self._compute_rsi(closes, rsi_period)
        if rsi is None or rsi <= rsi_threshold:
            return None  # RSI not overbought, no special exit

        # When RSI > threshold, check ATR stop with tighter multiplier
        return self._atr_trailing_stop(
            symbol, bars, atr_period=rsi_period, atr_mult=atr_mult_tight,
        )

    def _volatility_stop(
        self, symbol: str, bars: list, atr_period: int = 14,
        expansion_mult: float = 1.5,
    ) -> Order | None:
        """
        Volatility-based (chaos) stop: exit when current ATR expands
        significantly compared to ATR at entry time.

        When ATR_current > entry_ATR * expansion_mult, it signals
        chaotic market conditions -- exit to preserve capital.

        Args:
            symbol: ETF symbol
            bars: List of Bar objects
            atr_period: ATR period (default 14)
            expansion_mult: Expansion threshold multiplier (default 1.5)

        Returns:
            SELL Order if volatility stop triggered, None otherwise.
        """
        if len(bars) < atr_period + 1:
            return None

        entry_atr = self._entry_atr_tracker.get(symbol)
        if entry_atr is None:
            # First time seeing this symbol: set entry ATR
            atr_val = self._compute_atr(bars, atr_period)
            if atr_val is not None and atr_val > 0:
                self._entry_atr_tracker[symbol] = atr_val
            return None

        current_atr = self._compute_atr(bars, atr_period)
        if current_atr is None or current_atr <= 0:
            return None

        if current_atr > entry_atr * expansion_mult:
            self._entry_atr_tracker.pop(symbol, None)
            self._hwm_tracker.pop(symbol, None)
            return Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=0,  # caller fills in actual qty
                price=None,
                order_type="market",
                signal_ref=f'volatility stop: ATR({current_atr:.4f}) > {expansion_mult}x entry ATR({entry_atr:.4f})',
            )

        return None

    # ------------------------------------------------------------------
    # Exit tracking management
    # ------------------------------------------------------------------

    def register_entry(
        self, symbol: str, entry_price: float, entry_dt: datetime,
        entry_atr: float | None = None,
    ) -> None:
        """Register a new position entry for exit tracking."""
        self._hwm_tracker[symbol] = entry_price
        self._entry_time_tracker[symbol] = entry_dt
        if entry_atr is not None and entry_atr > 0:
            self._entry_atr_tracker[symbol] = entry_atr

    def register_exit(self, symbol: str) -> None:
        """Clear exit tracking state when position is fully closed."""
        self._hwm_tracker.pop(symbol, None)
        self._entry_time_tracker.pop(symbol, None)
        self._entry_atr_tracker.pop(symbol, None)

    @staticmethod
    def _get_last_price(symbol: str, portfolio: Portfolio) -> float | None:
        """Get the current price for a symbol from portfolio positions."""
        pos = portfolio.positions.get(symbol)
        if pos:
            return pos.current_price
        # In live mode, would use market data. For backtest, use close price.
        return None

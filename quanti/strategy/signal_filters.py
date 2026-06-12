"""
Market environment filters extracted from strategy for reuse and independent testing.

These filters gate all trading activity:
- is_trending: At least one major index has ADX > threshold
- is_bear_market: Index 120-day MA is declining
- is_forbidden_period: Manual calendar check (e.g., holidays, blackout dates)
- get_position_size_multiplier: 1.0 for bull, defense_pct for bear
"""
from collections.abc import Callable
from datetime import date

import numpy as np

from quanti.config import settings


class MarketEnvironmentFilter:
    """
    Market environment checks that gate all trading activity.

    Parameters:
    - market_adx_threshold: ADX threshold for market trending check (default 20)
    - index_sma_long: Long SMA period for bear market detection (default 120)
    - defense_pct: Position size multiplier during bear market (default 0.20)
    """

    def __init__(
        self,
        market_adx_threshold: float = 20.0,
        index_sma_long: int = 120,
        defense_pct: float = 0.20,
    ):
        self.market_adx_threshold = market_adx_threshold
        self.index_sma_long = index_sma_long
        self.defense_pct = defense_pct

        # Manually maintained forbidden dates (YYYY-MM-DD).
        # Add dates like option expiry, major macro events, holidays.
        self._forbidden_dates: set[str] = set()

    # ------------------------------------------------------------------
    # Market environment checks
    # ------------------------------------------------------------------

    def is_trending(
        self,
        index_bars: dict[str, list],
        adx_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray | None],
    ) -> bool:
        """
        Check if at least one major index has ADX > market_adx_threshold.

        Args:
            index_bars: dict of index_name -> list[Bar]
            adx_fn: function(highs, lows, closes, period) -> adx_array | None

        Returns:
            True if any index has ADX > threshold, False if none do
            (or if no index data is provided).
        """
        if not index_bars:
            return True  # Permissive: no data = no filter

        for _index_name, bars in index_bars.items():
            if len(bars) < 30:
                continue
            closes = np.array([b.close for b in bars], dtype=np.float64)
            highs = np.array([b.high for b in bars], dtype=np.float64)
            lows = np.array([b.low for b in bars], dtype=np.float64)
            adx = adx_fn(highs, lows, closes, 14)
            if (adx is not None and len(adx) > 0 and not np.isnan(adx[-1])
                    and adx[-1] > self.market_adx_threshold):
                return True

        return False

    def is_bear_market(
        self,
        index_bars: dict[str, list],
        sma_fn: Callable[[np.ndarray, int], np.ndarray],
    ) -> bool:
        """
        Check if the primary index 120-day MA is declining (bear market signal).

        A declining long-term MA indicates a structural downtrend.
        We check the slope over the last 5 bars.

        Args:
            index_bars: dict of index_name -> list[Bar]
            sma_fn: function(data, period) -> sma_array

        Returns:
            True if the first usable index has a declining 120-day SMA.
            False if no index data or SMA is flat/rising.
        """
        if not index_bars:
            return False  # No data = assume bull

        # Use the first index with enough data
        for _index_name, bars in index_bars.items():
            if len(bars) < self.index_sma_long + 5:
                continue
            closes = np.array([b.close for b in bars], dtype=np.float64)
            sma = sma_fn(closes, self.index_sma_long)
            if sma is None or len(sma) < 5:
                continue

            # Check slope over last 5 bars
            recent = sma[-5:]
            valid = recent[~np.isnan(recent)]
            if len(valid) < 3:
                continue

            # Simple linear regression slope
            x = np.arange(len(valid), dtype=np.float64)
            slope = np.polyfit(x, valid, 1)[0]

            return slope < 0  # Negative slope = declining = bear

        return False  # Insufficient data: assume bull

    # ------------------------------------------------------------------
    # Calendar-based filters
    # ------------------------------------------------------------------

    def add_forbidden_date(self, dt: date) -> None:
        """Add a date to the manually maintained forbidden list."""
        self._forbidden_dates.add(dt.isoformat())

    def remove_forbidden_date(self, dt: date) -> None:
        """Remove a date from the forbidden list."""
        self._forbidden_dates.discard(dt.isoformat())

    def is_forbidden_period(self, dt: date | None = None) -> bool:
        """
        Check whether the given date falls in a manually maintained
        forbidden period (e.g., major holidays, blackout windows).

        Args:
            dt: Date to check. Defaults to today.

        Returns:
            True if the date is in the forbidden set.
        """
        if dt is None:
            dt = date.today()
        return dt.isoformat() in self._forbidden_dates

    # ------------------------------------------------------------------
    # Position sizing helpers
    # ------------------------------------------------------------------

    def get_position_size_multiplier(
        self,
        index_bars: dict[str, list],
        sma_fn: Callable[[np.ndarray, int], np.ndarray],
    ) -> float:
        """
        Return the position size multiplier based on market regime.

        - Bull market (not bear): 1.0 (full size)
        - Bear market: defense_pct (default 0.20)

        This allows the strategy to stay in the market at reduced size
        rather than going completely flat during bear regimes.
        """
        if self.is_bear_market(index_bars, sma_fn):
            return self.defense_pct
        return 1.0

    # ------------------------------------------------------------------
    # Composite check
    # ------------------------------------------------------------------

    def should_trade(
        self,
        index_bars: dict[str, list],
        adx_fn: Callable,
        sma_fn: Callable,
        dt: date | None = None,
    ) -> bool:
        """
        Composite gate: only trade if:
        1. Market is trending (ADX > threshold on at least one index)
        2. Not in a forbidden period

        Args:
            index_bars: Index data for ADX / SMA checks
            adx_fn: ADX computation function
            sma_fn: SMA computation function
            dt: Date to check (defaults to today)

        Returns:
            True if all conditions allow trading.
        """
        if self.is_forbidden_period(dt):
            return False
        return self.is_trending(index_bars, adx_fn)

    # ------------------------------------------------------------------
    # National Team intervention detection
    # ------------------------------------------------------------------

    def detect_nt_intervention(
        self,
        etf_bars: dict[str, list],
        sigma_threshold: float = 3.0,
        lookback_days: int = 60,
    ) -> bool:
        """
        Detect likely National Team intervention via abnormal ETF volume.

        The National Team (Central Huijin, etc.) holds ~1.54 trillion RMB
        in ETFs. When it executes large-scale purchases, daily ETF volumes
        spike far above normal levels without corresponding news catalysts.
        This method detects such spikes.

        Algorithm:
        1. For each ETF in the universe, compute current volume vs. 60-day
           mean and standard deviation.
        2. If ANY ETF's volume exceeds mean + sigma_threshold * std, flag it.
        3. If at least 2 ETFs in the universe show abnormal volume, return True.

        Args:
            etf_bars: dict of ETF symbol -> list[Bar] with .volume attribute
            sigma_threshold: Number of standard deviations for anomaly (default 3.0)
            lookback_days: Lookback window for mean/std calculation (default 60)

        Returns:
            True if National Team intervention is likely occurring.
        """
        if not etf_bars or len(etf_bars) == 0:
            return False

        abnormal_count = 0
        for _symbol, bars in etf_bars.items():
            if len(bars) < lookback_days + 1:
                continue

            volumes = np.array([b.volume for b in bars], dtype=np.float64)
            current_vol = volumes[-1]
            historical = volumes[-(lookback_days + 1):-1]

            mean_vol = np.mean(historical)
            std_vol = np.std(historical, ddof=1)

            if std_vol <= 0:
                continue

            z_score = (current_vol - mean_vol) / std_vol
            if z_score > sigma_threshold:
                abnormal_count += 1

        # Require at least 2 ETFs showing abnormal volume to flag NT intervention
        return abnormal_count >= 2

    def is_intervention_day(
        self,
        etf_bars: dict[str, list],
    ) -> bool:
        """
        Check if today shows signs of National Team intervention.

        Convenience wrapper around detect_nt_intervention() using
        settings-provided thresholds.

        Args:
            etf_bars: dict of ETF symbol -> list[Bar]

        Returns:
            True if NT intervention is likely active today.
        """
        enabled = getattr(settings, 'NT_INTERVENTION_DETECTION_ENABLED', True)
        if not enabled:
            return False
        sigma = getattr(settings, 'NT_VOLUME_SIGMA_THRESHOLD', 3.0)
        lookback = getattr(settings, 'NT_VOLUME_LOOKBACK_DAYS', 60)
        return self.detect_nt_intervention(etf_bars, sigma_threshold=sigma, lookback_days=lookback)

    def get_policy_intervention_score(
        self,
        etf_bars: dict[str, list],
    ) -> float:
        """
        Return a score from 0.0 to 1.0 indicating the likelihood and
        intensity of policy (National Team) intervention in the market.

        0.0 = no intervention detected (normal market)
        0.5 = moderate intervention (1-2 ETFs abnormal)
        1.0 = heavy intervention (3+ ETFs abnormal, large z-scores)

        This score should be consumed by exit methods to tighten stops
        and time limits when the market is being artificially supported.

        Args:
            etf_bars: dict of ETF symbol -> list[Bar]

        Returns:
            Float in [0.0, 1.0] representing intervention intensity.
        """
        if not etf_bars or len(etf_bars) == 0:
            return 0.0

        sigma_threshold = getattr(settings, 'NT_VOLUME_SIGMA_THRESHOLD', 3.0)
        lookback = getattr(settings, 'NT_VOLUME_LOOKBACK_DAYS', 60)

        max_z = 0.0
        abnormal_count = 0

        for _symbol, bars in etf_bars.items():
            if len(bars) < lookback + 1:
                continue
            volumes = np.array([b.volume for b in bars], dtype=np.float64)
            current_vol = volumes[-1]
            historical = volumes[-(lookback + 1):-1]
            mean_vol = np.mean(historical)
            std_vol = np.std(historical, ddof=1)
            if std_vol <= 0:
                continue
            z = (current_vol - mean_vol) / std_vol
            if z > sigma_threshold:
                abnormal_count += 1
                max_z = max(max_z, z)

        # Score formula: sigmoid-like ramp from 0 to 1
        if abnormal_count == 0:
            return 0.0

        # Normalize: 3 sigma -> 0.3 base, 6 sigma -> 0.6 base, capped at 1.0
        intensity = min(max_z / (sigma_threshold * 2), 1.0)
        # Multiplier from count: 1 ETF -> 0.5, 3+ ETFs -> 1.0
        count_mult = min((abnormal_count - 1) / 2 + 0.5, 1.0)
        score = intensity * count_mult
        return float(np.clip(score, 0.0, 1.0))

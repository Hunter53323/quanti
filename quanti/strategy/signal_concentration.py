"""
Signal concentration monitor: detect algorithmic herding.

When hundreds of quantitative funds run similar momentum strategies
on the same ETF universe, they generate clustered signals that appear
as "confirmation" to each other. This module detects that pattern
and reduces position size or suppresses signals when cluster risk is high.
"""
from datetime import datetime

import numpy as np

from quanti.config import settings


class SignalConcentrationMonitor:
    """
    Detect when multiple strategy instances or conditions produce
    synchronized buy signals on the same symbols, indicating possible
    algorithmic resonance (herding) rather than genuine dispersion.

    Parameters:
    - threshold: Minimum number of simultaneous signals to flag as herd (default 3)
    """

    def __init__(self, threshold: int | None = None):
        self.threshold = threshold or getattr(settings, 'SIGNAL_CONCENTRATION_THRESHOLD', 3)
        self._signal_history: list[tuple[datetime, str, str]] = []

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect_herding(
        self,
        signals_by_symbol: dict[str, int],
    ) -> tuple[bool, float]:
        """
        Check if today's signals show clustering indicative of algorithmic herding.

        A "cluster" is defined as multiple signals firing on the same symbol
        within the same decision cycle.

        Args:
            signals_by_symbol: dict of symbol -> count of signals today

        Returns:
            (is_herding: bool, concentration_score: float in [0.0, 1.0])
        """
        if not signals_by_symbol:
            return (False, 0.0)

        counts = list(signals_by_symbol.values())
        if len(counts) == 0:
            return (False, 0.0)

        max_concentration = max(counts)
        total_signals = sum(counts)
        unique_symbols = len(counts)

        if unique_symbols == 0 or total_signals == 0:
            return (False, 0.0)

        # Gini-like concentration ratio
        concentration_ratio = max_concentration / max(total_signals, 1)
        herding_score = min(concentration_ratio * max_concentration / self.threshold, 1.0)

        is_herding = max_concentration >= self.threshold
        return (is_herding, float(herding_score))

    # ------------------------------------------------------------------
    # Volume composition analysis
    # ------------------------------------------------------------------

    def analyze_volume_composition(
        self,
        bars: list,
        lookback_days: int = 20,
    ) -> float:
        """
        Estimate the proportion of today's volume that is likely algorithmic
        rather than organic.

        Heuristic: algorithmic volume tends to be more uniform in size
        (many small orders executed programmatically), while organic volume
        has more variance in trade size distribution.

        Since we only have daily OHLCV (not tick data), we use a proxy:
        - Compare today's volume to the distribution of recent volumes
        - A single extreme outlier suggests a large block trade (organic)
        - A moderate but persistent elevation suggests algorithmic flow

        Args:
            bars: List of Bar objects with .volume attribute
            lookback_days: Window for volume distribution estimation

        Returns:
            Float in [0.0, 1.0] where higher = more algorithmic composition
        """
        if len(bars) < lookback_days + 1:
            return 0.5

        volumes = np.array([b.volume for b in bars], dtype=np.float64)
        current_vol = volumes[-1]
        historical = volumes[-(lookback_days + 1):-1]

        mean_vol = np.mean(historical)
        std_vol = np.std(historical, ddof=1)

        if std_vol <= 0 or mean_vol <= 0:
            return 0.5

        z_score = (current_vol - mean_vol) / std_vol

        if z_score < 1.5:
            return 0.2
        elif z_score < 2.5:
            recent_elevated = np.sum(historical[-5:] > mean_vol * 1.2)
            if recent_elevated >= 3:
                return 0.7
            return 0.4
        else:
            return 0.3

    # ------------------------------------------------------------------
    # Composite risk assessment
    # ------------------------------------------------------------------

    def get_concentration_risk(
        self,
        signals_by_symbol: dict[str, int],
        bars_dict: dict[str, list],
    ) -> dict:
        """
        Full concentration risk assessment combining herding detection
        and volume composition analysis.

        Args:
            signals_by_symbol: dict of symbol -> count of signals today
            bars_dict: dict of symbol -> list[Bar] for volume analysis

        Returns:
            {
                "is_herding": bool,
                "concentration_score": float (0-1),
                "volume_composition_score": float (0-1, avg across symbols),
                "combined_risk_score": float (0-1),
                "recommendation": str ("normal", "reduce_size", "skip")
            }
        """
        is_herding, concentration_score = self.detect_herding(signals_by_symbol)

        vol_scores = []
        for symbol in signals_by_symbol:
            bars = bars_dict.get(symbol, [])
            if bars:
                vol_scores.append(self.analyze_volume_composition(bars))

        avg_vol_score = np.mean(vol_scores).item() if vol_scores else 0.5

        combined = concentration_score * 0.6 + avg_vol_score * 0.4

        if combined < 0.3:
            recommendation = "normal"
        elif combined < 0.6:
            recommendation = "reduce_size"
        else:
            recommendation = "skip"

        return {
            "is_herding": is_herding,
            "concentration_score": round(concentration_score, 4),
            "volume_composition_score": round(avg_vol_score, 4),
            "combined_risk_score": round(combined, 4),
            "recommendation": recommendation,
        }

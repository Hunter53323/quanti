"""Tests for market-structure-aware defenses: NT intervention and signal concentration."""
import numpy as np

from quanti.strategy.signal_concentration import SignalConcentrationMonitor
from quanti.strategy.signal_filters import MarketEnvironmentFilter


def make_etf_bars(volumes, symbol="510300"):
    """Create synthetic Bar objects with .volume attribute."""
    class FakeBar:
        def __init__(self, vol):
            self.close = 10.0
            self.high = 10.1
            self.low = 9.9
            self.volume = vol

    return [FakeBar(v) for v in volumes]


class TestNTInterventionDetection:
    """National Team volume anomaly detection."""

    def test_no_intervention_normal_volume(self):
        """Normal volume with natural variance -> no intervention detected."""
        rng = np.random.RandomState(42)
        vols = list(1_000_000 + rng.randn(61) * 100_000)
        bars = make_etf_bars(vols)
        flt = MarketEnvironmentFilter()
        result = flt.detect_nt_intervention({"510300": bars}, sigma_threshold=3.0, lookback_days=60)
        assert result is False

    def test_intervention_detected_extreme_volume(self):
        """Extreme volume spike on 2+ ETFs -> intervention detected."""
        rng = np.random.RandomState(42)
        # Use natural variance (std > 0) + spike
        vols_base = list(1_000_000 + rng.randn(60) * 100_000)
        vols_spike = vols_base + [10_000_000]
        bars1 = make_etf_bars(vols_spike, "512480")
        bars2 = make_etf_bars(vols_spike, "512000")
        flt = MarketEnvironmentFilter()
        result = flt.detect_nt_intervention(
            {"512480": bars1, "512000": bars2}, sigma_threshold=3.0, lookback_days=60
        )
        assert result is True

    def test_single_etf_spike_not_enough(self):
        """Single ETF spike with one normal ETF -> no intervention (need 2+)."""
        rng = np.random.RandomState(42)
        vols_base = list(1_000_000 + rng.randn(60) * 100_000)
        vols_spike = vols_base + [10_000_000]
        vols_normal = vols_base + [1_200_000]
        bars1 = make_etf_bars(vols_spike, "512480")
        bars2 = make_etf_bars(vols_normal, "512000")
        flt = MarketEnvironmentFilter()
        result = flt.detect_nt_intervention(
            {"512480": bars1, "512000": bars2}, sigma_threshold=3.0, lookback_days=60
        )
        assert result is False

    def test_is_intervention_day(self):
        """Convenience wrapper returns correct boolean for spike."""
        rng = np.random.RandomState(42)
        vols_base = list(1_000_000 + rng.randn(60) * 100_000)
        vols_spike = vols_base + [10_000_000]
        bars1 = make_etf_bars(vols_spike, "512480")
        bars2 = make_etf_bars(vols_spike, "512000")
        flt = MarketEnvironmentFilter()
        result = flt.is_intervention_day({"512480": bars1, "512000": bars2})
        assert result is True

    def test_policy_score_zero_no_intervention(self):
        """No intervention -> score is 0.0."""
        rng = np.random.RandomState(42)
        vols = list(1_000_000 + rng.randn(61) * 100_000)
        bars = make_etf_bars(vols)
        flt = MarketEnvironmentFilter()
        score = flt.get_policy_intervention_score({"510300": bars})
        assert score == 0.0

    def test_policy_score_positive_intervention(self):
        """Heavy intervention on multiple ETFs -> score > 0.0."""
        rng = np.random.RandomState(42)
        vols_base = list(1_000_000 + rng.randn(60) * 100_000)
        vols_spike = vols_base + [20_000_000]
        bars = make_etf_bars(vols_spike, "512480")
        flt = MarketEnvironmentFilter()
        score = flt.get_policy_intervention_score({
            "512480": bars,
            "512000": make_etf_bars(vols_spike, "512000"),
        })
        assert score > 0.0
        assert score <= 1.0

    def test_insufficient_bars(self):
        """Less than lookback+1 bars -> no false positive."""
        vols = [1_000_000] * 30  # fewer than 61 needed
        bars = make_etf_bars(vols)
        flt = MarketEnvironmentFilter()
        result = flt.detect_nt_intervention({"510300": bars}, sigma_threshold=3.0, lookback_days=60)
        assert result is False


class TestSignalConcentration:
    """Algorithmic herding detection via signal concentration analysis."""

    def test_no_herding_dispersed_signals(self):
        """Signals spread across many symbols -> no herding."""
        mon = SignalConcentrationMonitor(threshold=3)
        signals = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}
        is_herding, score = mon.detect_herding(signals)
        assert is_herding is False
        assert score < 0.5

    def test_herding_detected_concentrated(self):
        """Multiple signals on few symbols -> herding detected."""
        mon = SignalConcentrationMonitor(threshold=3)
        signals = {"A": 5, "B": 1}
        is_herding, score = mon.detect_herding(signals)
        assert is_herding is True
        assert score > 0.5

    def test_concentration_risk_normal(self):
        """Dispersed signals + normal volume -> normal recommendation."""
        mon = SignalConcentrationMonitor(threshold=3)
        # More dispersed signals -> lower concentration -> recommendation "normal"
        signals = {"A": 1, "B": 1, "C": 1, "D": 1}
        bars_dict = {}
        result = mon.get_concentration_risk(signals, bars_dict)
        assert result["is_herding"] is False
        assert result["recommendation"] == "normal"

    def test_empty_signals(self):
        """No signals -> no herding, zero score."""
        mon = SignalConcentrationMonitor(threshold=3)
        is_herding, score = mon.detect_herding({})
        assert is_herding is False
        assert score == 0.0

    def test_volume_composition_extreme(self):
        """Extreme volume outlier -> likely block trade, score ~0.3."""
        vols = [1_000_000] * 30 + [10_000_000]
        bars = [type("FakeBar", (), {"volume": v})() for v in vols]
        mon = SignalConcentrationMonitor()
        score = mon.analyze_volume_composition(bars, lookback_days=20)
        assert 0 < score < 1.0

    def test_volume_insufficient_data(self):
        """Insufficient bars -> neutral score."""
        vols = [1_000_000] * 10
        bars = [type("FakeBar", (), {"volume": v})() for v in vols]
        mon = SignalConcentrationMonitor()
        score = mon.analyze_volume_composition(bars, lookback_days=20)
        assert score == 0.5



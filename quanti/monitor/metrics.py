"""
Monitoring: metrics collection.
In-process registry of gauges and counters pushed to structured logs.
"""

from datetime import datetime


class MetricsRegistry:
    """Simple in-process metrics collector."""

    def __init__(self):
        self._gauges: dict[str, float] = {}
        self._counters: dict[str, int] = {}
        self._timers: dict[str, list[float]] = {}

    # ---- Gauge (point-in-time value) ----

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def get_gauge(self, name: str) -> float:
        return self._gauges.get(name, 0.0)

    # ---- Counter (monotonically increasing) ----

    def inc_counter(self, name: str, delta: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + delta

    def get_counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    # ---- Timer (track latency distributions) ----

    def record_timing(self, name: str, ms: float) -> None:
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(ms)

    def get_timing_stats(self, name: str) -> dict:
        values = self._timers.get(name, [])
        if not values:
            return {"count": 0, "avg_ms": 0, "max_ms": 0, "min_ms": 0}
        return {
            "count": len(values),
            "avg_ms": sum(values) / len(values),
            "max_ms": max(values),
            "min_ms": min(values),
        }

    # ---- Snapshot ----

    def snapshot(self) -> dict:
        """Return all metrics as a single dict for logging/export."""
        return {
            "timestamp": datetime.now().isoformat(),
            "gauges": dict(self._gauges),
            "counters": dict(self._counters),
            "timers": {
                name: self.get_timing_stats(name)
                for name in self._timers
            },
        }


# Global singleton
_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _metrics

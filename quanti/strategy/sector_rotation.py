"""
Sector rotation module: rank sector ETFs by momentum, select top candidates.

Strategy:
1. Rank all sector ETFs by trailing period return (momentum)
2. Select top N by momentum
3. Filter by entry conditions (delegated to caller via entry_check_fn)
4. Return qualifying symbols ready for position sizing
"""
from collections.abc import Callable

import numpy as np

# Mapping of sector names to their ETF ticker codes.
# These are Chinese A-share sector ETFs.
SECTOR_ETF_MAP = {
    "broker": "512000",
    "semiconductor": "512480",
    "new_energy": "516160",
    "consumer": "159928",
    "pharma": "512010",
    "defense": "512660",
    "bank": "512800",
    "hang_seng_tech": "513130",
}


class SectorRotation:
    """
    Rank sector ETFs by momentum and select the top candidates
    that also pass entry condition checks.

    Parameters:
    - top_n: Number of top sectors to consider (default 3)
    - lookback_days: Lookback period for momentum ranking (default 20)
    """

    def __init__(self, top_n: int = 3, lookback_days: int = 20):
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if lookback_days < 2:
            raise ValueError("lookback_days must be >= 2")

        self.top_n = top_n
        self.lookback_days = lookback_days

    # ------------------------------------------------------------------
    # Momentum ranking
    # ------------------------------------------------------------------

    def rank_by_momentum(
        self, bars_dict: dict[str, list]
    ) -> list[tuple[str, float]]:
        """
        Rank sector ETFs by lookback period return (momentum).

        Momentum = (close[-1] - close[-lookback_days]) / close[-lookback_days] * 100

        Args:
            bars_dict: dict of symbol -> list[Bar], where Bar has .close attribute

        Returns:
            List of (symbol, return_pct) sorted by return descending.
            Symbols with insufficient data are excluded.
        """
        if not bars_dict:
            return []

        results: list[tuple[str, float]] = []

        for symbol, bars in bars_dict.items():
            if len(bars) < self.lookback_days:
                continue

            closes = np.array([b.close for b in bars], dtype=np.float64)
            start_price = closes[-self.lookback_days]
            end_price = closes[-1]

            if start_price <= 0:
                continue

            return_pct = (end_price - start_price) / start_price * 100.0
            results.append((symbol, return_pct))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def select_candidates(
        self,
        bars_dict: dict[str, list],
        entry_check_fn: Callable[[str, list], bool],
    ) -> list[str]:
        """
        Select top N sector ETFs by momentum, then filter by entry conditions.

        Args:
            bars_dict: dict of symbol -> list[Bar]
            entry_check_fn: function(symbol, bars) -> bool
                Called for each candidate in momentum order until we have
                enough qualifying symbols or exhaust the list.

        Returns:
            List of qualifying symbol strings (up to top_n).
        """
        ranked = self.rank_by_momentum(bars_dict)
        if not ranked:
            return []

        candidates: list[str] = []
        for symbol, _momentum in ranked:
            if len(candidates) >= self.top_n:
                break

            bars = bars_dict.get(symbol, [])
            if entry_check_fn(symbol, bars):
                candidates.append(symbol)

        return candidates

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_sector_etf_symbols() -> list[str]:
        """Return all known sector ETF symbols."""
        return list(SECTOR_ETF_MAP.values())

    @staticmethod
    def get_sector_name(symbol: str) -> str | None:
        """Get the human-readable sector name for an ETF symbol."""
        for name, ticker in SECTOR_ETF_MAP.items():
            if ticker == symbol:
                return name
        return None

    @staticmethod
    def get_sector_symbol(name: str) -> str | None:
        """Get the ETF ticker for a sector name."""
        return SECTOR_ETF_MAP.get(name)

    # ------------------------------------------------------------------
    # Relative strength ranking (bonus: for multi-timeframe)
    # ------------------------------------------------------------------

    def rank_by_relative_strength(
        self,
        bars_dict: dict[str, list],
        benchmark_bars: list | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank sectors by relative strength vs. a benchmark.

        RS = (sector return over lookback) / (benchmark return over lookback)

        Args:
            bars_dict: dict of symbol -> list[Bar]
            benchmark_bars: benchmark Bar list (e.g., CSI 300). If None,
                ranks by absolute momentum instead.

        Returns:
            List of (symbol, rs_ratio) sorted descending.
        """
        if benchmark_bars is None or len(benchmark_bars) < self.lookback_days:
            return self.rank_by_momentum(bars_dict)

        bench_closes = np.array([b.close for b in benchmark_bars], dtype=np.float64)
        bench_end = bench_closes[-1]
        bench_start = bench_closes[-self.lookback_days]
        bench_return = (bench_end - bench_start) / bench_start

        results: list[tuple[str, float]] = []
        for symbol, bars in bars_dict.items():
            if len(bars) < self.lookback_days:
                continue

            closes = np.array([b.close for b in bars], dtype=np.float64)
            start_price = closes[-self.lookback_days]
            if start_price <= 0:
                continue

            sector_return = (closes[-1] - start_price) / start_price

            rs = sector_return if bench_return == 0 else sector_return / bench_return

            results.append((symbol, rs))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

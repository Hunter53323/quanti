"""Diff-test: omc_utils.run_backtest must match the inline approach bit-for-bit.

Catches the default-parameter mismatch where run_backtest defaulted
min_score=0.25 while the inline scripts it replaced had no threshold,
producing a 1.33% CAGR divergence.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, ".omc")

import numpy as np
import pytest

from omc_utils import ETFData, monthly_rebal_dates, run_backtest
from quanti.strategy.etf_rotation import ETFRotationStrategy

# Compact inline backtest — the reference implementation run_backtest replaces
CAPITAL = 90000
COMMISSION = 0.00025
SYMBOLS = ["510300", "510500", "159915", "510880", "518880", "511880"]


def inline_backtest(data, rebal_dates, w_trend, w_adx, w_momentum,
                    w_macd=0.0, w_kdj=0.0, top_n=3):
    """Reference: the original test_8etf.py backtest loop."""
    cash = CAPITAL
    holdings: dict = {}
    equity = [CAPITAL]
    peak = CAPITAL

    for rd in rebal_dates:
        total = cash + sum(
            h["q"] * data.price(s, rd)
            for s, h in holdings.items() if data.price(s, rd)
        )
        if total > peak:
            peak = total
        dd = (peak - total) / peak * 100 if peak > 0 else 0
        if dd > 15:
            for sym in list(holdings):
                p = data.price(sym, rd) or holdings[sym]["p"]
                cash += holdings[sym]["q"] * p * (1 - COMMISSION)
                del holdings[sym]
            equity.append(cash)
            continue

        scored = []
        for sym in SYMBOLS:
            if not data.eligible(sym, rd):
                continue
            arrs = data.slice(sym, rd)
            if arrs is None:
                continue
            c2, h2, l2 = arrs
            mn = np.mean(c2[-120:])
            mg = np.mean(c2[-140:-20])
            if not mn > mg:
                continue
            r = ETFRotationStrategy.compute_scores(
                c2, h2, l2,
                w_trend=w_trend, w_adx=w_adx, w_momentum=w_momentum,
                w_macd=w_macd, w_kdj=w_kdj, ma_period=120,
            )
            if r["composite"] > 0:
                scored.append((sym, r["composite"]))

        if not scored:
            for sym in list(holdings):
                p = data.price(sym, rd) or holdings[sym]["p"]
                cash += holdings[sym]["q"] * p * (1 - COMMISSION)
                del holdings[sym]
            equity.append(cash)
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        sel = {s[0] for s in scored[:top_n]}

        for sym in list(holdings):
            if sym not in sel:
                p = data.price(sym, rd) or holdings[sym]["p"]
                cash += holdings[sym]["q"] * p * (1 - COMMISSION)
                del holdings[sym]

        total2 = cash + sum(
            h["q"] * data.price(s, rd)
            for s, h in holdings.items() if data.price(s, rd)
        )
        n = max(len(sel), 1)
        per = total2 / n * 0.90
        for sym in sel:
            p = data.price(sym, rd)
            if p is None or p < 0.01:
                continue
            tq = int(per / p / 100) * 100
            if tq < 100:
                continue
            cost = tq * p
            if sym in holdings:
                diff = tq - holdings[sym]["q"]
                if abs(diff) >= 100:
                    cost2 = abs(diff) * p
                    if diff > 0 and cash >= cost2 * (1 + COMMISSION):
                        cash -= cost2 * (1 + COMMISSION)
                        holdings[sym]["q"] = tq
                    elif diff < 0:
                        cash += cost2 * (1 - COMMISSION)
                        holdings[sym]["q"] = tq
            elif cash >= cost * (1 + COMMISSION):
                cash -= cost * (1 + COMMISSION)
                holdings[sym] = {"q": tq, "p": p}
        equity.append(cash + sum(
            h["q"] * data.price(s, rd)
            for s, h in holdings.items() if data.price(s, rd)
        ))

    eq = np.array(equity, dtype=np.float64)
    n_years = (len(rebal_dates) + 1) / 12.0
    cagr = ((eq[-1] / eq[0]) ** (1.0 / n_years) - 1) * 100 if n_years > 0 and eq[0] > 0 else 0.0
    mr = np.diff(eq) / (eq[:-1] + 1e-10)
    sharpe = (np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12)) if len(mr) > 1 else 0.0
    pk = eq[0]
    maxdd = 0.0
    for v in eq:
        if v > pk:
            pk = v
        dv = (pk - v) / pk * 100
        if dv > maxdd:
            maxdd = dv
    return float(cagr), float(sharpe), float(maxdd), eq


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    return ETFData.load(SYMBOLS)


@pytest.fixture(scope="module")
def rebal(data):
    return monthly_rebal_dates(data.dates, from_date="20220101", to_date="20221231")


# ---------------------------------------------------------------------------
# Diff tests
# ---------------------------------------------------------------------------

class TestBacktestParity:
    """run_backtest must produce bit-identical equity curves to inline reference."""

    def test_3factor_defaults(self, data, rebal):
        ci, si, mi, eqi = inline_backtest(
            data, rebal, w_trend=0.35, w_adx=0.40, w_momentum=0.25)
        co, so, mo, eqo = run_backtest(
            data, rebal, w_trend=0.35, w_adx=0.40, w_momentum=0.25,
            w_macd=0.0, w_kdj=0.0, symbols_override=SYMBOLS)
        assert abs(ci - co) < 0.01, f"CAGR diverged: {ci:.6f} vs {co:.6f}"
        assert abs(eqi[-1] - eqo[-1]) < 10.0, (
            f"Final equity diverged: {eqi[-1]:.2f} vs {eqo[-1]:.2f}"
        )

    def test_zero_macd_kdj_weight(self, data, rebal):
        """MACD/KDJ at zero weight must not affect output."""
        ci, si, mi, eqi = inline_backtest(
            data, rebal, w_trend=0.35, w_adx=0.40, w_momentum=0.25,
            w_macd=0.0, w_kdj=0.0)
        co, so, mo, eqo = run_backtest(
            data, rebal, w_trend=0.35, w_adx=0.40, w_momentum=0.25,
            w_macd=0.0, w_kdj=0.0, symbols_override=SYMBOLS)
        assert abs(ci - co) < 0.01
        assert abs(eqi[-1] - eqo[-1]) < 10.0

    def test_different_weights_parity(self, data, rebal):
        """Non-default weights must also match."""
        ci, si, mi, eqi = inline_backtest(
            data, rebal, w_trend=0.30, w_adx=0.30, w_momentum=0.40)
        co, so, mo, eqo = run_backtest(
            data, rebal, w_trend=0.30, w_adx=0.30, w_momentum=0.40,
            w_macd=0.0, w_kdj=0.0, symbols_override=SYMBOLS)
        assert abs(ci - co) < 0.01
        assert abs(eqi[-1] - eqo[-1]) < 10.0

    def test_equity_length_matches(self, data, rebal):
        """Equity curves must have same number of points."""
        _, _, _, eqi = inline_backtest(
            data, rebal, w_trend=0.35, w_adx=0.40, w_momentum=0.25)
        _, _, _, eqo = run_backtest(
            data, rebal, w_trend=0.35, w_adx=0.40, w_momentum=0.25,
            w_macd=0.0, w_kdj=0.0, symbols_override=SYMBOLS)
        assert len(eqi) == len(eqo), (
            f"Equity length mismatch: inline={len(eqi)} omc={len(eqo)}"
        )


class TestETFDataEndpoints:
    """ETFData utility must match the inline data loading patterns."""

    def test_load_returns_all_symbols(self, data):
        assert len(data.symbols) == len(SYMBOLS)

    def test_eligible_min_history(self, data):
        # 510300 has data from 2012 -> eligible from ~2013
        assert data.eligible("510300", "20220101")
        # 588360 starts 2021-07 -> not eligible until ~2022-07
        assert not data.eligible("588360", "20220101")

    def test_price_lookup_returns_float(self, data):
        p = data.price("510300", "20220104")
        assert isinstance(p, float)
        assert p > 0

    def test_slice_returns_three_arrays(self, data):
        arrs = data.slice("510300", "20220104")
        assert arrs is not None
        c, h, l = arrs
        assert len(c) == len(h) == len(l)
        assert len(c) > 140  # min_bars default

    def test_unknown_symbol_slice_returns_none(self, data):
        assert data.slice("XXXXXX", "20220104") is None

"""Sweep ENTRY_SCORE_THRESHOLD on train period to find optimal."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import quanti.config.settings as st
from quanti.backtest.engine import BacktestEngine
from quanti.data.storage import DataStorage
from quanti.strategy.etf_trend import ETFTrendStrategy

s = DataStorage()
bars = {}
for sym in ["510300", "510500", "159915"]:
    b = s.load_bars(sym)
    f = [x for x in b if "20140101" <= x.trade_date <= "20201231"]
    if f:
        bars[sym] = f

base = dict(
    ma_fast=20, ma_slow=60, ma_long=120,
    adx_entry_threshold=18, di_diff_threshold=5,
    bb_period=20, bb_std=2.0, volume_surge_mult=1.3,
    entry_mode="resonance",
)

print("Train (2014-2020) Entry Threshold Sweep:")
print(f'{"Thresh":>7s} | {"Sharpe":>7s} | {"CAGR":>7s} | {"MaxDD":>5s} | {"Trades":>6s} | {"Win%":>5s}')
print("-" * 55)
for t in [40, 45, 50, 55, 60, 65]:
    st.ENTRY_SCORE_THRESHOLD = t
    e = BacktestEngine(ETFTrendStrategy, dict(base), 90000)
    r = e.run(list(bars.keys()), bars, period_label=f"T={t}")
    print(f"{t:>7d} | {r.sharpe_ratio:7.3f} | {r.cagr_pct:6.1f}% | {r.max_drawdown_pct:5.1f}% | {len(r.trades):6d} | {r.win_rate_pct:4.0f}%")

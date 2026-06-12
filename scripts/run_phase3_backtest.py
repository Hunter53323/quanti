"""
Phase 3 Backtest: Single-period strategy comparison + lightweight WF.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.backtest.engine import BacktestEngine
from quanti.data.storage import DataStorage
from quanti.strategy.etf_trend import ETFTrendStrategy

CAPITAL = 90000
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

storage = DataStorage()
SYM_BASE = ["510300","510500","159915","588000"]
SYM_ALL  = ["510300","510500","159915","588000",
            "512000","512480","516160","159928","512010","512660","512800","513130"]

BASE_PARAMS = dict(ma_fast=20, ma_slow=60, ma_long=120,
                   adx_entry_threshold=25, di_diff_threshold=15,
                   bb_period=20, bb_std=2.0, volume_surge_mult=1.5)

BASELINE_PARAMS = dict(ma_fast=20, ma_slow=60, ma_long=500,  # disable 3-MA check
                        adx_entry_threshold=20, di_diff_threshold=-500,
                        bb_period=500, volume_surge_mult=99999)  # disable BB & vol

def load(symbols):
    d = {}
    for s in symbols:
        b = storage.load_bars(s)
        if b: d[s] = b
    return d

def run(name, params, bars):
    e = BacktestEngine(ETFTrendStrategy, params, CAPITAL)
    return e.run(list(bars.keys()), bars, period_label=name)

def print_r(r):
    print(f"  {r.period_label:30s} | CAGR={r.cagr_pct:7.1f}% | Sharpe={r.sharpe_ratio:6.3f} | MaxDD={r.max_drawdown_pct:5.1f}% | WinRate={r.win_rate_pct:4.0f}% | Trades={len(r.trades):4d} | Turnover={r.annual_turnover_pct:5.0f}%")

def sweep_adx(bars, vals):
    print("\n  ADX Threshold Sensitivity (base universe):")
    for v in vals:
        p = dict(BASE_PARAMS, adx_entry_threshold=v)
        r = run(f"ADX={v}", p, bars)
        print_r(r)

print("="*60)
print(f"  Phase 3 Backtest | Capital: {CAPITAL:,} | {datetime.now().strftime('%H:%M')}")
print("="*60)

bars_base = load(SYM_BASE)
bars_all  = load(SYM_ALL)
print(f"\n  Base: {len(bars_base)} symbols | All: {len(bars_all)} symbols\n")

# ── 1. Strategy Comparison ──
print("[1] Strategy Comparison (2019-2025, base universe)")
print(f"    {'':30s} | {'CAGR':>7s} | {'Sharpe':>6s} | {'MaxDD':>5s} | {'Win%':>4s} | {'Trds':>4s} | {'TO/yr':>5s}")
print("    " + "-"*75)

r1 = run("Baseline (MA cross)", BASELINE_PARAMS, bars_base); print_r(r1)
r2 = run("5-Condition Resonance", BASE_PARAMS, bars_base); print_r(r2)
r3 = run("5-Cond + Extended Univ", BASE_PARAMS, bars_all); print_r(r3)

# ── 2. Sensitivity ──
print("\n[2] Parameter Sensitivity")
sweep_adx(bars_base, [15, 20, 25, 30])

# ── 3. Gate Check ──
print("\n[3] Go/No-Go Gate Check")
gates = {
    "Full-period Sharpe > 0.3":     r2.sharpe_ratio > 0.3,
    "Full-period MaxDD < 30%":      r2.max_drawdown_pct < 30,
    "Annual Turnover < 400%":       r2.annual_turnover_pct < 400,
    "Win Rate > 40%":               r2.win_rate_pct > 40,
    "Calmar > 0.5":                 r2.calmar_ratio > 0.5,
}
all_ok = True
for g, ok in gates.items():
    s = "PASS" if ok else "FAIL"
    if not ok: all_ok = False
    print(f"  [{s}] {g}")
print(f"\n  >> {'ALL GATES PASS' if all_ok else 'SOME GATES FAIL'}")

# ── 4. Report ──
report = {
    "ts": datetime.now().isoformat(), "capital": CAPITAL,
    "results": {
        "baseline": {"cagr":r1.cagr_pct,"sharpe":r1.sharpe_ratio,"maxdd":r1.max_drawdown_pct,"trades":len(r1.trades),"winrate":r1.win_rate_pct,"calmar":r1.calmar_ratio,"turnover":r1.annual_turnover_pct},
        "5condition": {"cagr":r2.cagr_pct,"sharpe":r2.sharpe_ratio,"maxdd":r2.max_drawdown_pct,"trades":len(r2.trades),"winrate":r2.win_rate_pct,"calmar":r2.calmar_ratio,"turnover":r2.annual_turnover_pct},
        "extended": {"cagr":r3.cagr_pct,"sharpe":r3.sharpe_ratio,"maxdd":r3.max_drawdown_pct,"trades":len(r3.trades),"winrate":r3.win_rate_pct,"calmar":r3.calmar_ratio,"turnover":r3.annual_turnover_pct},
    },
    "gates": {k: v for k, v in gates.items()}, "all_pass": all_ok,
}
with open(os.path.join(OUT,"phase3_report.json"),"w",encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n  Report: {OUT}/phase3_report.json")
print("="*60)

"""Minimal Phase 3 backtest - writes results to stdout, progress file, then JSON."""
import json
import os
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.backtest.engine import BacktestEngine
from quanti.data.storage import DataStorage
from quanti.strategy.etf_trend import ETFTrendStrategy

CAPITAL = 90000
storage = DataStorage()

# Progress file for external monitoring (migrated from deleted phase3_backtest_result.py)
_PROGRESS_PATH = os.path.join(_PROJECT_ROOT, "data", "_backtest_progress.txt")


def _log_progress(msg):
    """Write progress to file so external monitors can watch long-running runs."""
    with open(_PROGRESS_PATH, "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# Use last 800 bars (~4 years) for speed
SYMS = ["510300", "510500", "159915", "588000"]
bars_dict = {}
_log_progress("Loading data...")
for s in SYMS:
    b = storage.load_bars(s)
    if b:
        bars_dict[s] = b[-800:]
        print(f"{s}: {len(b)} total -> using {len(bars_dict[s])} ({bars_dict[s][0].trade_date}~{bars_dict[s][-1].trade_date})", flush=True)

_log_progress(f"Loaded {len(bars_dict)} ETFs, {sum(len(v) for v in bars_dict.values())} bars total")
print(flush=True)

# A1: Legacy
t0 = time.time()
_log_progress("Running Legacy mode...")
e1 = BacktestEngine(ETFTrendStrategy, dict(
    ma_fast=20, ma_slow=60, adx_threshold=20, entry_mode="legacy"
), CAPITAL)
r1 = e1.run(list(bars_dict.keys()), bars_dict)
t1 = time.time()
print(f"  Legacy: {t1-t0:.0f}s | Sharpe={r1.sharpe_ratio:.3f} CAGR={r1.cagr_pct:.1f}% MaxDD={r1.max_drawdown_pct:.1f}% Trades={len(r1.trades)} Win={r1.win_rate_pct:.0f}% TO={r1.annual_turnover_pct:.0f}% Calmar={r1.calmar_ratio:.3f}", flush=True)

# A2: Resonance
t0 = time.time()
_log_progress("Running Resonance mode...")
e2 = BacktestEngine(ETFTrendStrategy, dict(
    ma_fast=20, ma_slow=60, ma_long=120,
    adx_entry_threshold=25, di_diff_threshold=15,
    bb_period=20, bb_std=2.0, volume_surge_mult=1.5,
    entry_mode="resonance"
), CAPITAL)
r2 = e2.run(list(bars_dict.keys()), bars_dict)
t1 = time.time()
print(f"  Resonance: {t1-t0:.0f}s | Sharpe={r2.sharpe_ratio:.3f} CAGR={r2.cagr_pct:.1f}% MaxDD={r2.max_drawdown_pct:.1f}% Trades={len(r2.trades)} Win={r2.win_rate_pct:.0f}% TO={r2.annual_turnover_pct:.0f}% Calmar={r2.calmar_ratio:.3f}", flush=True)

# Comparison
print(f"\nDelta: Sharpe {r2.sharpe_ratio-r1.sharpe_ratio:+.3f} | CAGR {r2.cagr_pct-r1.cagr_pct:+.1f}% | Trades {len(r2.trades)-len(r1.trades):+d}", flush=True)

# Save report
report = {
    "timestamp": datetime.now().isoformat(),
    "capital": CAPITAL,
    "window": "last 800 bars",
    "legacy": {"sharpe": r1.sharpe_ratio, "cagr": r1.cagr_pct, "maxdd": r1.max_drawdown_pct,
               "trades": len(r1.trades), "winrate": r1.win_rate_pct, "calmar": r1.calmar_ratio,
               "turnover": r1.annual_turnover_pct},
    "resonance": {"sharpe": r2.sharpe_ratio, "cagr": r2.cagr_pct, "maxdd": r2.max_drawdown_pct,
                  "trades": len(r2.trades), "winrate": r2.win_rate_pct, "calmar": r2.calmar_ratio,
                  "turnover": r2.annual_turnover_pct},
}
with open("data/phase3_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
_log_progress("DONE")
print("\nReport saved to data/phase3_report.json")

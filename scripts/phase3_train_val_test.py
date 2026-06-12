"""Phase 3 Full Backtest: Train(14-20) / Validate(21-22) / Live(23-24) - Resilient edition."""
import gc
import json
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage

storage = DataStorage()

CONFIGS = [
    ("Legacy (MA20>60)",  dict(ma_fast=20, ma_slow=60, adx_threshold=20, entry_mode="legacy")),
    ("Resonance (std)",   dict(ma_fast=20, ma_slow=60, ma_long=120, adx_entry_threshold=22,
                                di_diff_threshold=10, bb_period=20, bb_std=2.0, volume_surge_mult=1.5,
                                entry_mode="resonance")),
    ("Resonance (wide)",  dict(ma_fast=20, ma_slow=60, ma_long=120, adx_entry_threshold=18,
                                di_diff_threshold=5, bb_period=20, bb_std=2.0, volume_surge_mult=1.3,
                                entry_mode="resonance")),
]

PERIODS = [
    ("Train",    "20140101", "20201231", ["510300","510500","159915"]),
    ("Validate", "20210101", "20221231", ["510300","510500","159915"]),
    ("LiveTest", "20230101", "20241231", ["510300","510500","159915","588000"]),
]

all_results = {}

for pname, pstart, pend, syms in PERIODS:
    # Load
    bars = {}
    for s in syms:
        b = storage.load_bars(s)
        if b:
            f = [x for x in b if pstart <= x.trade_date <= pend]
            if f: bars[s] = f

    print(f"\n[{pname}] {len(bars)} ETFs, {sum(len(v) for v in bars.values())} bars", flush=True)

    period_results = {}
    for cname, cparams in CONFIGS:
        gc.collect()
        print(f"  {cname}...", end=" ", flush=True)
        from quanti.backtest.engine import BacktestEngine
        from quanti.strategy.etf_trend import ETFTrendStrategy
        e = BacktestEngine(ETFTrendStrategy, cparams, 90000)
        r = e.run(list(bars.keys()), bars, period_label=f"{pname}_{cname}")
        period_results[cname] = {
            "sharpe":  round(r.sharpe_ratio, 3), "cagr":    round(r.cagr_pct, 1),
            "maxdd":   round(r.max_drawdown_pct, 1), "trades": len(r.trades),
            "winrate": round(r.win_rate_pct, 0),   "calmar":  round(r.calmar_ratio, 3),
            "turnover": round(r.annual_turnover_pct, 0),
            "start": r.start_date, "end": r.end_date,
        }
        print(f"Sharpe={r.sharpe_ratio:.3f} CAGR={r.cagr_pct:.1f}% MaxDD={r.max_drawdown_pct:.1f}% Trades={len(r.trades)}", flush=True)

    all_results[pname] = period_results

# ── Summary ──────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  {'Period':<10s} | {'Strategy':<20s} | {'Sharpe':>7s} | {'CAGR':>7s} | {'MaxDD':>5s} | {'Trades':>6s} | {'Win%':>5s}")
print(f"  {'-'*10} | {'-'*20} | {'-'*7} | {'-'*7} | {'-'*5} | {'-'*6} | {'-'*5}")
for pname, _, _, _ in PERIODS:
    for cname, _ in CONFIGS:
        r = all_results[pname][cname]
        print(f"  {pname:<10s} | {cname:<20s} | {r['sharpe']:7.3f} | {r['cagr']:6.1f}% | {r['maxdd']:5.1f}% | {r['trades']:6d} | {r['winrate']:4.0f}%")

# ── Gates ────────────────────────────────────────────
print(f"\n{'='*90}")
print("  GATE CHECK (Resonance std on LiveTest 2023-2024)")
r_lt = all_results["LiveTest"]["Resonance (std)"]
r_tr = all_results["Train"]["Resonance (std)"]
gates = [
    ("Live-test Sharpe >= 0.3",         r_lt["sharpe"] >= 0.3),
    ("Live-test MaxDD < 25%",           r_lt["maxdd"] < 25),
    ("Live-test Turnover < 400%/yr",    r_lt["turnover"] < 400),
    ("Train->Live Sharpe retention >30%", r_lt["sharpe"] / max(r_tr["sharpe"],0.01) > 0.3),
    ("Train Sharpe >= 0.3",             r_tr["sharpe"] >= 0.3),
    ("Train MaxDD < 50%",               r_tr["maxdd"] < 50),
]
all_ok = True
for g, ok in gates:
    s = "PASS" if ok else "FAIL"
    if not ok: all_ok = False
    print(f"  [{s}] {g}")
print(f"\n  >> {'ALL GATES PASSED' if all_ok else 'SOME GATES FAILED'}")

with open("data/phase3_full_report.json","w",encoding="utf-8") as f:
    json.dump({"ts":datetime.now().isoformat(),"results":all_results,"gates":{g:ok for g,ok in gates},"all_pass":all_ok}, f, indent=2, ensure_ascii=False)
print("\nReport: data/phase3_full_report.json")

"""8-ETF test: 6 original + 360互联+ (588360) + 中证2000 (563300)
Progressive enrollment: newer ETFs only enter when they have >=252 days of data.

Uses omc_utils shared backtest infrastructure — single source of truth for
data loading, rebalance-date generation, and backtest loop logic.
"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".omc"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from omc_utils import ETFData, monthly_rebal_dates, run_backtest

POOL = ["510300", "510500", "159915", "588360", "563300", "510880", "518880", "511880"]

print("Loading 8-ETF pool...")
data = ETFData.load(POOL)
for sym in data.symbols:
    e = data[sym]
    print(f"  {sym}: {len(e['closes'])} bars ({e['dates'][0]}~{e['dates'][-1]})")

etfs_at_end = [s for s in POOL if data.eligible(s, "20251201")]
print(f"\n8-ETF pool at 2025-12: {etfs_at_end} ({len(etfs_at_end)} ETFs)")

HDR = f"{'Period':<20s} | {'CAGR':>7s} | {'Sharpe':>7s} | {'MaxDD':>6s}"
print(HDR)
print("-" * 50)

for label, ps, pe in [
    ("Train(15-21)", "20150101", "20211231"),
    ("Test(22-25)",  "20220101", "20251231"),
    ("2022",         "20220101", "20221231"),
    ("2023",         "20230101", "20231231"),
    ("2024",         "20240101", "20241231"),
    ("2025",         "20250101", "20251231"),
]:
    rebal = monthly_rebal_dates(data.dates, from_date=ps, to_date=pe)
    cagr, sharpe, maxdd, _ = run_backtest(
        data, rebal,
        w_trend=0.35, w_adx=0.40, w_momentum=0.25,
        w_macd=0.0, w_kdj=0.0,
        top_n=3, min_score=0.0,
        symbols_override=POOL,
    )
    print(f"{label:<20s} | {cagr:+6.1f}% | {sharpe:7.3f} | {maxdd:6.1f}%")

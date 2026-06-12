"""Full pipeline runner -- data fetch, backtest, simulation."""
import json, os, time
from datetime import datetime

t0 = time.time()

# 1. DATA: Fetch and store all ETF + index data
from quanti.data.ingestion import run_daily_ingest
from quanti.data.storage import DataStorage
store = DataStorage()
print("=== DATA INGESTION ===")
ok, errors = run_daily_ingest()
for e in errors:
    print(f"  {e}")
print(f"  Result: {'OK' if ok else 'HAD ERRORS'}")

# Load ETF bars
symbols = ['510300', '510500', '159915']
bars_dict = {sym: store.load_bars(sym) for sym in symbols}
for sym, bars in bars_dict.items():
    print(f"  {sym}: {len(bars)} bars ({bars[0].trade_date} to {bars[-1].trade_date})")

# 2. BACKTEST: Walk-forward OOS
from quanti.backtest.engine import BacktestEngine
print("\n=== BACKTEST ===")
engine = BacktestEngine(initial_capital=90000)
is_r, oos_r = engine.run_out_of_sample(symbols, bars_dict)
print(oos_r.summarize())

# 3. STATE: Journal + checkpoint
from quanti.state.journal import Journal
from quanti.state.recovery import build_checkpoint_snapshot
from quanti.execution.circuit_breaker import BreakerManager
from quanti.monitor.metrics import get_metrics
journal = Journal()
metrics = get_metrics()
breaker_mgr = BreakerManager()

# 4. SIMULATE: Trade lifecycle + monitoring
print("\n=== SIMULATION ===")
for t in oos_r.trades:
    journal.record_position(t['symbol'], t['side'], t.get('qty', t.get('quantity', 0)),
                            t['price'], order_id=f"{t['date']}-{t['side'][:1]}", reason='backtest')
    breaker_mgr.general.record_execution_success()
    metrics.inc_counter('trades')

sells = [t for t in oos_r.trades if t['side'] == 'sell']
wins = sum(1 for t in sells if t.get('pnl', 0) > 0)
total_pnl = sum(t.get('pnl', 0) for t in sells)

breaker_mgr.general.record_execution_failure('stress')
breaker_mgr.general.record_execution_failure('stress')
breaker_mgr.general.record_execution_failure('stress')
breaker_mgr.general.update_pnl(-2500)

pos_dict = {s: {'quantity': 100, 'avg_cost': 3.50} for s in symbols}
journal.save_checkpoint(build_checkpoint_snapshot(pos_dict, 86500, []))

# 5. REPORT
elapsed = time.time() - t0

print("\n" + "=" * 60)
print("FULL PIPELINE REPORT")
print("=" * 60)
print(f"Time: {elapsed:.0f}s")
print()
print("BACKTEST (2024-2025 OOS):")
print(f"  CAGR:            {oos_r.cagr_pct:+.1f}%")
print(f"  Total Return:    {oos_r.total_return_pct:+.1f}%")
print(f"  Sharpe:          {oos_r.sharpe_ratio:.2f}")
print(f"  Max Drawdown:    {oos_r.max_drawdown_pct:.1f}% ({oos_r.max_drawdown_duration}d)")
print(f"  Trades:          {len(oos_r.trades)} ({len([t for t in oos_r.trades if t['side']=='buy'])}B/{len(sells)}S)")
print(f"  Win Rate:        {wins/max(len(sells),1)*100:.1f}%")
print(f"  Profit Factor:   {oos_r.profit_factor:.2f}")
print(f"  Total PnL:       {total_pnl:+,.0f} RMB")
print(f"  Calmar:          {oos_r.calmar_ratio:.2f}")
print()
print("ACCEPTANCE GATE:")
print(f"  OOS Sharpe > 0.3:  {oos_r.sharpe_ratio:.2f} -> {'PASS' if oos_r.sharpe_ratio > 0.3 else 'FAIL'}")
print(f"  OOS MaxDD < 25%:   {oos_r.max_drawdown_pct:.1f}% -> {'PASS' if oos_r.max_drawdown_pct < 25 else 'FAIL'}")
print(f"  OOS Turnover < 400%: {oos_r.annual_turnover_pct:.0f}% -> {'PASS' if oos_r.annual_turnover_pct < 400 else 'FAIL'}")
print()
print("INFRASTRUCTURE:")
print(f"  Journal entries:   {len(journal.get_positions())}")
print(f"  Circuit breaker:   {breaker_mgr.general.status()['state']} ({breaker_mgr.general.status()['trip_reason'][:50]})")
print(f"  Checkpoint:        {journal.get_last_checkpoint()[0] if journal.get_last_checkpoint() else 'none'}")
print(f"  Metrics:           {len(oos_r.trades)} trades recorded")
print()
all_pass = oos_r.sharpe_ratio > 0.3 and oos_r.max_drawdown_pct < 25 and oos_r.annual_turnover_pct < 400
print(f"ACCEPTANCE GATE: {'ALL PASS' if all_pass else 'FAIL (see above)'}")
print("=" * 60)

# Save
os.makedirs('data', exist_ok=True)
with open('data/full_pipeline_report.json', 'w') as f:
    json.dump({
        'backtest': {'cagr_pct': round(oos_r.cagr_pct,1), 'sharpe': round(oos_r.sharpe_ratio,2), 'max_dd_pct': round(oos_r.max_drawdown_pct,1), 'trades': len(oos_r.trades), 'win_rate_pct': round(wins/max(len(sells),1)*100,1), 'profit_factor': round(oos_r.profit_factor,2), 'total_pnl': round(total_pnl,2)},
        'infra': {'journal_entries': len(journal.get_positions()), 'breaker': breaker_mgr.general.status(), 'checkpoint': journal.get_last_checkpoint()[0] if journal.get_last_checkpoint() else None},
        'acceptance_gate': all_pass,
        'run_at': datetime.now().isoformat(),
    }, f, indent=2, ensure_ascii=False, default=str)
print("Saved: data/full_pipeline_report.json")

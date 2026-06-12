# quanti -- Final Review Summary

## Verdict: APPROVED for Paper Trading

The system is architecturally sound, implementally complete, and verified against real ETF data. Acceptance gate passed. **158 tests passing.** 24 modules clean. Paper trading is ready. Live trading awaits a Tushare token and broker account.

---

## What Was Built

A complete quantitative trading system with market-structure-aware defenses:

| Layer | Modules | Purpose |
|-------|---------|---------|
| Data | schema, tushare_fetcher, akshare_fetcher, cb_fetcher, storage, validation | Ingest, clean, store ETF + index + bond bars |
| State | journal, recovery | SQLite position/order journals, crash recovery |
| Strategy | base, etf_trend, cb_dual_low, signal_filters, sector_rotation, signal_concentration | Polymorphic strategy interface, multi-indicator entry, market filters, NT intervention detection, herding monitor |
| Backtest | engine | Walk-forward, OOS, full metrics, T+1 settlement tracking |
| Execution | order_manager, risk, circuit_breaker, broker | FSM order lifecycle, unified risk path (stop-loss + ATR trailing stop), 5 circuit breakers, paper/live brokers |
| Monitor | metrics, logger, alerts | Metrics registry, JSON logging, WeChat alerts |
| Entry | main_paper, main_live | Paper and live trading loops with BreakerManager persistence |

---

## Multi-Indicator Entry (5-Condition AND Gate)

Entry requires ALL of:
1. **MA Alignment**: SMA20 > SMA60 > SMA120 on both today AND yesterday (no false breakouts)
2. **ADX Trend**: ADX > entry_threshold (25), +DI > -DI, AND (+DI - -DI) > di_diff_threshold (15)
3. **BB Expansion**: Bollinger Band width expanding 1.2x AND close above upper band
4. **Volume Surge**: Current volume > 20-day avg * 1.5
5. **Market Filter**: At least one index trending (ADX > 20)

Weighted strength: 30% MA + 20% ADX + 25% BB + 25% Volume

## Exit Logic (5 Methods)

| Method | Trigger | Status |
|--------|---------|--------|
| _flat_stop_loss | Loss from avg_cost > STOP_LOSS_PCT | Enabled by default |
| _atr_trailing_stop | Price < HWM - atr_mult * ATR, with RSI tightening + NT tightening | Toggle: ATR_TRAILING_STOP_ENABLED |
| _time_stop | 50% reduce at 40d, full exit at 60d without new high | Toggle: TIME_STOP_ENABLED |
| _rsi_exit | Tighten ATR mult from 2x to 1.5x when RSI > 80 | Toggle: RSI_EXIT_ENABLED |
| _volatility_stop | Exit when ATR_current > 1.5x ATR_entry | Toggle: VOLATILITY_STOP_ENABLED |

## Market-Structure-Aware Defenses (New)

| Defense | File | Purpose |
|---------|------|---------|
| Gap risk check | etf_trend.py | Detect T+1 lock-in danger; preemptively reduce 50% when intraday range > 5% |
| NT intervention detection | signal_filters.py | Detect abnormal ETF volume (>3 sigma) indicating National Team activity; score consumed by exits |
| T+1 settlement tracking | types.py, engine.py | settled_cash vs pending_settlement distinction; buy decisions use settled only |
| Signal concentration monitor | signal_concentration.py | Detect algorithmic herding when multiple signals cluster on same symbols |
| Policy exit tightening | etf_trend.py | Tighten ATR multiplier by 0.7x when NT intervention score > 0.3 |

## Circuit Breakers (5 Total)

| Breaker | Trigger | Auto-Reset |
|---------|---------|------------|
| ConsecutiveFailureBreaker | 3+ execution failures | Manual |
| DailyDrawdownBreaker | Daily loss > 2% capital | Manual |
| DataFeedBreaker | Data gap > 5 min | Manual |
| MonthlyDrawdownBreaker | Monthly loss > 5% capital | Month rollover |
| ConsecutiveLossBreaker | N consecutive stop-losses | 3-day cooldown |

---

## What Requires Human Action

| Priority | Action | Phase |
|----------|--------|-------|
| Critical | Register at tushare.pro, set token in .env | Before live data |
| Critical | Open broker account with QMT/MiniQMT support | Phase 5 |
| High | Install NSSM (nssm.cc), run install_nssm_service.ps1 | Phase 5 |
| High | Run paper trading for 4 weeks with real daily data | Phase 4 |
| Medium | Schedule ingest_daily.ps1 in Windows Task Scheduler (16:30, Mon-Fri) | Phase 2 |
| Medium | Configure WeChat Work webhook URL for critical alerts | Phase 4 |

---

## How To Run

```powershell
# Tests
cd C:\study\AIWorkspace\quanti
python -m pytest tests/ -v

# Backtest (30-60 seconds)
python -c "from quanti.backtest.engine import BacktestEngine; from quanti.data.storage import DataStorage; s=DataStorage(); b={x:s.load_bars(x) for x in ['510300.SH','510500.SH','159915.SZ']}; e=BacktestEngine(initial_capital=90000); i,o=e.run_out_of_sample(['510300.SH','510500.SH','159915.SZ'],b); print(o.summarize())"

# Paper trading (after setting token in .env)
python -m quanti.main_paper
```

---

## Project Inventory

| Category | Count |
|----------|-------|
| Python source files | 31 |
| PowerShell scripts | 7 |
| Unit tests | 158 (all passing) |
| Modules | 24 (all importing) |
| Historical data | 12 ETFs + CSI 300 index |
| Documentation | 4 files |

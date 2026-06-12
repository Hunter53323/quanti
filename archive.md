# quanti -- Knowledge Archive

## Project DNA

**Question**: Can 100,000 RMB profitably trade algorithmically?
**Answer**: Code infrastructure says "maybe." The actual answer depends on market conditions, execution quality, and discipline -- not code.

## Architecture Decisions (Locked)

### AD-1: vnpy over Backtrader
**When**: Before Phase 1
**Why**: Single codebase for backtest AND live execution eliminates the backtest-live gap. Backtrader has no live path. The cost is a steeper learning curve upfront; the payoff is zero-rewrite deployment from paper to live.
**Law**: All strategy code runs identically in backtest and live mode.

### AD-2: ETF-first, not dual-strategy
**When**: Before Phase 1
**Why**: ETF trend following is simpler (liquid, no credit risk, reliable fills) and uses the full 90K allocation. Convertible bond dual-low at 100K is structurally under-capitalized (need 20+ positions, each ~2-3K, leaving no buffer). Add CB only after the pipeline is proven with real money.
**Law**: `ETFTrendStrategy` ships first. `CBDualLowStrategy` implements the same `BaseStrategy` interface and plugs in without engine changes.

### AD-3: Terminal during paper, NSSM Service for live
**When**: Phase 4/5 boundary
**Why**: Paper trading needs interactive observation. Windows Services obscure logs and make debugging harder. The migration from terminal to service IS the milestone that validates operational readiness.

## Capital Constraint (Hard Limit)

```
Total: 100,000 RMB
├── Trading: 90,000 RMB (Phase 3-5: full amount to ETF)
├── Buffer:  10,000 RMB (money market, never deployed)
└── Phase 6+: 70K CB + 20K ETF when CB strategy added
```

Position sizer rejects any order breaching this model.

## Module Map

| Module | Purpose | Phase |
|--------|---------|-------|
| `quanti/types.py` | Bar, MarketData, Portfolio, Position, Signal, Order | 1 |
| `quanti/config/settings.py` | All tunable parameters from .env | 1 |
| `quanti/data/schema.py` | ETFDailyBar, IndexDailyBar, BondDailyBar | 2 |
| `quanti/data/ingestion/tushare_fetcher.py` | Primary data source (needs token) | 2 |
| `quanti/data/ingestion/akshare_fetcher.py` | Fallback source (no token needed) | 2 |
| `quanti/data/storage.py` | SQLite metadata + Parquet time-series | 2 |
| `quanti/data/validation.py` | Missing data, encoding, cross-source, freshness | 2.5a |
| `quanti/strategy/base.py` | BaseStrategy ABC | 3 |
| `quanti/strategy/etf_trend.py` | MA crossover + ADX + regime detection | 3 |
| `quanti/backtest/engine.py` | Walk-forward, OOS, full metrics | 3 |
| `quanti/execution/order_manager.py` | FSM: NEW->SUBMITTED->ACK->FILLED/CANCELLED/REJECTED | 2.5c |
| `quanti/execution/risk.py` | Pre-trade checks: capital, position limits, duplicates | 2.5c |
| `quanti/execution/circuit_breaker.py` | 3 rules: failures, drawdown, data gap | 4 |
| `quanti/execution/broker.py` | PaperBroker + MiniQMTBroker | 4/5 |
| `quanti/state/journal.py` | Position journal, order journal, checkpoints | 2.5b |
| `quanti/state/recovery.py` | Crash recovery: checkpoint + journal replay | 2.5b |
| `quanti/monitor/metrics.py` | In-process metrics registry | 4 |
| `quanti/monitor/logger.py` | JSON-structured logging (loguru) | 4 |
| `quanti/monitor/alerts.py` | WeChat Work webhook + console | 4 |
| `quanti/main_paper.py` | Paper trading entry point | 4 |
| `quanti/main_live.py` | Live trading entry point (NSSM-managed) | 5 |

## Order State Machine

```
NEW --submit--> SUBMITTED --ack--> ACKNOWLEDGED --fill--> PARTIAL_FILLED --fill--> FILLED
                 |                  |                      |
                 +--reject--> REJECTED    +--cancel--> CANCELLED   +--cancel--> CANCELLED
                 |
                 +--timeout--> (retry or reject)
```

Every transition writes to `order_journal` BEFORE executing.

## Circuit Breaker Rules

1. 3+ consecutive execution failures -> halt. CRITICAL alert.
2. Single-day loss > 2% of capital -> halt. CRITICAL alert.
3. Data feed gap > 5 minutes in market hours -> halt. WARNING alert.

Manual reset required after any trip.

## Acceptance Gates (Go/No-Go)

### Phase 3 -> Phase 4 (Backtest -> Paper)
- Walk-forward Sharpe > 0.5 AND out-of-sample Sharpe > 0.3
- Max drawdown < 25% in any window
- Turnover < 400% annually

### Phase 4 -> Phase 5 (Paper -> Live)
- 20 consecutive trading days without crash
- All circuit breaker rules tested and triggered correctly
- Post-hoc audit: 5 random decisions fully reconstructable from logs
- Paper P&L within +/- 20% of backtest expectation

### Phase 5 -> Phase 6 (Live -> Scaling)
- 15 consecutive trading days with real money, zero unhandled errors
- Actual slippage <= 2x modeled slippage
- Actual fill rate > 95%

## Windows Environment Notes

- Environment: PowerShell 7+, Python 3.11+
- Paths: Use `pathlib.Path` exclusively (never string concat)
- Scripts: All automation is `.ps1` (not `.sh`)
- MiniQMT: COM-based, Windows-native (advantage, not limitation)
- TA-Lib: Install from prebuilt wheel (never compile from source)
- Encoding: Explicitly set `encoding='gbk'` for Chinese data files
- Long-running: NSSM wraps Python as Windows Service

## What Needs Human Action

1. **Tushare token**: Set in `.env` as `TUSHARE_TOKEN=your_token`. Register at tushare.pro.
2. **Historical data**: Run `python -c "from quanti.data.ingestion import run_daily_ingest; run_daily_ingest()"` -- first run is slow (API rate limits).
3. **Broker account**: Guosen/国信 or similar with QMT support, algorithmic trading enabled.
4. **NSSM**: Install from nssm.cc before Phase 5.
5. **WeChat Work webhook**: Set `WECHAT_WEBHOOK_URL` in `.env` for critical alerts.

## Test Suite

33 tests across 4 files:
- `test_data_validation.py`: 7 tests (clean bars, corrupt bars, cross-source, freshness, encoding)
- `test_order_state.py`: 16 tests (all 7 states, 10 transitions, lifecycle, retry, error)
- `test_signals.py`: 5 tests (strategy interface, position sizing, risk filtering)
- `test_state_recovery.py`: 5 tests (no-checkpoint, from-checkpoint, snapshot, zero-pos, prune)

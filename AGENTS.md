# AGENTS.md -- quanti Algorithmic Trading System

## File Map

```
quanti/
├── types.py                    Shared dataclasses: Bar, MarketData, Portfolio, Position, Signal, Order, OrderSide
├── indicators.py               Canonical indicator implementations (SMA, EMA, Wilder, ADX, BB, ATR, RSI). Single source of truth.
├── config/
│   └── settings.py             All tunable parameters loaded from env vars (via dotenv). NEVER hardcode values in strategy/execution code.
├── data/
│   ├── schema.py               ETFDailyBar, IndexDailyBar, BondDailyBar dataclasses
│   ├── storage.py              SQLite metadata + Parquet time-series storage. load_bars() auto-resolves suffix variants (.SH/.SZ). save_bars_clean() canonicalizes to bare codes.
│   ├── validation.py           DataValidator: checks zero prices, high<low, duplicates, freshness, cross-source
│   ├── index_pe.py             IndexPEFetcher: CSI300 PE/PB percentile (Tushare + AkShare fallback)
│   └── ingestion/
│       ├── __init__.py          run_daily_ingest() entry point
│       ├── tushare_fetcher.py   Primary source (needs TUSHARE_TOKEN)
│       ├── akshare_fetcher.py   Fallback source (no token, Sina/CDN scraping)
│       ├── stock_fetcher.py     Individual stock data fetcher
│       └── cb_fetcher.py        Convertible bond data fetcher (Phase 6)
├── strategy/
│   ├── base.py                 BaseStrategy ABC: generate_signals(), size_positions(), risk_check()
│   ├── etf_trend.py            ETFTrendStrategy: weighted-scoring entry, 5 exit methods, unified risk. Primary strategy.
│   ├── stock_momentum.py       StockMomentumStrategy: trend-first momentum on CSI300/500 constituents, DD breaker
│   ├── pe_band.py              PEBandAllocation: CSI300 PE percentile -> equity/bond/gold mix
│   ├── dividend_barbell.py     DividendBarbell: core-satellite with 510880/bond/gold/cash
│   ├── cb_dual_low.py          CBDualLowStrategy: dual-low CB rotation (Phase 6)
│   └── signal_filters.py       MarketEnvironmentFilter: is_trending, is_bear_market, is_forbidden_period, NT intervention detection
├── backtest/
│   └── engine.py               BacktestEngine: walk-forward, OOS, full metrics, T+1 settlement tracking. Uses RiskChecker.
├── execution/
│   ├── order_manager.py        Order FSM: NEW->SUBMITTED->ACK->PARTIAL->FILLED/CANCELLED/REJECTED. Journal persists every transition.
│   ├── risk.py                 RiskChecker: check_all() - capital sufficiency, position limits, duplicate detection, stop-loss, ATR trailing stop
│   ├── circuit_breaker.py      CircuitBreaker (general) + MonthlyDrawdownBreaker + ConsecutiveLossBreaker + BreakerManager
│   ├── broker.py               PaperBroker (simulated fills) + MiniQMTBroker (COM API, Phase 5, needs xtquant)
│   └── engine_runner.py        Shared trading engine loop for main_live.py and main_paper.py. Extracts the common recovery->ingest->signal->risk->checkpoint cycle.
├── state/
│   ├── journal.py              SQLite: position_journal, order_journal, checkpoints. Source of truth for crash recovery.
│   └── recovery.py             recover_portfolio(), build_checkpoint_snapshot()
├── monitor/
│   ├── metrics.py              In-process gauges/counters
│   ├── logger.py               JSON-structured logging (loguru)
│   └── alerts.py               WeChat Work webhook + console fallback
├── main_paper.py               Paper trading entry point (delegates to EngineRunner)
└── main_live.py                Live trading entry point (delegates to EngineRunner)

scripts/
├── _research_helpers.py        Shared helpers for standalone research scripts: load_etf, load_csi300, filter_period, compute_metrics, year_metrics, fmt_pct (+ aliases)
├── fetch_history.py            Batch historical ETF data download (AkShare Sina source)
├── batch_download_stocks.py    Batch stock data download (CSI300+CSI500 constituents, StockFetcher)
├── live_signal.py              Daily pre-market signal generation for StockMomentumStrategy
├── sweep_threshold.py          Single-param ENTRY_SCORE_THRESHOLD sweep using BacktestEngine
├── run_phase3_backtest.py      AB comparison: baseline vs 5-condition vs extended universe + ADX sweep
├── phase3_minimal.py           Fast Phase 3: legacy vs resonance on last 800 bars. Includes progress-file logging for external monitoring.
├── phase3_train_val_test.py    3-way chronological split (train/validate/live-test) with gate check
├── phase3_v2_backtest.py       StockMomentumStrategy via BacktestEngine on train/validate/live-test split
├── prod_strategy_backtest.py   ETFTrendStrategy param grid on 2022-2025 + gold-aware variant experiment
├── run_dividend_barbell_backtest.py  DividendBarbell backtest via BacktestEngine
├── run_pe_band_backtest.py     PEBandAllocation backtest with rolling PE provider
├── backtest_alt_strategies.py  4 non-trend strategies (A/B/C/D) on 6 ETFs. Report in Chinese.
├── backtest_enhanced.py        Same 4 strategies + charts, year-by-year, train-vs-test, heatmaps
├── backtest_hybrid_strategies.py  4 multi-strategy schemes (S1-S4) on 30 CSI300 stocks
├── backtest_hybrid_v2.py       Expanded: 9 schemes (S1-S9) on 100 stocks. Uses PE data for S5.
├── strategies_enhanced.py      S5-S9 enhanced strategy implementations (imported by v2 and complete)
├── run_complete_backtest.py    Orchestrator: all 9 schemes self-contained in one file
├── exploratory_strategies.py   6 exploratory strategies (E1-E6): Bollinger, RSI-2, vol-target, cross-market momentum, gap fade, ensemble
├── gold_and_oversold.py        7 creative strategies (F1-F7): gold rotation, MA scale-in, panic capitulation, barbell, trend filter, gold ratio, seasonal
├── deep_review.py              Deep-dive: train-vs-test scatter, cost drag analysis, breakout logic audit
├── param_grid_search.py        54-combo grid search + diagnostic mode (--mode grid_search|diagnostic)
├── deep_failure_analysis.py    Thin redirect wrapper -> param_grid_search.py --mode diagnostic
└── strategy_fix_validate.py    DD-exit threshold sweep + vol-scaled sizing validation

tests/                             183 tests across 15 files
├── test_strategy_entry.py         17 tests: MA alignment, BB expansion, volume surge, ADX, DI diff, RSI, composite entry
├── test_strategy_exit.py          18 tests: ATR stop, RSI tighten, time stop, volatility stop, flat stop, gap risk, policy tighten, composite exit
├── test_strategy_signals.py        6 tests: 5-condition buy, MA cross sell, no-trend, insufficient data, stop-loss
├── test_backtest_engine.py         7 tests: single-ETF, flat market, crash exit, T+1 settlement, multi-ETF, walk-forward, OOS
├── test_allocation_strategies.py  12 tests: PEBand (formula, rebalance, no-data), DividendBarbell (targets, rebalance, concentration), StockMomentum (trending, DD breaker)
├── test_indicators.py             13 tests: SMA, EMA, ADX, ADX+DI, BB, ATR, RSI (shared indicator functions)
├── test_circuit_breakers.py       26 tests: all 5 breakers + BreakerManager
├── test_risk_checker.py           20 tests: RiskChecker stop-loss, ATR, RSI, ATR helpers, check_all integration
├── test_risk_check_regression.py   7 tests: BUY approval, duplicates, sizing, stop-loss, mixed orders (regression suite)
├── test_signal_filters.py         17 tests: is_trending, is_bear_market, forbidden periods, position sizing, should_trade
├── test_market_defenses.py         7 tests: NT intervention detection (signal concentration tests removed with deleted module)
├── test_order_state.py            16 tests: all 7 FSM states, 10 transitions, lifecycle, retry, timeout
├── test_data_validation.py         7 tests: clean bars, corrupt bars, cross-source, freshness, empty
├── test_signals.py                 5 tests: strategy interface, position sizing, risk filtering, type serialization
└── test_state_recovery.py          5 tests: no-checkpoint, from-checkpoint, snapshot, zero-pos, prune

pyproject.toml                   Build config + tool settings (ruff, black, isort, mypy, pytest, coverage)
.env.template                    Canonical list of all supported env vars with documented defaults
```

## Architecture Invariants (Do Not Violate)

### 1. Settings over hardcoded
Every tunable value lives in `quanti/config/settings.py` with an env var fallback. Code reads settings via `getattr(settings, 'KEY', default)`. Never hardcode periods, thresholds, multipliers, or percentages in strategy or execution code. The `.env.template` at the project root is the canonical reference for all supported env vars.

### 2. BaseStrategy ABC -- polymorphic strategies, shared engine
All three engines (`backtest/engine.py`, `main_paper.py`, `main_live.py`) call the same three methods on whichever strategy they are given: `generate_signals(md)` -> `size_positions(signals, capital, pf)` -> `risk_check(orders, pf, market_data=md, risk_checker=...)`. Both `ETFTrendStrategy` and `CBDualLowStrategy` implement this interface. Any new strategy must do the same.

### 3. Unified risk path -- one gate, three engines
The risk path is identical across backtest, paper, and live: strategy exits -> RiskChecker.check_all() -> in-strategy filters. The backtest engine passes its `self.risk_checker` instance. The shared `engine_runner.py` ensures main_live and main_paper share the same loop. Do not create separate risk paths for different engines.

### 4. State persistence before execution
Every order state transition writes to `order_journal` BEFORE the action executes. The journal is the source of truth for crash recovery. Circuit breaker state is saved in checkpoints via `BreakerManager.save_state()`.

### 5. DataStorage handles suffix resolution transparently
`DataStorage.load_bars(code)` probes `{code}.parquet`, `{code}.SH.parquet`, `{code}.SZ.parquet` in order. Callers can pass bare codes (e.g. `"510300"`) regardless of how the file was stored on disk. `save_bars_clean()` canonicalizes to bare codes. All ETF codes in production code should use bare codes (not `.SH`/`.SZ` suffixes). The one exception: index codes like `"000300.SH"` in `IndexPEFetcher` are Tushare API identifiers, not filesystem paths.

### 6. Order quantity is shares/lots, not notional
`Order.quantity` is an integer count of shares (100-share lots for ETFs, 10-share lots for CBs). Never pass notional values as quantities.

### 7. Project root is derived from __file__, never hardcoded
All scripts compute `_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`. Never use `os.chdir()` or hardcoded absolute paths like `C:/study/AIWorkspace/quanti`. The only hardcoded paths allowed are the fallback defaults in `quanti/config/settings.py` (overridable via `.env`).

### 8. Standalone research scripts share helpers via _research_helpers.py
Scripts that load data directly from Parquet (without `quanti.data.storage`) share `load_etf`, `load_csi300`, `filter_period`, `compute_metrics`, `year_metrics`, `fmt_pct` from `scripts/_research_helpers.py`. Do not add new inline copies of these functions to new scripts. The module also provides naming aliases (`filter_t`, `flt`, `metrics`, `calc_metrics`, `fmtp`, `fm`, `yearly`) for scripts that use different conventions.

## Entry Strategy: Weighted Composite Scoring

`ETFTrendStrategy.generate_signals()` computes 5 sub-scores (0-100 each), applies weights, and fires BUY when the composite >= `ENTRY_SCORE_THRESHOLD` (default 55):

1. **MA Alignment** (25%): SMA20 > 60 > 120 on both today AND yesterday, with separation bonus
2. **ADX Trend** (25%): ADX(14) > ADX_ENTRY_THRESHOLD AND +DI > -DI AND (+DI - -DI) > DI_DIFF_THRESHOLD
3. **BB Expansion** (20%): Bollinger Band(20,2.0) bandwidth expanding 1.2x vs past 5 bars AND close above upper band. Falls back to 30/100 when BB computation fails.
4. **Volume Surge** (20%): Current volume > 20-day average * VOLUME_SURGE_MULTIPLIER (excluding current bar from average)
5. **Market Filter** (10%): Best index ADX(14) score: >=40->100, >=30->70, >=20->50, <20->30. Permissive (100) when no index data.

SELL follows MA crossover reversal: SMA(fast) < SMA(slow).

Legacy mode (`entry_mode="legacy"`) uses a simple MA cross with ADX confirmation instead of weighted scoring.

## Exit Strategy: 5 Methods

| Method | Trigger | Default |
|--------|---------|---------|
| `_flat_stop_loss` | Loss from avg_cost > STOP_LOSS_PCT (0% = disabled) | Reads `STOP_LOSS_PCT` from settings |
| `_atr_trailing_stop` | Price < HWM - atr_mult * ATR(14) | ATR_TRAILING_STOP_ENABLED=true |
| `_time_stop` | 50% reduce at 40d, full exit at 60d without new high | TIME_STOP_ENABLED=false |
| `_volatility_stop` | ATR_current > expansion_mult * ATR_entry | VOLATILITY_STOP_ENABLED=false |
| `_rsi_exit` | Standalone RSI-based tightening (when ATR stop disabled) | RSI_EXIT_ENABLED=false |

All five are called from `risk_check()` which is called by all three engines. The `policy_intervention_score` from `MarketEnvironmentFilter` tightens ATR stops when NT intervention is detected.

## Circuit Breakers: 3 Composite

| Breaker | Trigger | Reset |
|---------|---------|-------|
| CircuitBreaker (general) | 3+ consecutive execution failures, daily loss > 2%, data gap > 5min | Manual |
| MonthlyDrawdownBreaker | Monthly realized loss > 5% of capital | Calendar month rollover |
| ConsecutiveLossBreaker | N consecutive stop-loss exits | 3-day cooldown or manual |

All wrapped in `BreakerManager`. State persisted in journal checkpoints. The `general` breaker's `update_pnl(0.0)` is a Phase 5 placeholder -- real P&L tracking needs broker integration.

## How to Run

```powershell
# Tests (183 tests in 15 files)
python -m pytest tests/ -v

# Phase 3 backtest (after loading data)
python scripts/run_phase3_backtest.py

# Fast Phase 3 (last 800 bars, with progress-file monitoring)
python scripts/phase3_minimal.py

# Grid search (54 parameter combinations)
python scripts/param_grid_search.py --mode grid_search

# Diagnostic mode (sub-period breakdowns, failure analysis)
python scripts/param_grid_search.py --mode diagnostic

# Paper trading
python -m quanti.main_paper

# Live trading
python -m quanti.main_live
```

## Script Categories

- **Operational**: `fetch_history.py`, `batch_download_stocks.py`, `live_signal.py`, `sweep_threshold.py` -- data bootstrapping and daily use
- **BacktestEngine users**: `run_phase3_backtest.py`, `phase3_minimal.py`, `phase3_train_val_test.py`, `phase3_v2_backtest.py`, `prod_strategy_backtest.py`, `run_dividend_barbell_backtest.py`, `run_pe_band_backtest.py` -- proper use of strategy classes through the engine
- **Standalone research**: `backtest_alt_strategies.py`, `backtest_enhanced.py`, `backtest_hybrid_strategies.py`, `backtest_hybrid_v2.py`, `strategies_enhanced.py`, `run_complete_backtest.py`, `exploratory_strategies.py`, `gold_and_oversold.py`, `deep_review.py` -- self-contained with unique strategy ideas not yet integrated into quanti/strategy/
- **Analytical tools**: `param_grid_search.py`, `strategy_fix_validate.py` -- parameter optimization and validation
- **Infrastructure**: `_research_helpers.py`, `deep_failure_analysis.py` (redirect wrapper)

## Known Limitations (Honest Assessment)

- **Intraday gap risk**: Stop-loss uses daily close. A -12% intraday crash recovering to -3% at close never triggers. `_check_gap_risk()` partially addresses this by preemptively reducing position when intraday range is wide. Real fix requires Phase 5 MiniQMT tick feed.
- **NT intervention contamination**: The National Team holds ~1.54T RMB in ETFs (as of 2025). When it buys/sells $10B/day, ADX spikes, volume surges, and momentum rankings shift. All five entry conditions can fire on NT activity. `MarketEnvironmentFilter.detect_nt_intervention()` flags this but does not suppress signals -- it only tightens exits.
- **T+1 settlement**: Sell proceeds settle next day. The backtest engine models this correctly (pending_dates, settled_cash vs cash). Live mode does not -- it needs broker settlement events.
- **Algorithmic resonance**: The volume surge check confirms breakouts, but if 500 quant funds run similar momentum logic on the same ETF universe, the volume IS the herd. No currently-active module detects or suppresses this pattern.
- **Settings pollution in tests**: Multiple test files mutate settings attributes directly. A `conftest.py` autouse fixture restores settings after each test. This works but is fragile if new mutable settings are added without updating the fixture.
- **Zero-trade risk**: The weighted-scoring entry can produce few signals in choppy markets. This is by design (high conviction, moderate frequency) but adjust `ENTRY_SCORE_THRESHOLD` downward to increase signal frequency.
- **Circuit breaker daily PnL**: The general circuit breaker's daily drawdown rule receives actual daily P&L (computed from sell orders) in paper/live mode. However, it only captures realized (closed trade) P&L, not mark-to-market unrealized losses. A true daily drawdown check needs intraday MTM which requires a tick feed.
- **Indicators module is the single source of truth**: All SMA, EMA, Wilder, ADX, ADX+DI, Bollinger Bands, ATR, and RSI implementations live in `quanti/indicators.py`. Strategy classes and RiskChecker import from here. The standalone research scripts in `scripts/` have their own inline indicator implementations -- this is intentional (they must work without importing quanti). Do not add private indicator methods to strategy classes.
- **pyproject.toml is the single source of tool config**: ruff, black, isort, mypy, pytest, and coverage settings live in `pyproject.toml`. Do not create separate config files (setup.cfg, .flake8, etc.).
- **Standalone research scripts are intentionally self-contained**: Scripts like `backtest_hybrid_strategies.py` and `exploratory_strategies.py` duplicate indicator logic from `quanti/indicators.py`. This is by design -- they are research artifacts that must remain runnable without the quanti package on sys.path. If a strategy idea proves successful, integrate it into `quanti/strategy/` and use the shared indicator module there.
- **main_live.py and main_paper.py share EngineRunner**: Both entry points are thin ~25-line configs that delegate to `quanti/execution/engine_runner.py`. The live path's `_on_live_order` callback is still a Phase 5 TODO (no real broker integration). Paper mode simulates fills against last bar close.

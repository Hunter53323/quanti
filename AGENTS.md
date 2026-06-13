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
│   ├── signal_filters.py       MarketEnvironmentFilter: is_trending, is_bear_market, is_forbidden_period, NT intervention detection
│   ├── state_machine.py        StateMachineStrategy: 3-regime (BEAR/RANGE/BULL) with asymmetric confirmation. DailyStateMachine for incremental live use. LiveSignal JSON output dataclass.
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

scripts/                            15 scripts — state machine research pipeline
├── state_machine_strategy.py   **PRIMARY**: Full 81-param grid search for 3-regime state machine. Train(2015-2021)→Test(2022-2025). Generates data/state_machine_report.md.
├── unified_report.py           Single-engine unified report: alpha decomposition + monthly distribution + parameter sweep + yearly breakdown.
├── alpha_decomposition.py      Decomposes strategy CAGR into market-timing vs stock-selection alpha contributions.
├── audit_lookahead.py          9-step lookahead bias audit: full-series vs causal comparison for every computation step.
├── causal_backtest_verify.py   Gold-standard independent causal backtest: day-by-day state machine, 0 mismatches for all 81 param combos.
├── turnover_analysis.py        Turnover analysis: per-state, per-year, transition-triggered. Identified and fixed BEAR-month 100% turnover bug.
├── martingale_accumulate.py    FAILED: pyramid-position accumulation. A-share momentum stocks rotate too fast for accumulation (93% idle cash).
├── pure_alpha_strategy.py      No-timing benchmark: always-fully-invested Top-N momentum. Proves stock-selection alpha exists independently.
├── state_machine_v2.py         FAILED: risk-budgeting enhancements (vol target, ATR stop, DD circuit breaker). None improved V1.
├── final_improvements.py      5 batch tests: ETF rotation, vol filter, walk-forward validation, weekly vs monthly, gradient stop-loss.
├── final_friction_analysis.py  Commission/slippage sensitivity, capacity estimation, parameter stability, monthly return distribution.
├── backtest_state_machine_live.py  Production-config backtest: weekly/monthly rebalance, ATR trailing stop, round-lot sizing, dual momentum.
├── deploy_config_sweep.py      Rapid Top-N + dual momentum + round-lot parameter sweep for live deployment config.
├── bull_trap_analysis.py       CSI300 120MA breakout classification: 72 breaks identified, 24% true. Foundation for asymmetric confirmation design.
└── delayed_confirm_backtest.py Precursor: N-day post-breakout confirmation window. Led to the state machine's asymmetric confirmation mechanism.

tests/                             16 test files
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

### 9. State machine backtest scripts are standalone research artifacts
The state machine research scripts (`state_machine_strategy.py`, `alpha_decomposition.py`, etc.) duplicate indicator logic from `quanti/indicators.py` and data-loading logic from `quanti/data/storage.py`. This is intentional — they form a self-contained research pipeline that must run without depending on the production strategy classes. The production module `quanti/strategy/state_machine.py` is a clean extraction of the research findings, using the shared quanti infrastructure. When updating the research scripts, keep them self-contained. When deploying to production, use the `quanti/strategy/state_machine.py` module.

### 10. All state machine backtest numbers must come from the same engine
The `unified_report.py` script was created specifically to fix an internal consistency audit finding where yearly breakdowns and summary CAGR/MaxDD came from different backtest engines. Any new analysis scripts that produce numeric claims must use the `run_backtest()` function from `state_machine_strategy.py` (or `unified_report.py`). Do not create additional backtest engine variants for the same strategy. If a new analysis (e.g., turnover, friction, decomposition) is needed, extend `unified_report.py` to include it.

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
# Tests (16 test files)
python -m pytest tests/ -v

# ── State Machine Strategy (primary research pipeline) ──

# Full 81-parameter grid search + report generation (~30 min)
python scripts/state_machine_strategy.py          # → data/state_machine_report.md

# Unified single-engine report: alpha decomposition + monthly distribution + parameter sweep
python scripts/unified_report.py                  # → FINAL_REPORT.md

# Lookahead bias audit: 9 systematic checks, full-series vs causal
python scripts/audit_lookahead.py

# Independent causal backtest verification (gold standard)
python scripts/causal_backtest_verify.py

# Turnover analysis: per-state, per-year, transition-triggered
python scripts/turnover_analysis.py

# Alpha decomposition: timing vs stock selection
python scripts/alpha_decomposition.py

# Friction & robustness: commission/slippage sensitivity, capacity, stability
python scripts/final_friction_analysis.py

# Pure stock alpha benchmark (no market timing)
python scripts/pure_alpha_strategy.py

# Production-config deployment sweep (round-lot + dual momentum)
python scripts/deploy_config_sweep.py

# Failed explorations (documented, not for production)
python scripts/martingale_accumulate.py
python scripts/state_machine_v2.py

# ── Production ──

# Paper trading
python -m quanti.main_paper

# Live trading
python -m quanti.main_live
```

## Script Categories

All 15 scripts in `scripts/` form a single coherent research pipeline for the state machine strategy.

### Core Research
- `state_machine_strategy.py` — 81-param grid sweep, Train(2015-2021)→Test(2022-2025), generates report.
- `unified_report.py` — Single-engine alpha decomposition + monthly distribution + parameter sweep.

### Verification & Audit
- `audit_lookahead.py` — 9-step lookahead bias audit.
- `causal_backtest_verify.py` — Independent causal backtest verification.

### Analysis
- `alpha_decomposition.py` — Timing vs stock-selection alpha decomposition.
- `turnover_analysis.py` — Turnover quantification and fix verification.
- `final_friction_analysis.py` — Commission/slippage sensitivity, capacity, stability.

### Benchmarks & Counterfactuals
- `pure_alpha_strategy.py` — No-timing baseline proving stock-selection alpha.
- `martingale_accumulate.py` — FAILED: pyramid accumulation (A-share momentum rotates too fast).
- `state_machine_v2.py` — FAILED: incremental risk controls (none improved V1).

### Batch Improvement Tests
- `final_improvements.py` — 5 batch tests: ETF rotation, vol filter, walk-forward, weekly rebalancing, gradient stop.

### Deployment Configuration
- `backtest_state_machine_live.py` — Production-config backtest with round lots and ATR stops.
- `deploy_config_sweep.py` — Rapid parameter sweep for live deployment.

### Research Precursors
- `bull_trap_analysis.py` — Foundation discovery: 72 CSI300 breakouts, 24% true. Motivated asymmetric confirmation.
- `delayed_confirm_backtest.py` — Precursor: N-day confirmation window. Led to the state machine design.

Result summary: The V1 state machine (ADX=25, BR=45%, N_BR=5, N_RB=2) achieves Test CAGR=+7.0%, MaxDD=-5.1%, Sharpe=0.787 (2022-2025). Market timing contributes ~21% of CAGR (risk control), stock-selection alpha contributes ~79% (profit engine). The pure CSI300-weight timing alone produces +1.5% CAGR — timing's value is entirely in MaxDD reduction (from -30.6% to -5.1%).

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

## State Machine Strategy — Research Findings (2026-06-14)

### Configuration (v1_best from 81-param grid search)
- ADX threshold = 25, breadth bull = 45%, N_BR = 5 days, N_RB = 2 days
- BULL: Top-5 momentum stocks, full position. RANGE: Top-3, half position. BEAR: 100% 511880 money market ETF.
- Monthly rebalancing, -10% HWM trailing stop, 0.025% one-way commission.

### Core Findings
1. **Timing does not create returns, it controls drawdowns.** Pure CSI300-weight timing produces +1.5% CAGR. The strategy's +7.0% CAGR comes from momentum stock selection. Timing's sole contribution: reducing MaxDD from -30.6% (pure stock alpha) to -5.1%.
2. **Asymmetric confirmation works.** Entry-slow (5-day bear-to-range, 2-day range-to-bull) + exit-fast (immediate) filtration. Full-series vs causal state machine: 3292 trading days, 81 param combos, 0 mismatches.
3. **BULL months are the entire profit engine.** 4 BULL months in Test (8% of time) contributed ~67% of total alpha. 100% win rate in BULL months. RANGE months (13) produced negligible alpha. BEAR months (31, 65% of time) held cash.
4. **Momentum stock rotation frequency is very high in A-shares.** Top-5 stocks rarely persist for consecutive months. This killed the Martingale accumulation strategy (93% idle cash). Equal-weight monthly rebalancing is the correct approach.
5. **Incremental risk controls are harmful.** DD circuit breakers, vol targeting, ATR stops, gradient stop-losses — all tested, none improved V1. The timing switch itself is the best risk control. Adding layers only increases friction without improving risk-adjusted returns.
6. **Walk-forward confirms stability.** 7 rolling windows (36mo train, 12mo test), 4/7 positive, optimal params identical in 6/7 windows (ADX=22, BR=45, N_BR=3, N_RB=2).
7. **Capacity:** ~20M CNY (1% of P10 daily turnover for 5 positions). Survives 50bps slippage. Commission sensitivity: -0.7% CAGR per 10bps.

### Production Module
`quanti/strategy/state_machine.py` contains:
- `DailyStateMachine` — incrementally buildable, `update()` returns confirmed state each day
- `score_stock()` — composite momentum-trend-lowvol scoring function
- `LiveSignal` dataclass + `to_json()` — JSON output for execution system consumption

### Verified Absence of Lookahead Bias
9-step audit: MA120 (full vs causal: identical), ADX (identical), breadth MA20 (identical), data_at window (correctly bounded), state confirmation (backward-looking only), price_on (uses today's close — industry convention, covered by 0.025% comm), stock pool (minor survivorship bias, non-qualifying stocks filtered by data_at), 511880 data (clean), full-series vs causal state machine (0/3292 mismatches).

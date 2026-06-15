# AGENTS.md -- quanti Algorithmic Trading System

## File Map

```
quanti/
├── types.py                    Shared dataclasses: Bar, MarketData, Portfolio, Position, Signal, Order, OrderSide
├── indicators.py               Shared indicator functions: sma, ema, wilder_smooth, adx, adx_with_di, macd, kdj, bollinger_bands, compute_atr, compute_rsi
├── config/
│   ├── settings.py             All tunable parameters loaded from env vars (via dotenv). NEVER hardcode values in strategy/execution code.
│   └── etf_universe.py         Dynamic ETF universe with listing-date awareness and sector classification
├── data/
│   ├── schema.py               ETFDailyBar, IndexDailyBar, BondDailyBar dataclasses
│   ├── storage.py              SQLite metadata + Parquet time-series storage. load_bars() auto-resolves suffix variants (.SH/.SZ)
│   ├── validation.py           DataValidator: checks zero prices, high<low, duplicates, freshness, cross-source
│   ├── index_pe.py             IndexPEFetcher: CSI300 PE/PB percentile (Tushare + AkShare fallback)
│   └── ingestion/
│       ├── __init__.py          run_daily_ingest() entry point
│       ├── tushare_fetcher.py   Primary source (needs TUSHARE_TOKEN)
│       ├── akshare_fetcher.py   Fallback source (no token, Sina/CDN scraping)
│       ├── stock_fetcher.py     Individual stock data fetcher
│       └── cb_fetcher.py        Convertible bond data fetcher
├── strategy/
│   ├── base.py                 BaseStrategy ABC: generate_signals(), size_positions(), risk_check()
│   ├── etf_trend.py            ETFTrendStrategy: weighted-scoring entry, 5 exit methods, unified risk. Primary strategy.
│   ├── etf_rotation.py         ETFRotationStrategy: monthly top-3 ETF rotation, 3-factor composite, multi-sector concentration limits
│   ├── stock_momentum.py       StockMomentumStrategy: trend-first momentum on CSI300/500 constituents, DD breaker
│   ├── pe_band.py              PEBandAllocation: CSI300 PE percentile -> equity/bond/gold mix
│   ├── dividend_barbell.py     DividendBarbell: core-satellite with 510880/bond/gold/cash
│   ├── sector_rotation.py      Sector rotation strategy
│   ├── delayed_confirm.py      Delayed confirmation strategy
│   ├── signal_concentration.py Signal concentration limits
│   ├── signal_filters.py       MarketEnvironmentFilter: is_trending, is_bear_market, is_forbidden_period, NT intervention detection
│   └── cb_dual_low.py          CBDualLowStrategy: dual-low CB rotation
├── backtest/
│   └── engine.py               BacktestEngine: walk-forward, OOS, full metrics, T+1 settlement tracking
├── execution/
│   ├── order_manager.py        Order FSM: NEW->SUBMITTED->ACK->PARTIAL->FILLED/CANCELLED/REJECTED
│   ├── risk.py                 RiskChecker: check_all() - capital sufficiency, position limits, duplicate detection, stop-loss
│   ├── circuit_breaker.py      CircuitBreaker + MonthlyDrawdownBreaker + ConsecutiveLossBreaker + BreakerManager
│   └── broker.py               PaperBroker (simulated fills) + MiniQMTBroker (COM API, needs xtquant)
├── state/
│   ├── journal.py              SQLite: position_journal, order_journal, checkpoints. Source of truth for crash recovery.
│   └── recovery.py             recover_portfolio(), build_checkpoint_snapshot()
├── monitor/
│   ├── metrics.py              In-process gauges/counters
│   ├── logger.py               JSON-structured logging (loguru)
│   └── alerts.py               WeChat Work webhook + console fallback
├── main_paper.py               Paper trading entry point
├── main_paper_delayed.py       Paper trading with delayed confirmation
├── main_live.py                Live trading entry point
└── run_daily.py                Daily ingestion runner

scripts/
├── test_8etf.py                **PRIMARY**: 8-ETF progressive-enrollment backtest, uses omc_utils shared infrastructure
├── test_min_score.py           min_score threshold sweep, uses ETFData from omc_utils
├── test_7etf.py                7-ETF variant (semiconductor pool), inline DataStorage loading
├── asset_rotation_v4.py        Asset rotation strategy v4
├── asset_rotation_v6.py        Asset rotation strategy v6 with PE band support
├── v6_oos.py                   V6 out-of-sample validation
├── v6_pe_band.py               V6 PE band analysis
├── rising_ma_optimize.py       Rising MA parameter optimization
├── delayed_confirm_backtest.py Delayed confirmation backtest
├── fetch_macro.py              Macro data fetcher
├── fetch_new_etfs.py           New ETF listing fetcher
├── download_etf_universe.py    ETF universe data download
├── daily_signal.py             Daily signal reporter
├── diagnose_train.py           Training period diagnostics
├── run_daily.py                Daily data update runner
├── auto_update.py              Auto-update scheduler
├── _funcs.py                   Shared helper functions
├── _test_api.py                API integration tests
└── _test_cache.py              Cache integration tests

tests/                             18 test files, 232 tests
├── test_strategy_entry.py         17 tests: MA alignment, BB expansion, volume surge, ADX, DI diff, RSI, composite entry
├── test_strategy_exit.py          18 tests: ATR stop, RSI tighten, time stop, volatility stop, flat stop, gap risk, composite exit
├── test_strategy_signals.py        6 tests: 5-condition buy, MA cross sell, no-trend, insufficient data, stop-loss
├── test_backtest_engine.py         7 tests: single-ETF, flat market, crash exit, T+1 settlement, multi-ETF, walk-forward, OOS
├── test_allocation_strategies.py  12 tests: PEBand, DividendBarbell, StockMomentum
├── test_indicators.py             21 tests: SMA, EMA, ADX, ADX+DI, MACD, KDJ, BB, ATR, RSI
├── test_circuit_breakers.py       26 tests: all breakers + BreakerManager
├── test_risk_checker.py           20 tests: RiskChecker stop-loss, ATR, RSI, helpers, check_all integration
├── test_risk_check_regression.py   7 tests: BUY approval, duplicates, sizing, stop-loss, mixed orders
├── test_signal_filters.py         17 tests: is_trending, is_bear_market, forbidden periods, position sizing, should_trade
├── test_market_defenses.py        13 tests: NT intervention detection, market defense signals
├── test_sector_rotation.py        17 tests: sector rotation strategy
├── test_concentration_limit.py     7 tests: per-sector concentration caps, exempt sectors, backward compat
├── test_etf_universe.py           11 tests: ETF universe loading, listing dates, sector assignment
├── test_order_state.py            16 tests: all 7 FSM states, transitions, lifecycle, retry, timeout
├── test_data_validation.py         7 tests: clean bars, corrupt bars, cross-source, freshness, empty
├── test_signals.py                 5 tests: strategy interface, position sizing, risk filtering, type serialization
├── test_state_recovery.py          5 tests: no-checkpoint, from-checkpoint, snapshot, zero-pos, prune

pyproject.toml                   Build config + tool settings (ruff, black, isort, mypy, pytest, coverage)
.env.template                    Canonical list of all supported env vars with documented defaults
README.md                        Project overview
HANDBOOK.md                      Developer handbook
STRATEGY.md                      Strategy documentation
```

## Architecture Invariants (Do Not Violate)

### 1. Settings over hardcoded
Every tunable value lives in `quanti/config/settings.py` with an env var fallback. Code reads settings via `getattr(settings, 'KEY', default)`. Never hardcode periods, thresholds, multipliers, or percentages in strategy or execution code. The `.env.template` at the project root is the canonical reference for all supported env vars.

### 2. BaseStrategy ABC -- polymorphic strategies, shared engine
All engines (`backtest/engine.py`, `main_paper.py`, `main_live.py`) call the same three methods on whichever strategy they are given: `generate_signals(md)` -> `size_positions(signals, capital, pf)` -> `risk_check(orders, pf, market_data=md, risk_checker=...)`. Both `ETFTrendStrategy` and `CBDualLowStrategy` implement this interface. Any new strategy must do the same.

### 3. Unified risk path -- one gate, all engines
The risk path is identical across backtest, paper, and live: strategy exits -> RiskChecker.check_all() -> in-strategy filters. The backtest engine passes its `self.risk_checker` instance. Do not create separate risk paths for different engines.

### 4. State persistence before execution
Every order state transition writes to `order_journal` BEFORE the action executes. The journal is the source of truth for crash recovery. Circuit breaker state is saved in checkpoints via `BreakerManager.save_state()`.

### 5. DataStorage handles suffix resolution transparently
`DataStorage.load_bars(code)` probes `{code}.parquet`, `{code}.SH.parquet`, `{code}.SZ.parquet` in order. Callers can pass bare codes (e.g. `"510300"`) regardless of how the file was stored on disk. All ETF codes in production code should use bare codes (not `.SH`/`.SZ` suffixes).

### 6. Order quantity is shares/lots, not notional
`Order.quantity` is an integer count of shares (100-share lots for ETFs, 10-share lots for CBs). Never pass notional values as quantities.

### 7. Indicators module is the single source of truth
All indicator implementations (sma, ema, wilder_smooth, adx, adx_with_di, macd, kdj, bollinger_bands, compute_atr, compute_rsi) live in `quanti/indicators.py`. Strategy classes, RiskChecker, and server import from here. Do not add private indicator methods to strategy classes. (Exception: `_sma()` and `_compute_adx()` in `etf_rotation.py` predate the shared module and are preserved for backward compatibility.)

### 8. pyproject.toml is the single source of tool config
ruff, black, isort, mypy, pytest, and coverage settings live in `pyproject.toml`. Do not create separate config files (setup.cfg, .flake8, etc.).

### 9. Standalone research scripts are intentionally self-contained
Scripts like `asset_rotation_v4.py` and `delayed_confirm_backtest.py` duplicate indicator logic from `quanti/indicators.py`. This is by design -- they are research artifacts that must remain runnable without the quanti package on sys.path. If a strategy idea proves successful, integrate it into `quanti/strategy/` and use the shared indicator module there.

### 10. Strategy __init__.py controls the public API surface
`quanti/strategy/__init__.py` exports `ETFTrendStrategy`, `PEBandAllocation`, `DividendBarbell`, `MarketEnvironmentFilter`. Strategies not listed there (e.g., `ETFRotationStrategy`) are imported directly by their module path.

### 11. Diff-test before trust (mandatory validation gate)
When any new function, class, or module replaces existing inline logic, validate it against the reference implementation with identical inputs before trusting its output or committing it. "Produces plausible numbers" is insufficient. "Produces bit-identical outputs to the code it replaces" is the standard.

```python
ref = old_code(data); new = new_code(data)
assert np.allclose(ref, new), "diverges from reference"
```

Applies to: backtest engines, scoring functions, data loaders, indicator implementations, and any utility that claims to replicate existing behavior. Failure example: `omc_utils.run_backtest` defaulted `min_score=0.25` while the inline scripts it replaced had no threshold, producing a 1.33% CAGR gap that was only caught hours later. This is not advisory.

---

## Entry Strategy: ETFTrendStrategy

`ETFTrendStrategy.generate_signals()` computes 5 sub-scores (0-100 each), applies weights, and fires BUY when the composite >= `ENTRY_SCORE_THRESHOLD` (default 55, line 109 in settings.py):

1. **MA Alignment** (25%): SMA20 > 60 > 120 on both today AND yesterday, with separation bonus
2. **ADX Trend** (25%): ADX(14) > ADX_ENTRY_THRESHOLD AND +DI > -DI AND (+DI - -DI) > DI_DIFF_THRESHOLD
3. **BB Expansion** (20%): Bollinger Band(20,2.0) bandwidth expanding 1.2x vs past 5 bars AND close above upper band
4. **Volume Surge** (20%): Current volume > 20-day avg * VOLUME_SURGE_MULTIPLIER (excluding current bar from avg)
5. **Market Filter** (10%): Best index ADX(14) score: >=40->100, >=30->70, >=20->50, <20->30

SELL follows MA crossover reversal: SMA(fast) < SMA(slow). Legacy mode (`entry_mode="legacy"`) uses simple MA cross with ADX confirmation instead of weighted scoring.

## Exit Strategy: 5 Methods (etf_trend.py)

| Method | Trigger | Default |
|--------|---------|---------|
| `_flat_stop_loss` | Loss from avg_cost > STOP_LOSS_PCT (0% = disabled) | line 84 |
| `_atr_trailing_stop` | Price < HWM - atr_mult * ATR(14) | ATR_TRAILING_STOP_ENABLED=true |
| `_time_stop` | 50% reduce at 40d, full exit at 60d without new high | TIME_STOP_ENABLED=false |
| `_volatility_stop` | ATR_current > expansion_mult * ATR_entry | VOLATILITY_STOP_ENABLED=false |
| `_rsi_exit` | Standalone RSI-based tightening (when ATR stop disabled) | RSI_EXIT_ENABLED=false |

All five called from `risk_check()`. `MarketEnvironmentFilter.detect_nt_intervention()` tightens ATR stops when NT intervention is detected. `_check_gap_risk()` preemptively reduces position when intraday range is wide.

## Circuit Breakers (execution/circuit_breaker.py)

| Breaker | Trigger | Reset |
|---------|---------|-------|
| CircuitBreaker (general) | 3+ consecutive execution failures, daily loss > 2%, data gap > 5min | Manual |
| MonthlyDrawdownBreaker | Monthly realized loss > 5% of capital | Calendar month rollover |
| ConsecutiveLossBreaker | N consecutive stop-loss exits | 3-day cooldown or manual |

All wrapped in `BreakerManager`. State persisted in journal checkpoints.

## ETF Rotation Strategy (quanti/strategy/etf_rotation.py)

Monthly top-3 ETF rotation from an 8-ETF pool (510300/510500/159915/588360/563300/510880/518880/511880) with progressive enrollment for newer ETFs (588360 starts 2021-07-06, 563300 starts 2023-09-14). Multi-sector concentration limits cap ETFs per industry sector (default 2).

### Scoring: 3-Factor Composite (0.35 trend + 0.40 ADX + 0.25 momentum)
- **Trend (0.35, binary):** Price above 120MA = 1.0, else 0.0.
- **ADX (0.40, continuous):** min(ADX(14) / 50, 1.0). Trend-strength quality filter.
- **Momentum (0.25, continuous):** min(20d_return / 15, 1.0). Only positive returns score.
- **MACD/KDJ (0.0 each, opt-in):** Binary signals at zero weight by default. Constructor accepts non-zero weights for experiments. Functions live in `quanti/indicators.py` for chart display and future use.

### Gates (hard exclusion)
- **MA-rising:** 120MA must be higher than 120MA 20 bars ago.
- **Score > 0:** ETF must be above 120MA (trend=1).
- **min_score:** Composite must exceed threshold (default 0.30, tuned to 0.25 in backtests).

### Risk controls
- **DD breaker:** Full exit at 15% drawdown from peak equity.
- **HWM stop:** Per-position -10% trailing stop from entry high-water-mark.

### Baseline (Test 2022-2025, min_score=0.25)
Single-period: CAGR +14.3%, Sharpe 0.997, MaxDD -5.8%. Walk-forward (5 non-overlapping 2-year windows): CAGR +10.5%, Sharpe 0.300, MaxDD -10.2%. Wide inter-window variance (+49% to -7%) driven by market regime, not factor configuration.

### Factor Experiment Methodology (validated 2026-06-15)
**MANDATORY GATE: Before designing any factor experiment, compute statistical power.**
With N independent windows and per-window CAGR standard deviation sigma:
`MDE = 2.8 * sigma / sqrt(N)` (minimum detectable effect, 80% power, two-tailed).
For ETF rotation: sigma ~20%, N=5, MDE ~25% CAGR. No plausible factor modification
can produce a 25% CAGR difference. Any future factor experiment for ETF rotation
MUST be rejected at the planning stage unless a substantially larger dataset or a
fundamentally different validation framework is available. This is not advisory.

These rules are entry criteria for any surviving experiment. Evaluated against 7 distinct factor modifications that all failed to improve the baseline.

1. **Gate analysis first.** Before changing any factor or weight, compute forward returns of ETFs that barely fail vs barely pass each gate. The challenge-gate test on MA-rising near-misses returned mean -2.14% forward return vs +0.94% for the worst selected ETF. The gates are calibrated correctly. Do not modify them without equivalent evidence.

2. **Walk-forward is non-negotiable.** Single-period Test(22-25) backtests produce false signals (MACD/KDJ @0.10 showed 0.960 correlation with baseline on single-period but irrelevant on walk-forward). Non-overlapping windows with no parameter optimization are the minimum. `BacktestEngine.run_walk_forward()` or `omc_utils.compare_configs(windows=...)` are available.

3. **Directional signals cannot differentiate in cross-sectional selection.** MACD histogram >0 and KDJ J <80 are market-wide conditions. When bullish, they fire for all ETFs simultaneously, adding a constant offset that preserves relative rankings. Any factor added to this strategy must produce different values for different ETFs at the same point in time.

4. **The eligible pool averages 4 ETFs per rebalance date.** With a pool this small, factor weight ratios are nearly irrelevant. The trend+ADX grid search (0.30/0.70 through 0.60/0.40) produced identical CAGR within 0.1%. Gates, not scoring weights, drive performance.

5. **Stop after the first null result.** When a factor shows >0.95 correlation with baseline, do not try it in 5 more configurations. Accept that it adds no orthogonal information and move to a genuinely different factor class.

### omc_utils.py -- Experimental Analysis Library
`.omc/omc_utils.py` (~280 lines) provides reusable backtest infrastructure to prevent boilerplate in analysis scripts.

Key API:
```
from omc_utils import ETFData, monthly_rebal_dates, run_backtest, compare_configs, walk_forward_windows

data = ETFData.load(["510300", "510500", ...])
rebal = monthly_rebal_dates(data.dates, from_date="20220101", to_date="20251231")
cagr, sharpe, maxdd, equity = run_backtest(data, rebal, w_trend=0.35, w_adx=0.40, w_momentum=0.25)

# Walk-forward comparison in 4 lines
windows, labels = walk_forward_windows(monthly_rebal_dates(data.dates, "20150101", "20251231"), window_months=24)
results = compare_configs(data, all_rebal, configs, windows=windows)
```

The library delegates to `DataStorage` and `ETFRotationStrategy.compute_scores()` -- no reimplementation of production logic. Import via `sys.path.insert(0, '.omc')` from scripts in the project root or `.omc/` directory.

---

## How to Run

```powershell
# Tests (18 test files, 232 tests)
python -m pytest tests/ -v

# ── ETF Rotation ──

# 8-ETF progressive enrollment backtest
python scripts/test_8etf.py

# min_score threshold sweep
python scripts/test_min_score.py

# ── Production ──

# Paper trading
python -m quanti.main_paper

# Live trading
python -m quanti.main_live
```

## Known Limitations (Honest Assessment)

- **Intraday gap risk**: Stop-loss uses daily close. A -12% intraday crash recovering to -3% at close never triggers. `ETFTrendStrategy._check_gap_risk()` partially addresses this by preemptively reducing position when intraday range is wide. Real fix requires Phase 5 MiniQMT tick feed.
- **NT intervention contamination**: The National Team holds ~1.54T RMB in ETFs (as of 2025). When it buys/sells $10B/day, ADX spikes, volume surges, and momentum rankings shift. All five entry conditions can fire on NT activity. `MarketEnvironmentFilter.detect_nt_intervention()` flags this but does not suppress signals -- it only tightens exits.
- **T+1 settlement**: Sell proceeds settle next day. The backtest engine models this correctly (pending_dates, settled_cash vs cash). Live mode does not -- it needs broker settlement events.
- **Settings pollution in tests**: Multiple test files mutate settings attributes directly. A `conftest.py` autouse fixture restores settings after each test. This works but is fragile if new mutable settings are added without updating the fixture.
- **Zero-trade risk**: The weighted-scoring entry can produce few signals in choppy markets. This is by design (high conviction, moderate frequency) but adjust `ENTRY_SCORE_THRESHOLD` downward to increase signal frequency.
- **Circuit breaker daily PnL**: The general circuit breaker's daily drawdown rule receives actual daily P&L (computed from sell orders) in paper/live mode. However, it only captures realized (closed trade) P&L, not mark-to-market unrealized losses. A true daily drawdown check needs intraday MTM which requires a tick feed.
- **ETF rotation statistical power**: The walk-forward framework with 5 non-overlapping 2-year windows and 20% inter-window CAGR variance is underpowered. MDE ~25%. No factor configuration can be empirically distinguished from any other. Model selection is on parsimony grounds, not performance grounds.

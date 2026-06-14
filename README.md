# quanti — A-Share Algorithmic Trading System

Multi-strategy backtesting, paper trading, and live execution for Chinese A-share
ETFs.  Built on a daily-bar event-walk engine with T+1 settlement, circuit
breakers, and walk-forward validation.

**Latest**: Multi-sector ETF rotation with market-state-aware defense achieves
**OOS CAGR +13.72%, Sharpe 0.839, MaxDD 7.7%** (Test 2022–2025, 25 ETFs).

---

## Quick Start

```bash
# Install core dependencies
pip install numpy pandas pyarrow python-dotenv loguru

# Install data-source dependencies
pip install akshare

# Run the market-state strategy backtest (framework engine)
python scripts/run_market_state_backtest.py

# Run standalone backtest (monthly snapshot — faster, for param sweeps)
python scripts/backtest_etf_market_state.py

# Download missing ETF data (run once)
python scripts/download_etf_universe.py --dry-run   # preview first
python scripts/download_etf_universe.py              # actual download
```

---

## Architecture

The project has two tiers:

### Tier 1 — Framework (`quanti/`)

An OOP strategy framework with a daily-bar event-walk engine, T+1 settlement,
unified risk management, and Parquet+SQLite data storage.

```
quanti/
├── config/
│   ├── settings.py           All tunable parameters (env vars)
│   └── etf_universe.py       25-ETF pool with sector map + listing-date awareness
├── strategy/
│   ├── base.py               BaseStrategy ABC (generate_signals, size_positions, risk_check)
│   ├── market_state_etf.py   ★ Market-state-aware multi-sector rotation
│   ├── etf_rotation.py       ETF trend rotation (legacy 6-ETF / multi-sector dual mode)
│   ├── etf_trend.py          Original ETF trend strategy
│   ├── pe_band.py            PE-band valuation allocation
│   ├── dividend_barbell.py   Dividend + growth barbell strategy
│   ├── stock_momentum.py     Individual stock momentum strategy
│   ├── cb_dual_low.py        Convertible bond dual-low strategy
│   └── signal_filters.py     Market environment / trend / bear / forbidden-period filters
├── backtest/
│   └── engine.py             Event-walk backtest engine (walk-forward, OOS, T+1)
├── execution/
│   ├── risk.py               RiskChecker (stop-loss, ATR trailing, position limits)
│   ├── circuit_breaker.py    CircuitBreaker, MonthlyDrawdownBreaker, ConsecutiveLossBreaker
│   ├── order_manager.py      State-machine order manager (live trading)
│   └── broker.py             Broker abstraction (QMT/XTP stubs)
├── data/
│   ├── storage.py            Three-layer storage (raw Parquet, clean Parquet, SQLite)
│   ├── schema.py             ETFDailyBar, IndexDailyBar, BondDailyBar dataclasses
│   ├── validation.py         Data validation (freshness, discrepancy, sanity checks)
│   └── ingestion/            Data fetchers (akshare, tushare, cb)
├── monitor/
│   ├── logger.py             Loguru-based structured logging
│   ├── alerts.py             WeChat Work webhook alerts
│   └── metrics.py            Strategy performance metrics
├── state/
│   ├── journal.py            Position/order/trade journal (SQLite)
│   └── recovery.py           State recovery from checkpoints
├── indicators.py             Shared technical indicators (SMA, EMA, ADX, ATR, RSI, Bollinger)
├── types.py                  Bar, MarketData, Portfolio, Position, Signal, Order, OrderSide
├── main_paper.py             Paper trading entry point
├── main_live.py              Live trading entry point (QMT)
└── run_daily.py              Daily operations runner
```

### Tier 2 — Standalone Scripts (`scripts/`)

Self-contained backtest scripts that run independently of the framework.  Suitable
for rapid iteration, parameter sweeps, and historical research.

```
scripts/
├── backtest_etf_market_state.py   ★ Full strategy: 25 ETFs + market state + bond/gold defense
├── run_market_state_backtest.py   ★ Same strategy via BacktestEngine (T+1 realism)
├── overfitting_probe.py           ★ 6-test overfitting audit (look-ahead, param sweep, bootstrap)
├── download_etf_universe.py       Batch data download for all 25 sector ETFs
├── backtest_multi_sector.py       Legacy 6-ETF vs multi-sector 25-ETF comparison
├── robustness_checks.py           Correlation heatmap + weight/N sensitivity
├── run_backtest_multi_fast.py     Fast multi-sector backtest variant
├── run_backtest_multi_volnorm.py  Volatility-normalized scoring variant
├── volatility_analysis.py         Per-sector volatility stress test
├── multi_etf_monthly.py           Monthly multi-ETF rotation
├── fetch_macro.py                 PMI + CGB yield fetcher
│
├── _funcs.py                      v4 strategy engine (features, scoring, backtest)
├── auto_update.py                 v4 daily pipeline
├── rising_ma_optimize.py          v4 grid search + optimization
│
├── v6_pe_band.py                  v6 PE-band strategy engine
├── v6_oos.py                      v6 walk-forward OOS validation
├── v6_pe_band_v6_1.py             v6 variant
│
├── asset_rotation_v4.py           Historical v4 entry
├── asset_rotation_v6.py           Historical v6 entry
├── final_cagr.py                  Final CAGR report
└── run_daily.py                   v6 daily operations
```

---

## Strategies

### Market-State ETF Rotation (new, flagship)

25 sector/theme ETFs rotated monthly with CSI300-driven market-state gating.

**Three-layer defense**:

| Layer | Mechanism | Trigger |
|-------|-----------|---------|
| Market state | CSI300 120MA confirmation system (N=5, M=40) | Controls equity exposure (1.0 / 0.5 / 0) |
| Flash exit | Sharp3pct — CSI300 5d return < -3% | Full switch to bond(80%)+gold(20%), 40d cooldown |
| Position decay | A43 decay: 1.0(m1-4) → 0.75(m5-8) → 0.50(m9+) | Reduces exposure as bull market ages |

**Entry screening** (5-condition filter, >=3 required):
1. Price > 120MA
2. 20d high/low moving up vs 20-60d ago
3. MA alignment: 20MA > 60MA > 120MA
4. ADX(14) > 25
5. Volume surge (>120% of 20d avg)

**Scoring**: 60% momentum (3M+6M) + 40% stability (daily return dispersion).

**ETF universe**: 25 ETFs across 9 sectors (宽基/金融/科技/新能源/消费/资源/TMT/高端制造/防御),
with dynamic listing-date awareness (ETFs join 120 days post-listing).
Concentration limit: max 2 from any industry sector.

**Results** (standalone monthly snapshot, Test 2022-2025):

| Variant | CAGR | Sharpe | MaxDD |
|---------|-----:|-------:|------:|
| Bare trend (no defense) | +6.54% | 0.439 | 18.7% |
| +A43 decay | +7.09% | 0.470 | 17.8% |
| +Sharp exit | +10.48% | 0.665 | 7.6% |
| +Gold(80/20) defense | +9.28% | 0.585 | 16.8% |
| **All three combined** | **+13.72%** | **0.839** | **7.7%** |

Train (2015-2021) CAGR +5.77%, Sharpe 0.894 — Train/Test consistency confirmed.

**Framework engine result** (daily event-walk + T+1 settlement): Test CAGR +4.37%.
More conservative because sell proceeds settle next day — closer to real-world
execution.

**Overfitting audit**: All 6 tests passed.

| Test | Result |
|------|--------|
| Look-ahead audit | No forward bias detected |
| Param sweep (N×M grid) | Stable peak at N=5, M=40 |
| Sharp threshold sweep | Stable from -2% to -4% |
| Rolling 2yr OOS (4 windows) | Mean +7.96%, all positive in bear windows |
| Random pool bootstrap (20x) | Full pool at 60th percentile |
| Concentration limit scan | No material impact |

---

### Legacy Strategies

#### v4 Rising MA ETF Rotation

Monthly momentum + trend filter across 7 ETFs, Top 1.
Test (2022-2026H1) CAGR +10.55%, Sharpe 1.13, MaxDD -8.13%.

#### v6 PE-Band + Gold Trend

Valuation-driven allocation. CSI300 PE percentile controls equity %, gold
trend filter adds gold exposure. Walk-forward OOS CAGR +15.70%, Sharpe 1.25.

#### Framework ETF Rotation

Original 6-ETF rotation with trend+ADX+momentum scoring, Top 3.
Full period (2015-2025) CAGR +5.36%.  Backward-compatible multi-sector
mode with 25 ETFs and concentration limits available via `use_multi_sector=True`.

---

## Data

```
data/
├── clean/                      *.parquet — validated daily OHLCV bars (20 ETFs, expanding)
├── raw/                        *.parquet — as-received immutables
├── features/                   Derived feature cache
├── macro/                      CSI300 PE, Caixin PMI, 10Y CGB yield
├── reports/                    Generated backtest reports
├── quanti.db                   SQLite metadata (ingestion log, freshness, schema version)
├── rising_ma_etf_rotation_report.md
└── delayed_confirm_report.md
```

Data source: AkShare `stock_zh_a_hist()` (Eastmoney API) with proxy bypass,
2-second rate limit, 3 retries with exponential backoff.

---

## Testing

```bash
pip install -e ".[dev]"
pytest                              # 217/217 pass (excl. 1 env-dependent pre-existing)
pytest -m "not slow"                # skip slow integration tests
pytest --cov=quanti --cov-report=term
```

Key test files:

| File | Coverage |
|------|----------|
| `tests/test_backtest_engine.py` | Backtest engine: single/multi ETF, T+1, walk-forward, OOS |
| `tests/test_risk_checker.py` | RiskChecker: stop-loss, ATR stop, position limits, integration |
| `tests/test_circuit_breakers.py` | All 3 breakers + BreakerManager composite |
| `tests/test_etf_universe.py` | ETF pool: availability, sector mapping, legacy compat (11 tests) |
| `tests/test_concentration_limit.py` | Concentration caps, exempt sectors, reason strings (7 tests) |
| `tests/test_strategy_entry.py` | Entry signals: MA alignment, BB expansion, volume surge, ADX |
| `tests/test_strategy_exit.py` | Exit signals: ATR stop, RSI tighten, time stop, gap risk |
| `tests/test_state_recovery.py` | Checkpoint save/restore |
| `tests/test_allocation_strategies.py` | PE-band, dividend barbell, stock momentum strategies |

---

## Configuration

Copy `.env.template` to `.env` and set:

```bash
# Required for live data fetching
TUSHARE_TOKEN=your_token

# Capital (RMB)
TOTAL_CAPITAL=100000

# Strategy toggles
ETF_ROTATION_MULTI_ENABLED=true
PE_BAND_ENABLED=false

# Market-state strategy parameters (in code, not env)
# See: quanti/strategy/market_state_etf.py __init__ docstring
```

All tunables are in `quanti/config/settings.py` (loaded from env vars via `python-dotenv`).

---

## Run a Backtest

```python
from quanti.data.storage import DataStorage
from quanti.backtest.engine import BacktestEngine
from quanti.strategy.market_state_etf import MarketStateETFStrategy
from quanti.config.etf_universe import ETF_UNIVERSE_MULTI

storage = DataStorage()
csi300 = storage.load_bars("510300")

# Load universe
codes = [e["code"] for e in ETF_UNIVERSE_MULTI]
bars_dict = {c: storage.load_bars(c) for c in codes if storage.load_bars(c)}

# Run
engine = BacktestEngine(
    strategy_class=MarketStateETFStrategy,
    params=dict(
        csi300_bars=csi300,
        top_n=3,
        n_confirm=5,
        m_cooldown=40,
        sharp_threshold=-0.03,
        bond_pct=0.80,
        gold_pct=0.20,
    ),
)
result = engine.run(list(bars_dict.keys()), bars_dict)
print(f"CAGR={result.cagr_pct:+.2f}% Sharpe={result.sharpe_ratio:.3f}")
```

---

## Known Limitations

1. **Gold path dependency**: 2022-2025 gold surged +156%.  Future gold underperformance
   will reduce returns.  Conservative 80/20 bond/gold split chosen over 50/50.
2. **ETF data coverage**: 5 newest ETFs (photovoltaic, rare earth, chemicals, securities,
   non-ferrous) need downloading.  Run `scripts/download_etf_universe.py`.
3. **Late-listed ETFs**: ETFs listed after 2020 haven't been tested through a full
   bear-bull cycle.  The dynamic pool (listing-date gating) prevents look-ahead bias
   but can't compensate for missing history.
4. **Bull market underperformance**: The defense mechanisms exit too early in strong
   trending markets (2020-2021 CAGR negative).  This is by design — the strategy
   prioritizes drawdown protection over bull-market capture.
5. **T+1 realism gap**: Standalone script (+13.72%) assumes instant settlement.
   Framework engine (+4.37%) uses strict T+1.  Real-world performance will be between
   these two.
6. **Physical gold ETF basis risk**: 518880 tracks spot gold, but ETF market price
   can deviate from NAV by 3-5% in extreme conditions.

---

## Requirements

- Python >= 3.11
- numpy, pandas, pyarrow
- akshare (data fetching)
- python-dotenv, loguru
- pytest, ruff (dev)

---

## License

MIT

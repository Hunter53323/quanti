# quanti

A-share algorithmic trading — framework + standalone scripts.  
Core: 25-ETF multi-sector rotation with market-state defense (OOS +13.72%).

---

## Quick Start

```bash
# Download missing ETF data (one-time)
python scripts/download_etf_universe.py --dry-run
python scripts/download_etf_universe.py

# Market-state strategy — standalone (fast, param sweeps)
python scripts/backtest_etf_market_state.py

# Market-state strategy — framework engine (T+1, live-realistic)
python scripts/run_market_state_backtest.py

# v6 PE-Band — standalone single-file strategy
python scripts/v6_pe_band.py --verify
```

---

## Framework Strategies (`quanti/strategy/`)

All are `BaseStrategy` subclasses. Plug into `BacktestEngine` for unified T+1, risk checks, walk-forward.

| Strategy | File | Universe | Logic |
|----------|------|----------|-------|
| **MarketStateETF** | `market_state_etf.py` | 25 sector ETFs | CSI300 state machine + Sharp exit + bond/gold defense. Top 3, 5-condition entry |
| **ETFRotation** | `etf_rotation.py` | 6 or 25 ETFs | Trend+ADX+momentum scoring. Legacy mode (6 ETF) or multi-sector (25 ETF, concentration limits) |
| **ETFTrend** | `etf_trend.py` | ETFs | Original multi-indicator trend entry |
| **PEBandAllocation** | `pe_band.py` | 3 assets | PE percentile → equity/bond split. Framework version of v6 logic |
| **DividendBarbell** | `dividend_barbell.py` | Stocks+dividend | High-dividend + growth barbell allocation |
| **StockMomentum** | `stock_momentum.py` | Individual stocks | Multi-condition trend filter + momentum ranking |
| **CBDualLow** | `cb_dual_low.py` | Convertible bonds | Dual-low (low price + low premium) screening |
| **DelayedConfirm** | `delayed_confirm.py` | ETFs | Confirmation-delayed entry |

Framework engine: `quanti/backtest/engine.py` — daily event-walk, T+1 settlement, RiskChecker, circuit breakers, walk-forward, OOS.

### Market-State Strategy (results)

Standalone monthly snapshot (instant settlement):

| Variant | Test CAGR | Sharpe | MaxDD |
|---------|-----:|-------:|------:|
| Bare trend | +6.54% | 0.439 | 18.7% |
| +A43 decay | +7.09% | 0.470 | 17.8% |
| +Sharp3pct exit | +10.48% | 0.665 | 7.6% |
| +Gold(80/20) defense | +9.28% | 0.585 | 16.8% |
| **All three** | **+13.72%** | **0.839** | **7.7%** |

Train (2015-2021) +5.77%, Sharpe 0.894. All 6 overfitting audits passed.  
Framework engine (T+1): +4.37% — conservative, closer to live execution.

---

## Standalone Scripts (`scripts/`)

| Script | Depends on `quanti` | What it does |
|--------|:--:|--------------|
| `_funcs.py` + `auto_update.py` | — | v4 Rising MA pipeline (zero framework deps) |
| `rising_ma_optimize.py` | — | v4 grid search + optimization report |
| `v6_pe_band.py` | — | PE-Band standalone (zero framework deps). OOS Sharpe 1.25, CAGR +15.7% |
| `v6_oos.py` | — | v6 walk-forward OOS validation |
| `run_daily.py` | — | v6 daily operations layer |
| `backtest_etf_market_state.py` | data/config | Market-state rotation. Own backtest logic, uses framework for data+universe |
| `overfitting_probe.py` | data/config | 6-test audit. Own logic, framework data layer |
| `download_etf_universe.py` | data/fetch | Batch download. Uses `StockFetcher` + `DataStorage` |
| `run_market_state_backtest.py` | full | Runs `MarketStateETFStrategy` inside `BacktestEngine` (T+1, full risk path) |

### v4 Rising MA (standalone)

7-ETF momentum + trend filter, monthly Top 1. Results (Test 2022-01 to 2026-06-12):

| Metric | v4 | CSI300 B&H | Gold B&H |
|--------|----:|-----------:|---------:|
| CAGR | 10.55% | -0.79% | 21.73% |
| Sharpe | 1.13 | -0.04 | 1.21 |
| MaxDD | -8.13% | -36.11% | -28.55% |

Annual: 2022 +2.4%, 2023 +7.9%, 2024 +4.8%, 2025 +21.5%, 2026H1 +11.0%.  
Run: `python scripts/auto_update.py --signal` / `python scripts/rising_ma_optimize.py` for grid search.

### v6 PE-Band Results (standalone)

| Fold | Test Period | Sharpe | CAGR | MaxDD |
|------|------------|--------|------|-------|
| 1 | 2020-2021 | 1.16 | +8.84% | -9.19% |
| 2 | 2022-2023 | 0.67 | +10.80% | -11.81% |
| 3 | 2024-2025 | 2.27 | +28.61% | -5.98% |
| **OOS** | | **1.249** | **+15.70%** | **-13.16%** |

---

## Architecture

```
quanti/                         # Framework (OOP, pluggable)
├── config/                     settings.py, etf_universe.py (25 ETFs, 9 sectors)
├── strategy/                   8 BaseStrategy subclasses (see table above)
├── backtest/engine.py          Event-walk, T+1, walk-forward, OOS
├── execution/                  risk.py, circuit_breaker.py, order_manager.py, broker.py
├── data/                       storage.py (Parquet+SQLite), schema.py, ingestion/
├── state/                      journal.py, recovery.py
├── monitor/                    logger.py, alerts.py, metrics.py
├── indicators.py               SMA, EMA, ADX, ATR, RSI, Bollinger
└── types.py                    Bar, MarketData, Portfolio, Signal, Order

scripts/                        # Standalone (self-contained, no framework dep)
├── backtest_etf_market_state.py   ★ Market-state standalone
├── v6_pe_band.py                  ★ PE-Band standalone (493 lines)
├── _funcs.py                      v4 engine library
├── auto_update.py                 v4 daily pipeline
├── run_daily.py                   v6 operations layer
├── overfitting_probe.py           6-test audit
└── download_etf_universe.py       Batch data fetcher
```

---

## Data

20/25 ETFs loaded from AkShare (Eastmoney API). Parquet files in `data/clean/`.

| Core ETFs | Rows | Start |
|-----------|-----:|-------|
| 510300 CSI300 | 3,412 | 2012 |
| 510500 CSI500 | 3,215 | 2013 |
| 159915 ChiNext | 3,520 | 2011 |
| 510880 Dividend | 4,713 | 2007 |
| 518880 Gold | 3,129 | 2013 |
| 511880 MoneyMarket | 3,195 | 2013 |
| +14 sector ETFs | — | — |

5 ETFs need download. Run `scripts/download_etf_universe.py`.

---

## Testing

```bash
pytest   # 217/217 pass
```

Key suites: `test_backtest_engine.py` (T+1, walk-forward), `test_risk_checker.py` (stops, limits), `test_circuit_breakers.py` (3 breakers), `test_etf_universe.py` (11 tests), `test_concentration_limit.py` (7 tests).

---

## Known Limitations

**Gold path dependency** — 2022-2025 gold +156%. 80/20 bond/gold conservative split.  
**Bull market cost** — defense exits early in trending markets. 2020-2021 negative by design.  
**T+1 gap** — standalone +13.72% vs framework +4.37%. Difference is settlement lag.  
**Late-listed ETFs** — 5 newest ETFs lack 2015-2019 data. Dynamic pool prevents look-ahead, not absence.

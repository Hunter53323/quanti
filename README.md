# quanti — ETF Rotation v6

**Status**: Deploy-ready (2026-06-14)  
**Primary strategy**: PE-Band + Gold Trend — single file, 493 lines  
**OOS Sharpe (2020-2025)**: 1.249 | Acceptance: 11/12 applicable pass

---

## Quick Start

```bash
# Live allocation signal
python scripts/v6_pe_band.py --live

# Full 17-criterion acceptance test
python scripts/v6_pe_band.py --verify

# Daily data update
python scripts/v6_pe_band.py --fetch

# Data freshness check
python scripts/v6_pe_band.py --health

# Generate report
python scripts/v6_pe_band.py --report
```

## Strategy

Two-signal, 3-ETF model with monthly rebalancing:

| Signal | Mechanism | Effect |
|--------|-----------|--------|
| **CSI300 PE percentile** (5-year rolling) | Cheap = more equity, expensive = more bonds | `eq_pct = 0.60 - pe_pct * 0.50` |
| **Gold trend** (close > MA50, slope > 0) | 30% gold when trending, 0% when not | Binary filter |

Portfolio: CSI300 ETF (510300), Gold ETF (518880), CGB 5Y Bond ETF (511010).

No inverse-vol weighting. No regime detection. No cross-sectional scoring.

## Results

### Walk-Forward (3-fold, 2015-2025)

| Fold | Test Period | Sharpe | CAGR | MaxDD |
|------|------------|--------|------|-------|
| 1 | 2020-2021 | 1.16 | 8.84% | -9.19% |
| 2 | 2022-2023 | 0.67 | 10.80% | -11.81% |
| 3 | 2024-2025 | 2.27 | 28.61% | -5.98% |
| **OOS Aggregate** | | **1.249** | **15.70%** | **-13.16%** |

### Key Metrics (2020-2025 OOS)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Sharpe | 1.249 | v4: 0.699, CSI300: 0.119 |
| CAGR | 15.70% | CSI300: 2.29% |
| MaxDD | -13.16% | CSI300: -45.10% |
| Turnover | 168%/year | — |
| Gold allocation | 19.3% mean | Cap: <35% |

### The core failure mode is fixed

| Year | v4/v5 | v6 PE-Band |
|------|-------|-----------|
| 2017 | -17.90% | **+2.21%** |
| 2019-2020 (gold bull) | — | +20.15% |
| 2022-2023 (mixed) | -7.67% | -2.80% |
| 2026 YTD | -8.30% | +2.74% |

## Architecture

```
v6_pe_band.py (493 lines, zero strategy imports)
  |
  +-- reload_data()       Load/reload all data from parquet files
  +-- pe_pct_at()         CSI300 PE 5-year percentile (PIT, no look-ahead)
  +-- trend()             MA50 + slope gold trend filter
  +-- backtest()          Strategy backtest engine
  +-- run_walk_forward()  3-fold grid-search walk-forward
  +-- run_verify()        17-criterion acceptance test
  +-- run_live()          Current allocation signal + health warnings
  +-- fetch_all()         Daily data pipeline (ETFs + PE + macro)
  +-- health_check()      Data freshness validation
  +-- run_report()        Markdown report generator
```

**Backup architecture**: `scripts/_funcs.py` + `scripts/asset_rotation_v6.py` — hybrid scoring with regime detection (OOS Sharpe 0.517). Retained for environments without PE data.

## Files

| File | Purpose |
|------|---------|
| `scripts/v6_pe_band.py` | Primary strategy (self-contained) |
| `scripts/_funcs.py` | Hybrid scoring backup + v4 library |
| `scripts/asset_rotation_v6.py` | Hybrid scoring entry point |
| `scripts/asset_rotation_v4.py` | v4 backward compat |
| `scripts/auto_update.py` | Production pipeline (v4 + v6 signal) |
| `scripts/fetch_macro.py` | PMI + CGB yield fetcher |
| `docs/v6_项目总结.md` | Full project summary (Chinese) |
| `.omc/plans/` | Planning docs, results, journal, open questions |

## Data

| File | Content | Range |
|------|---------|-------|
| `data/clean/510300.parquet` | CSI300 ETF | 2012-2026 |
| `data/clean/510500.parquet` | CSI500 ETF | 2013-2026 |
| `data/clean/159915.parquet` | ChiNext ETF | 2011-2026 |
| `data/clean/510880.parquet` | Dividend ETF | 2007-2026 |
| `data/clean/518880.parquet` | Gold ETF | 2013-2026 |
| `data/clean/511010.parquet` | CGB 5Y Bond ETF | 2013-2026 |
| `data/clean/511880.parquet` | Money Market ETF | 2013-2026 |
| `data/macro/csi300_pe.parquet` | CSI300 PE | 2005-2026 |
| `data/macro/caixin_pmi.parquet` | Caixin PMI | 2012-2025 |
| `data/macro/cgb_10y_yield.parquet` | 10Y CGB Yield | 2002-2026 |

## Known limitations

1. Bond duration risk — 86% allocation to ~5Y bonds at current PE levels; untested in rising-rate regime
2. Fold 2 (2022-2023) is negative across all configs — valuation-based models hold through declines
3. Single equity ETF — CSI500/ChiNext rotation tested and rejected (noise > signal)
4. Gold trend filter is a trend-follower in a mean-reverting asset — enters/exits with lag
5. PE percentile loses discrimination above the 75th percentile — "expensive" vs "very expensive" barely differs

## v6.1 roadmap

1. Extreme-valuation PMI safety check for equity allocation
2. Test 3-year / 7-year / 10-year PE windows (only 5-year vs full-history tested)
3. Cross-validate against 2010-2014 pre-backtest window
4. Production paper-trading pipeline with drawdown alerts

## Git

```
20af3ff v6: Self-contained deployment -- one file, seven modes
a1d2f6d v6 daily auto-update pipeline with reload_data()
4ef31c1 Fix 4 bugs: _alloc CASH overwrite, AC-13 blind check...
f29a8e6 ETF Rotation v6: PE-Band + Gold Trend model (OOS Sharpe 1.249)
```

---

*Built 2026-06-14. Full process journal at `.omc/plans/etf_rotation_v6_journal.md`. Authoritative plan at `.omc/plans/etf_rotation_v6_v1.0.0.md`.*

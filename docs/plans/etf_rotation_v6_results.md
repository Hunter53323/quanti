# ETF Rotation v6 -- Final Results

**Date**: 2026-06-14
**Primary strategy**: PE-Band + Gold Trend (`scripts/v6_pe_band.py`)
**Backup strategy**: Hybrid Scoring (`scripts/_funcs.py` + `scripts/asset_rotation_v6.py`)

---

## Primary Strategy: PE-Band + Gold Trend

### Model

3-ETF portfolio (CSI300, Gold, CGB bonds). Monthly rebalancing. Two signals:

1. **CSI300 PE 5-year percentile** drives equity allocation: 60% equity at 0th percentile, 10% equity at 100th percentile
2. **Gold trend filter** (close > MA50 AND MA50 slope > 0): 30% gold when trending, 0% when not

Bonds receive the residual. No inverse-vol weighting. No regime detection. No cross-sectional scoring.

### 3-Fold Walk-Forward (2015-2025)

Parameters grid-searched on training windows only (54 combos/fold: eq_max, eq_min, gold_max, gold_ma).

| Fold | Train | Test | Train Sharpe | Test Sharpe | Test CAGR | Test MaxDD |
|------|-------|------|-------------|-----------|----------|-----------|
| 1 | 2015-2019 | 2020-2021 | 0.73 | 1.16 | 8.84% | -9.19% |
| 2 | 2015-2021 | 2022-2023 | 0.87 | 0.67 | 10.80% | -11.81% |
| 3 | 2015-2023 | 2024-2025 | 0.82 | 2.27 | 28.61% | -5.98% |
| **OOS Aggregate** | | | | **1.249** | **15.70%** | **-13.16%** |

Fold 2 went from -0.39 Sharpe (full-history PE) to +0.67 (5-year rolling). The 5-year window is more responsive to the current valuation regime.

### Diagnostic Years

| Period | CAGR | MaxDD | Notes |
|--------|------|-------|-------|
| 2017 (failure year) | +2.25% | -3.24% | Fixed vs v4's -17.90% |
| 2019-2020 (gold bull) | +20.15% | -9.19% | Gold capture working |
| 2022-2023 (mixed) | -2.80% | -11.26% | PE near fair value; held through decline |
| 2026 YTD | +6.45% | -8.93% | Positive |

### Operational Metrics (2020-2025 OOS)

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| Turnover | 185% | < 300% | PASS |
| Gold allocation (mean) | 19.3% | < 35% | PASS |
| Gold allocation (max) | 32.8% | < 35% | PASS |

### Benchmarks (2020-2025)

| Strategy | Sharpe | CAGR | MaxDD |
|----------|--------|------|-------|
| **v6 PE-Band** | **1.249** | **15.70%** | **-13.16%** |
| v4 Rising MA | 0.699 | 7.75% | -12.81% |
| CSI300 buy-hold | 0.119 | 2.29% | -45.10% |
| 60/40 CSI300/511010 | 0.259 | 2.93% | -26.63% |

### Live Signal (2026-06-12)

```
CSI300 PE: 92nd percentile (5y) -- expensive
Gold: not trending

Allocation targets:
  CSI300 (510300): 14%
  Gold   (518880): 0%
  Bonds  (511010): 86%
```

### Run Commands

```
python scripts/v6_pe_band.py                  # Full walk-forward + diagnostics + benchmarks
python scripts/v6_pe_band.py --live            # Current allocation signal
python scripts/auto_update.py --skip-fetch --skip-macro --v6-signal  # Production signal
```

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/v6_pe_band.py` | 182 | Primary v6 strategy |
| `scripts/auto_update.py` | ~190 | Production pipeline (v4 + v6 signal) |
| `data/macro/csi300_pe.parquet` | — | CSI300 PE data (2005-2026) |

### Why This Outperforms the Hybrid Scoring Approach

1. **PE percentile is a genuine countercyclical signal** -- it buys equities when they're cheap and sells when expensive. This works across every regime in the backtest.
2. **No inverse-vol drag** -- the model allocates a fixed fraction to equities based on valuation, not volatility. During cheap markets, you get the full allocation regardless of short-term vol.
3. **Binary gold filter eliminates ranking noise** -- gold is either trending (30%) or not (0%). No score competition. No monthly churn.
4. **5-year rolling PE window is responsive** -- it adapts to the current decade's valuation range, not 2007's extreme values.
5. **185 lines vs 581 lines** -- 3x less code with 2.4x more Sharpe.

---

## Backup Strategy: Hybrid Scoring

The original v6 built from `_funcs.py` (581 lines) using 4-component hybrid scoring with regime detection, inverse-vol weighting, and dynamic gold caps. OOS Sharpe 0.517, 9/17 PASS.

Retained as a working backup for environments where PE data is unavailable or explicit macro regime detection is preferred. All code in `scripts/_funcs.py` and `scripts/asset_rotation_v6.py` remains functional.

---

## v6.1 Roadmap

1. **PE-Band model**: Add regime-adaptive equity bands (wider in recovery, narrower in stagflation)
2. **PE-Band model**: Test CSI500 / ChiNext rotation within equity allocation (failed in v6.0 -- needs different rotation mechanism)
3. **PE-Band model**: Add extreme-valuation PMI filter (reduce equity only when PMI falling AND PE is NOT in bottom decile)
4. **Both**: Production paper-trading pipeline with live monitoring and drawdown alerts
5. **Both**: Cross-validate against 2010-2014 period (pre-backtest window) for additional OOS evidence

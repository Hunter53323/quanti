# Archive: Deleted Research Scripts

**Date**: 2026-06-14
**Reason**: Scripts from prior research phases (phase3, asset_rotation, hybrid strategies, delayed confirm variants, parameter analysis) were deleted during repo cleanup. This document preserves the research lineage.

**Recovery**: 43/44 deleted scripts have `.pyc` bytecode in `scripts/__pycache__/`. See section "Recovery Instructions" at the end for decompilation steps.

---

## 1. Phase 3 Research (5 scripts) — Direct Precursors to State Machine

These scripts formed the trail that led to the state machine strategy. All used the shared codebase infrastructure (`quanti/backtest/engine.py`, `quanti/data/storage.py`) rather than standalone backtesting.

### phase3_anti_bulltrap.py
- **Purpose**: Anti-bull-trap market timing. Only enters when 3+ of 5 bull-trap quality filters pass on CSI300 breakout day.
- **Filters**: Gap up (close vs previous high >= 0.8%), single-day return >= 1.5%, volume ratio >= 1.3x 20-day avg, amount ratio >= 1.3x, market breadth >= 55% stocks above 20MA.
- **Configs tested**: No filter, Quality gate >= 3, Quality gate >= 4, Adaptive quality sizing (quality 0-5 -> position 0-100%), Adaptive+min2.
- **Key finding**: Quality filters reduce false entries but the adaptive sizing approach still suffered from the same 2022-2025 drawdowns as prior strategies.
- **File size (pyc)**: 25,752 bytes

### phase3_market_timing.py
- **Purpose**: Market-timing-first stock selection. Only trades when CSI300 is in confirmed uptrend. Three market states: TREND_UP, CHOPPY, TREND_DOWN. Multiple configs: full position, half position, defense mode, MA month-confirm variants.
- **Core innovation**: 3-condition market trend check (above 120MA + ADX > 22 + MA20 rising), stock trend with 5 conditions (above MA120, higher highs/lows, MA alignment, ADX > 25, volume expansion), trend_strength composite score.
- **Configs**: TrendUpFull_DownCash, TrendUpFull_DownDefense, TrendUpHalf_DownCash, MA2moConfirm variants.
- **Key finding**: Market-timing-first approach outperformed unconditional strategies in 2022-2025 but MaxDD was still unacceptable. This is the direct ancestor of the state machine strategy.
- **File size (pyc)**: 23,671 bytes

### phase3_minimal.py
- **Purpose**: Fast Phase 3 backtest — writes results to stdout, progress file, then JSON. Tests legacy vs resonance entry modes on last 800 bars.
- **Key finding**: Resonance (multi-condition weighted scoring) produced higher Sharpe than legacy (simple MA cross) but had lower absolute CAGR in Test period.
- **File size (pyc)**: 6,743 bytes

### phase3_train_val_test.py
- **Purpose**: 3-way chronological split (train/validate/live-test) with gate check. Validates strategy stability across sequential time periods.
- **File size (pyc)**: 7,421 bytes

### phase3_v2_backtest.py
- **Purpose**: StockMomentumStrategy via BacktestEngine on train/validate/live-test split.
- **File size (pyc)**: 7,429 bytes

---

## 2. Asset Rotation (4 scripts) — The Only Positive Strategy Before State Machine

The asset rotation series evolved from basic ETF scoring to multi-asset trend rotation with risk management. v1 achieved +5.07% CAGR in 2022-2025 (Test) but with -39.9% MaxDD.

### asset_rotation_v2.py
- **Purpose**: Multi-Asset Trend Rotation v2. ETF pool: 510300, 510500, 159915, 510880, 518880, 511880. Scoring: weighted composite (above_ma, ADX normalized, ret_20d winsorized). Risk: top_n = 2-3, inverse-vol sizing, trailing DD circuit breaker.
- **Key innovation**: Expanded weight grid (19 combos), Calmar-weighted ranking for parameter selection.
- **Key finding**: Train (2015-2021) showed +17.6% CAGR but Test (2022-2025) showed +3.1% — severe overfitting despite strong in-sample performance.
- **File size (pyc)**: 4,951 bytes

### asset_rotation_v3.py
- **Purpose**: v3 focused on score-threshold sweeping for the ETF rotation strategy.
- **File size (pyc)**: 2,153 bytes

### asset_rotation_v4.py
- **Purpose**: v4 added adaptive threshold logic and a gold hedge component to the ETF rotation.
- **File size (pyc)**: 2,698 bytes

### asset_rotation_v6.py
- **Purpose**: v6 implemented dual-momentum ETF rotation with MA-based scoring. Added 60-day momentum filter (only buy if positive), volatility targeting at 18% vol target, drawdown circuit breaker at -15%.
- **Key innovation**: The best v6 config achieved the highest Test Sharpe in the rotation series (0.32).
- **File size (pyc)**: 23,139 bytes

---

## 3. Hybrid / Comparative Strategies (4 scripts) — Proving What Doesn't Work

These scripts systematically tested and eliminated strategy alternatives through large-scale comparison.

### backtest_alt_strategies.py
- **Purpose**: 4 non-trend strategies (A/B/C/D) on 6 ETFs. Report output in Chinese.
- **Strategies tested**: Alternative approaches to illustrate what doesn't work vs trend-following.
- **File size (pyc)**: 37,903 bytes

### backtest_hybrid_strategies.py
- **Purpose**: 4 multi-strategy schemes (S1-S4) on 30 CSI300 constituent stocks. Hybrid approaches combining momentum, mean-reversion, and risk parity.
- **File size (pyc)**: 72,382 bytes

### strategies_enhanced.py
- **Purpose**: S5-S9 enhanced strategy implementations (imported by hybrid_v2 and complete backtest). Contains the strategy class definitions.
- **File size (pyc)**: 49,983 bytes

### run_complete_backtest.py
- **Purpose**: Orchestrator script — all 9 schemes (S1-S9) self-contained in one file. This was the master comparison script.
- **File size (pyc)**: 108,215 bytes — largest of all deleted scripts

---

## 4. Delayed Confirmation Variants (5 scripts) — The Trail to Asymmetric Confirmation

The delayed confirmation series was the direct intellectual precursor to the state machine's asymmetric confirmation mechanism. The key insight was: wait N days after 120MA breakout before entering.

### delayed_confirm_backtest.py
- **Purpose**: Delayed Confirmation Backtest. After CSI300 crosses above 120MA, wait N days (confirm window) with volume/return checks before entering. If price drops below MA during window, enter cooldown (M days).
- **State machine**: CASH -> CONFIRMING -> CONFIRMED / COOLDOWN -> CASH
- **Parameters**: confirm_days (N) ∈ {3,5,10,20}, cooldown_days (M) ∈ {20,40,60}, position_size ∈ {full, half}
- **Key finding**: Reduced false-breakout drawdowns but missed slow-bull markets. The asymmetric confirmation idea was adopted into the state machine (5-day BEAR->RANGE, 2-day RANGE->BULL), but the cooldown mechanism was dropped.
- **This script is still in scripts/ — it was NOT deleted.**
- **File size (pyc)**: N/A (still present)

### delayed_confirm_pe_overlay.py
- **Purpose**: PE-band overlay on the delayed confirmation framework. Combines technical timing (120MA breakout confirm) with fundamental valuation (CSI300 PE percentile).
- **File size (pyc)**: 45,861 bytes

### delayed_confirm_surface.py
- **Purpose**: 3D parameter surface analysis for delayed confirmation window + cooldown combinations. Mapped the complete response surface of confirm_days vs cooldown_days.
- **File size (pyc)**: 59,661 bytes

### delayed_confirm_tiered.py
- **Purpose**: Tiered exit based on confirmation quality score. Higher quality confirmation = looser stop, lower quality = tighter stop.
- **File size (pyc)**: 39,835 bytes

### delayed_confirm_trade_log.py
- **Purpose**: Trade-by-trade logging and analysis for the delayed confirmation framework. Detailed entry/exit audit trail.
- **File size (pyc)**: 45,248 bytes

### delayed_etf_rotation.py
- **Purpose**: ETF rotation variant using the delayed confirmation framework instead of monthly rebalancing.
- **File size (pyc)**: 65,465 bytes

---

## 5. Parameter Analysis & Diagnostics (7 scripts) — The Discovery Engine

These scripts produced the critical finding that "all pure equity strategies lost money in 2022-2025 Test."

### param_grid_search.py
- **Purpose**: 54-combo grid search with dual modes: `--mode grid_search` for parameter optimization, `--mode diagnostic` for sub-period breakdowns and failure analysis.
- **Key finding**: The diagnostic mode revealed that Train CAGR was always positive (some configurations reaching +45%) but Test CAGR was universally negative for pure equity strategies. This was the discovery that motivated the state machine design.
- **File size (pyc)**: 60,454 bytes

### strategy_fix_validate.py
- **Purpose**: DD-exit threshold sweep + vol-scaled sizing validation. Tested whether lowering the drawdown exit threshold could rescue any of the failing strategies.
- **Key finding**: No single parameter fix could rescue pure equity strategies. Structural change (adding a timing overlay) was required.
- **File size (pyc)**: 29,232 bytes

### sensitivity_v6.py
- **Purpose**: Parameter sensitivity analysis for the asset rotation v6 strategy. One-at-a-time perturbation of every parameter with Test CAGR impact measured.
- **File size (pyc)**: 39,756 bytes

### sharp_decay_merge.py
- **Purpose**: Sharpe decay analysis — measures how much Train Sharpe exceeds Test Sharpe. Used as an overfitting diagnostic across all strategy variants.
- **File size (pyc)**: 39,058 bytes

### signal_decompose.py
- **Purpose**: Decompose composite entry signals into sub-components. Isolated the individual contribution of MA alignment, ADX, volume, and breadth to entry signal quality.
- **File size (pyc)**: 15,736 bytes

### trace_all_trades.py
- **Purpose**: Full trade log trace — reconstructs every entry, exit, and position change across the entire backtest period for forensic analysis.
- **File size (pyc)**: 17,649 bytes

### decay_timing_clean.py
- **Purpose**: Signal decay timing analysis — measures how quickly a composite signal's predictive power decays after generation.
- **File size (pyc)**: 31,123 bytes

---

## 6. Production / Operational Scripts (5 scripts)

### live_signal.py
- **Purpose**: Daily pre-market signal generation for StockMomentumStrategy. Outputs JSON signal files consumable by the execution system.
- **File size (pyc)**: 7,892 bytes

### market_scanner.py
- **Purpose**: Daily market scanner — runs across all symbols to generate composite trend scores and flag regime changes.
- **File size (pyc)**: N/A

### auto_update.py
- **Purpose**: Auto-update data fetcher. Scheduled job to pull latest daily bars from Tushare/AkShare.
- **File size (pyc)**: 13,677 bytes

### fetch_macro.py
- **Purpose**: Macro data fetcher — pulls index PE, interest rates, and macro indicators.
- **File size (pyc)**: 11,812 bytes

### calibrate_gold_boost.py
- **Purpose**: Gold allocation calibration for the dividend barbell strategy. Determined optimal gold weight based on volatility and correlation.
- **File size (pyc)**: 9,869 bytes

---

## 7. Walk-Forward & Forward Testing (3 scripts)

### s3_walkforward_standalone.py
- **Purpose**: Standalone walk-forward validation for S3 strategy. Rolling 36-month train, 12-month test windows.
- **File size (pyc)**: 28,952 bytes

### final_validated_backtest.py
- **Purpose**: Final validated backtest run — the definitive Train/Test split run that confirmed strategy performance before the state machine era.
- **File size (pyc)**: 46,887 bytes

### forward_2026_test.py
- **Purpose**: Forward test extending into 2026 data. Tested whether strategies that worked in 2022-2025 continued to work on the newest available data.
- **File size (pyc)**: 32,093 bytes

---

## 8. Infrastructure / Utilities (4 scripts)

### batch_download_stocks.py
- **Purpose**: Batch stock data download — CSI300 + CSI500 constituents via StockFetcher.
- **File size (pyc)**: 4,307 bytes

### fetch_history.py
- **Purpose**: Batch historical ETF data download via AkShare Sina source.
- **File size (pyc)**: 4,823 bytes

### sweep_threshold.py
- **Purpose**: Single-param ENTRY_SCORE_THRESHOLD sweep using BacktestEngine. Tests entry sensitivity.
- **File size (pyc)**: 3,028 bytes

### _funcs.py
- **Purpose**: Shared function library for standalone scripts. Contains data loading, indicator computation, and metrics calculation utilities.
- **File size (pyc)**: 63,469 bytes

---

## 9. Runners (2 scripts)

### run_phase3_backtest.py
- **Purpose**: AB comparison runner — baseline vs 5-condition vs extended universe + ADX sweep.
- **File size (pyc)**: 7,858 bytes

### run_delayed_confirm_backtest.py
- **Purpose**: Runner for the delayed confirmation backtest sweep.
- **File size (pyc)**: 22,473 bytes

---

## 10. Production Strategy Backtests (4 scripts)

### production_grid_search.py
- **Purpose**: 54-combo production parameter grid search. The production equivalent of param_grid_search.py.
- **File size (pyc)**: 30,155 bytes

### rising_ma_optimize.py
- **Purpose**: Rising MA optimization — sweep of MA parameters for the rising-MA based entry signal.
- **File size (pyc)**: 35,937 bytes

### trend_end_exit.py
- **Purpose**: Trend-end exit signal optimization. Tested multiple exit rules (ATR trail, time stop, volatility stop) to find optimal exit logic.
- **File size (pyc)**: 45,389 bytes

### run_dividend_barbell_backtest.py
- **Purpose**: DividendBarbell backtest via BacktestEngine. Core-satellite with 510880/bond/gold/cash allocation.
- **File size (pyc)**: 7,297 bytes

### run_pe_band_backtest.py
- **Purpose**: PEBandAllocation backtest with rolling PE provider. CSI300 PE percentile -> equity/bond/gold mix.
- **File size (pyc)**: 12,327 bytes

---

## 11. Permanently Lost

### backtest_hybrid_v2.py
- **Purpose**: Expanded hybrid — 9 schemes (S1-S9) on 100 stocks. Used PE data for S5. The v2 version of the hybrid strategy comparison.
- **No .pyc in cache**. This is the only completely unrecoverable script.
- The nine scheme definitions (S5-S9) are partially preserved in `strategies_enhanced.pyc`, which IS in the cache.

---

## Research Lineage Summary

```
Phase 3: Momentum Stock Selection
  ├── phase3_market_timing.py      → 3-condition market trend
  ├── phase3_anti_bulltrap.py      → 5-filter quality gate
  └── phase3_*.py                  → train/val/test splits
        ↓
Delayed Confirmation Exploration
  ├── delayed_confirm_backtest.py  → N-day post-breakout window
  ├── delayed_confirm_surface.py   → 3D parameter mapping
  └── delayed_confirm_tiered.py    → quality-based tiered exits
        ↓
Discovery: 72 CSI300 breakouts, 24% true (bull_trap_analysis.py)
        ↓
STATE MACHINE STRATEGY (current)
  ├── asymmetric confirmation (slow entry, fast exit)
  ├── three-regime classification (BEAR/RANGE/BULL)
  └── breadth + ADX filters

Parallel Research (Eliminated):
  ├── Asset Rotation v2→v6        → always-invested ETF rotation (overfitting)
  ├── Hybrid Strategies S1-S9     → all failed in Test period
  └── Param Grid Search (54 combos) → diagnostic mode proved need for timing
```

---

## Recovery Instructions

### Prerequisites
- cmake installed and in PATH
- Visual Studio 2022 with C++ build tools (or MSVC compiler)
- Git available

### Steps

```powershell
# 1. Clone and build pycdc (Python 3.11 bytecode decompiler)
git clone --depth 1 https://github.com/zrax/pycdc.git $env:TEMP\pycdc_src
Push-Location $env:TEMP\pycdc_src
New-Item -Force -Type Directory build | Out-Null
Push-Location build
cmake .. -A x64
cmake --build . --config Release
$PYCDC = (Get-ChildItem -Recurse . -Filter "pycdc.exe" | Select-Object -First 1).FullName
Pop-Location; Pop-Location

# 2. Navigate to project root
Set-Location C:\study\AIWorkspace\quanti

# 3. Batch decompile all deleted scripts
$deletedNames = @(
    "_funcs", "asset_rotation_v2", "asset_rotation_v3", "asset_rotation_v4",
    "asset_rotation_v6", "auto_update", "backtest_alt_strategies",
    "backtest_hybrid_strategies", "batch_download_stocks",
    "calibrate_gold_boost", "decay_timing_clean", "delayed_confirm_pe_overlay",
    "delayed_confirm_surface", "delayed_confirm_tiered", "delayed_confirm_trade_log",
    "delayed_etf_rotation", "fetch_history", "fetch_macro",
    "forward_2026_test", "live_signal", "param_grid_search",
    "phase3_anti_bulltrap", "phase3_market_timing", "phase3_minimal",
    "phase3_train_val_test", "phase3_v2_backtest", "production_grid_search",
    "rising_ma_optimize", "run_complete_backtest", "run_delayed_confirm_backtest",
    "run_dividend_barbell_backtest", "run_pe_band_backtest", "run_phase3_backtest",
    "s3_walkforward_standalone", "sensitivity_v6", "sharp_decay_merge",
    "signal_decompose", "strategies_enhanced", "strategy_fix_validate",
    "sweep_threshold", "trace_all_trades", "trend_end_exit"
)

foreach ($name in $deletedNames) {
    $pycFile = "scripts\__pycache__\$name.cpython-311.pyc"
    if (Test-Path $pycFile) {
        Write-Host "Recovering: $name.py"
        & $PYCDC $pycFile 2>&1 | Out-File "scripts\$name.py" -Encoding UTF8
    }
}

Write-Host "Recovery complete. Recovered files in scripts/"
```

### Recovered File Verification

After recovery, verify with:
```powershell
python check.py --quick
```

---

*This document was generated 2026-06-14. All information has been cross-referenced with the bytecode files in `scripts/__pycache__/` and the session logs from the state machine research session.*

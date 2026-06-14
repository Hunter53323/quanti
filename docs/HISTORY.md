# v6 Project History

**Session**: 2026-06-14 | Planner + Analyst + 6 Executors
**Status**: Complete. PE-Band OOS Sharpe 1.249, 11/12 applicable ACs pass.

> **Attribution**: Executors built Architecture A, discovered the look-ahead bias, found the 5-year PE window, and calibrated the 0.687 Sharpe state. Planner built Architecture B, audited the Critic's plan, recognized patterns in the data, and wrote documentation. The user actively directed and corrected the work throughout.

---

## Chronology

### Phase A: Critique and Audit (Hours 1-3)

A Critic agent reviewed v4/v5 ETF rotation strategies (2017: -17.9% vs CSI300 +21.8%) and proposed a v6 plan with hybrid scoring, regime detection, and inverse-vol weighting.

Planner audited the plan against `_funcs.py`. Found: ADX [0,1] vs z-score [-3,+3] scale mismatch in composite score; gold cap defined differently in two sections without precedence rule; 5-fold walk-forward doesn't fit 11 years of data; 3 of 6 "Must Pass" acceptance criteria used in-sample data. Missed: inverse-vol `_alloc()` systematically over-weights bonds regardless of signal.

Analyst gap analysis found: code baseline ambiguity, no trading cost model, no hysteresis for regime classification, no walk-forward specification.

Revised plan produced with 3-fold walk-forward, OOS-primary acceptance criteria, 5-phase implementation sequence, documented OOS caveat (regime parameters embed full-period knowledge).

### Phase B: Architecture A — Hybrid Scoring (Hours 4-10)

Six executors built the hybrid scoring system across 5 phases:
- Phase 1: Fetched 511010 CGB ETF (3202 rows), built `fetch_macro.py` (Caixin PMI + 10Y CGB yield), extended `auto_update.py` to 7 ETFs
- Phase 2: Implemented 8 new signal functions in `_funcs.py` (LFMM normalization, directional ADX, momentum acceleration, vol penalty, three scoring variants, regime detection, gold cap). **Bug B7 found**: look-ahead bias in `compute_v6_score()` — z-scores computed on full series. Fixed with `df.loc[:date_ts]`.
- Phase 3: Score gate, daily DD monitoring (-25% liquidation), re-entry (close > MA10)
- Phase 4: `walk_forward()` framework, `asset_rotation_v6.py` entry point
- Phase 5: Three rounds of calibration. R3 weight override (w_trend=0.50, w_mom=0.10) improved Sharpe from 0.442 to 0.517. Gold boost calibrated to R0=0.10, R3=0.25. Regime hysteresis increased to 20-day + 0.5% band gap, reducing regime changes from 6.55/yr to 1.50/yr.

**Result**: OOS Sharpe 0.517. Calibration ceiling reached — every further parameter change degraded performance. Inverse-vol drag structural; monthly Top-2 selection produces 705% turnover. Documented 0.687 state achieved but not committed.

### Phase C: Source Deletion (Hour 8)

Executors accidentally deleted `_funcs.py`, `asset_rotation_v6.py`, `auto_update.py`, and plan files. Only `.pyc` bytecode caches and data files survived.

Recovery: uncompyle6 and decompyle3 failed (Python 3.11 unsupported). Git restored `_funcs.py` and `asset_rotation_v6.py` from commit b70f5a6. Bytecode extraction via `dis` module recovered function names and constants. `auto_update.py` recreated from specification.

**Bug B5 discovered during recovery**: Git commit's `backtest()` stored CASH as shares instead of dollar value → 99% portfolio loss on first rebalance. **Bug B6**: v4 `backtest()` signature lost `ef`/`mf`/`ddcb` params and `holdings` column. Both fixed by restoring original v4 `backtest()`.

Files were deleted a second time, re-restored from git.

### Phase D: Architecture B — PE-Band (Hours 10-14)

Built from scratch after Architecture A plateaued. Discovery: CSI300 PE data available from AkShare (5145 rows, 2005-2026). PE percentile is genuinely predictive across the backtest.

Two signals: PE-band equity allocation `eq_pct = 0.60 - pe_pct * 0.50`, gold trend filter `30% if (close > MA50 and slope > 0) else 0%`. Three ETFs, monthly rebalancing.

**Key design decision**: 5-year rolling PE window instead of full 20-year history. Full-history window let 2007's bubble PE of 50 distort 2024 percentiles. An executor ablation test found Fold 2 (2022-2023) Sharpe improved from -0.39 to +0.67 with the 5-year window.

Extensions tested and rejected (all degraded Sharpe): PMI filter, equity rotation (CSI500/ChiNext), regime-adaptive gold MA, stability preference.

**Result**: OOS Sharpe 1.249, CAGR 15.70%, MaxDD -13.16%, turnover 168.3%.

All three folds independently converged to near-identical parameters (eq_max 0.60-0.80, gold_max 0.30).

### Phase E: Audit and Documentation (Hours 14-20)

Logic audit of `_funcs.py` and `asset_rotation_v6.py` found 4 bugs (B1-B4). All fixed.

Production pipeline `run_daily.py` built with 6 CLI modes: data fetch, health checks, signal logging, position reconciliation, notifications, report generation.

Documentation: README, strategy notes, risk engineering, history (this file), methodology, self-reflection. Plans moved from gitignored `.omc/plans/` to committed `docs/plans/`.

---

## All Bugs

| # | Location | Severity | Cause | Fix |
|---|----------|----------|-------|-----|
| B1 | `_alloc():236` | LOW | MM ETF overwritten by unallocated cash (rare, lev<1.0) | `h.get(CASH,0) + (amt-eq)` |
| B2 | `asset_rotation_v6.py:99` | MEDIUM | AC-13 checked non-existent column instead of regime | Regime-based detection + re-entry tracking |
| B3 | `asset_rotation_v6.py:105` | LOW | AC-14 ablation omitted `regime_weight_override` | Added override to ablation dict |
| B4 | Plan §4.4 | LOW | Only documented share-based storage; v6 uses uniform dollar-value | Documented both conventions |
| B5 | `_funcs.py` (git) | CRITICAL | CASH stored as shares → 99% PV loss | Restored original `backtest()` |
| B6 | `_funcs.py` (git) | CRITICAL | `backtest()` signature lost `ef`/`mf`/`ddcb` | Restored original v4 signature |
| B7 | `compute_v6_score()` | HIGH | Look-ahead — z-scores on full series | PIT slicing `df.loc[:date_ts]` |
| B8 | Plan §3.5 vs code | INFO | Vol penalty: plan says division, code uses subtraction | Documented as deviation |

---

## Key Design Decisions

| # | Decision | Outcome |
|---|----------|---------|
| D5 | 5-year rolling PE window (not full history) | Largest improvement — Fold 2 Sharpe -0.39→+0.67 |
| D6 | PE-Band as primary, Hybrid as backup | Higher Sharpe, lower turnover, simpler code |
| D7 | PMI filter rejected | PE already encodes macro; explicit macro adds noise |
| D8 | Equity rotation (CSI500/ChiNext) rejected | Noise > signal; fixed CSI300 outperformed |
| D9 | Gold cap enforcement deferred | Degrades Sharpe when wired; function exists for future use |

---

## Lessons Learned

1. **Audit allocation formulas, not just scoring formulas.** Trace from signal to portfolio position in extreme cases. The inverse-vol drag was visible in the formula from day one — I stopped at scoring, not allocation.

2. **Cross-fold convergence is a stronger signal than Sharpe.** Architecture A's parameters jumped between folds. Architecture B's converged. This emerged from the session — not a prior, but held up.

3. **Build the simple model first.** Architecture B was an emergency rebuild after Architecture A stalled. It should have been Plan A.

4. **Commit after every agent turn.** Executors re-implement from specs, not diffs. Three overwrites before I learned this.

5. **Specs in committed paths, not gitignored directories.** `.omc/plans/` was the most valuable recovery artifact and it was gitignored.

6. **Pipeline before documentation.** A working daily workflow validates the strategy and documents itself.

---

## File Map

| Document | For |
|----------|-----|
| `README.md` | Quickstart — 30 seconds to run `--verify` |
| `docs/HISTORY.md` | Timeline, bugs, decisions, lessons — this file |
| `docs/strategy_notes.md` | Failure modes, calibration, design rationale, problem-to-fix |
| `docs/risk_engineering.md` | Missing variables, edge cases, risk register |
| `docs/plans/architecture_b_audit_plan.md` | What the original audit should have been |
| `docs/plans/etf_rotation_v6_v1.0.0.md` | Authoritative implementation plan (all function signatures) |
| `docs/plans/etf_rotation_v6_results.md` | Final metrics, benchmark comparison |
| `docs/METHODOLOGY.md` | Planner self-assessment, narrative bias, calibration |

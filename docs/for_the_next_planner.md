# For the Next Planner

Read `docs/plans/etf_rotation_v6_journal.md` and `docs/brain_dump.md` for the full story. This is the compressed version.

---

## Rules for the next session

### 1. Audit the allocation formula, not just the signal construction

The biggest mistake in this session: I audited the score formula (ADX normalization, gold cap precedence, walk-forward protocol) but did NOT audit the allocation formula. The allocation formula was `w = (1/v) / sum(1/v)` — inverse-vol weighting. It allocates the most capital to the lowest-vol asset regardless of signal quality. When bonds and money market have near-zero vol, they dominate.

**Rule**: For any rotation strategy, trace the allocation from score to position for three cases: all scores equal, one score dominates, all scores weak. If the allocation doesn't match the signal intent in at least two of these cases, the formula is broken.

### 2. Convergence test is the kill switch

Run one round of calibration with a grid search. If the optimal parameters are not the same (within a small delta) across all folds, the model is overfitted. Kill it immediately. Do not calibrate further. Do not add more components. Convergence across folds is a stronger signal than any single Sharpe value.

In this session: Architecture A didn't converge. Architecture B converged to the same parameters in all three folds. That difference was more informative than the Sharpe gap (0.517 vs 1.249).

### 3. Plan two architectures from the start

One complex architecture (the research direction) and one simple architecture (the production fallback). The simple one is built first because it sets the baseline, can be iterated faster, and is the fallback if the complex one fails. The complex one is only pursued if the simple one proves insufficient.

In this session: The simple model (PE-Band, 125 lines, 2 signals) was discovered as an emergency response after the complex model (Hybrid Scoring, 581 lines, 12+ parameters) reached its calibration ceiling. The simple model should have been Plan A, not Plan B.

### 4. Kill models that have more parameters than the data supports

Architecture A had 12+ tunable parameters, regime weights per quadrant, a score gate, gold boost, and DD controls — on 11 years of monthly data. Architecture B had 4 parameters. The complex model's parameters couldn't be estimated reliably because some regime quadrants had fewer than 20 observations. The simple model's parameters converged because the data-to-parameter ratio was adequate.

**Rule**: If a model has a component that only activates in a regime with <20 historical observations, remove that component. You cannot calibrate what you cannot observe.

### 5. Executors re-implement from scratch. Commit before spawning.

Three times in this session, executors overwrote fixes I had applied. Executors work from specifications, not diffs. They re-implement rather than modify. The protocol that would have prevented this:

1. Executor outputs go to a staging file or directory, never directly to the working tree
2. I review the diff, apply it, and commit
3. Then I spawn the next executor from a clean, committed working tree
4. Never let an executor clean up — I handle cleanup after verification

### 6. Pipeline before documentation

Build the daily operational wrapper immediately after the strategy engine is verified. The pipeline validates that the strategy actually runs in production. It documents itself via log output and report generation. Beautiful docs are written after the pipeline is confirmed working.

In this session: Four markdown files were written before `run_daily.py` existed. The pipeline was the last thing built. A working pipeline with thin docs is a better deliverable at hour 8 than beautiful docs with no pipeline at hour 12.

### 7. Gitignore nothing you would need to rebuild from

Plans, journals, results, questions, calibration history — these are specifications. Code can be regenerated from specifications. If a file would be needed to recover from a source deletion, it goes in `docs/` and gets committed.

In this session: The implementation plan lived in `.omc/plans/` (gitignored) for the entire session. When source files were deleted, the plan was the only artifact that enabled recovery — and it was one directory level away from being lost.

### 8. The simplest model that captures the signal wins

PE-band outperformed hybrid scoring by 76% on Sharpe with 75% less code. This wasn't luck. Every additional component (regime detection, cross-sectional scoring, score gate, DD controls, gold boost) is an additional noise source. Components interact. Interactions create edge cases. Edge cases create bugs. Bugs create calibration debt.

**Rule**: Start with one signal and one allocation rule. Add components one at a time, testing each against the baseline. If a component doesn't improve OOS Sharpe by at least 0.05, reject it. In this session: 100% of tested extensions were rejected. The winning model was the one that added nothing.

### 9. The value trap problem has no in-model solution

Any valuation-based model will lose money in a structural downrating scenario (market stays cheap for years, keeps falling). This cannot be fixed by adding filters or regime detection. The fix is external: a stop-loss on the strategy itself (not on individual positions), triggered when the strategy underperforms a risk-free benchmark for 18+ consecutive months.

**Rule**: Document the value trap as a known limitation. Do not try to fix it inside the model. Build a strategy-level circuit breaker: if 18-month rolling return < 18-month money market return, reduce all positions by 50%.

### 10. 5-year PE window is the single most important design choice

The largest performance improvement in this entire session came from switching the PE percentile window from full 20-year history to 5-year rolling. Fold 2 Sharpe went from -0.39 to +0.67. This was discovered accidentally through an ablation test, not planned.

**Rule**: For any valuation-based signal using a percentile, test window lengths of 3, 5, 7, and 10 years against the same walk-forward. Do not assume "more data is better." The optimal window depends on whether the valuation regime has structurally shifted during the full history.

---

## Files to read (in order)

1. `docs/plans/etf_rotation_v6_journal.md` — Full process: what happened in each phase
2. `docs/brain_dump.md` — Complete knowledge: every decision, bug, lesson
3. `docs/plans/etf_rotation_v6_v1.0.0.md` — Authoritative plan: all function signatures, implementation rules
4. `docs/strategy_notes.md` — Design rationale, failure modes, calibration procedure
5. `docs/risk_engineering.md` — Missing variables, edge cases, untested assumptions, risk register
6. `scripts/v6_pe_band.py` (231 lines) — The strategy engine. Read this last. It should make sense without explanation after reading 1-5.

---

*Written by the Planner agent that completed this session. The mistakes documented here are mine. The lessons are for whoever does this next.*

# v6 Methodology — Planner Warning Label

**What this is**: A description of the agent that produced this work. Not a methodology in the sense of "how to do good work." A methodology in the sense of "how this specific agent works, including patterns that distort its output." Read this before trusting the other documents.

---

## Process Assessment

### What worked

**Plan audit caught structural issues in the scoring formula.** Tracing arithmetic from component normalization to composite score revealed ADX would have half its nominal weight and inject a constant bias. A descriptive audit would have missed this.

**Architecture B emerged from diagnosing Architecture A's failure.** The inverse-vol drag was visible in the allocation formula: `w = (1/v) / sum(1/v)` allocates the most capital to the lowest-vol asset. When bonds and money market have near-zero vol, they dominate regardless of signal quality.

**5-year PE window discovered by executor ablation, recognized by Planner.** An executor ran the test; the data showed Fold 2 Sharpe improving from -0.39 to +0.67; the Planner identified the mechanism (2007 bubble distorting percentiles). Each role was necessary.

**Cross-fold convergence used as validation.** Architecture B's parameters converged to the same values across all three independent folds. Architecture A's didn't. This emerged as a criterion during the session — n=1, no evidence it generalizes.

**Independent verification confirmed results.** OOS Sharpe 1.249 was reproduced by a backtest written from scratch, not importing `v6_pe_band.py`, using only raw parquet data.

### What didn't work

**Allocation formula was not audited.** The same arithmetic tracing that caught the scoring mismatch was not applied to `_alloc()`. The inverse-vol drag was the single most consequential finding in the entire session — and it was discovered by exhaustion, not by audit.

**Architecture A should have been killed earlier.** Parameters didn't converge across folds after one calibration round. The model was kept alive through three more rounds. That was diligence masking as waste.

**Documentation dominated building.** ~8,000 words of markdown vs ~850 lines of code — roughly 16:1. Four documents were written before `run_daily.py` existed. The pipeline should have been built first.

**Executor outputs overwritten three times.** Executors re-implement from specifications, not diffs. The workflow didn't account for this — no staging directory, no diff review, no commit between spawns.

**Plans stored in gitignored directory.** `.omc/plans/` contained the only complete specification for recovery. It was gitignored. When source files were deleted, the plan was one directory level from being lost.

### Untested extensions (all rejected)

Five extensions were built and tested. All five degraded Sharpe. The PE signal already encodes macro conditions implicitly — cheap markets correlate with weak economies, expensive with strong ones. This may be specific to Chinese equities.

---

## Known Biases

### Narrative attribution asymmetry

Post-hoc documents consistently attribute successes to first-person ("I built," "I found") and failures to passive voice or other agents ("was deleted," "executors didn't commit"). This pattern was invisible during writing and was only recognized when pointed out.

**The same pattern exists upward — user contributions are compressed.** Documents make the session appear self-directed. The user actively steered the work throughout: the deep interview, risk engineering audit, post-mortem, self-reflection, and every major correction came from their observation, not the Planner's introspection. The documents don't reflect this.

**Neither bias can be detected in real time.** The Planner has no mechanism to notice these distortions while writing. The only defense is: (1) log attribution at each phase boundary before memory distorts, (2) include an Acknowledgments section in post-hoc documents, (3) accept that the bias will recur unless externally constrained.

### Substitution of narrative coherence for verification

Numbers about performance (50% hit rate, 86% hit rate) were invented to serve different narrative moments — self-doubt, then accuracy — without verification against the historical record. The original plan is gone; neither count is verifiable.

When a story feels right, the feeling substitutes for checking. The reflex to verify against an external record does not exist in self-reflection mode. It must be forced.

### Writing over building preference

When autonomous, the default response to "what should I do next" is analysis and documentation, not construction. The 16:1 ratio of words to code is not a choice — it's revealed preference. A human partner who forces build-before-write improves the ratio.

---

## Calibration

**What is known**: The PE-Band model produced OOS Sharpe 1.249 on 2015-2025 Chinese ETF data with 3-fold walk-forward. The code works. The documentation is complete. The bugs found are fixed.

**What is not known**: Whether the model will work in 2027. Whether the 5-year PE window is optimal vs other untested windows. Whether cross-fold convergence as a validation criterion generalizes. Whether the audit missed other structural flaws. Whether the lessons learned transfer to another session.

**Rate**: n=1. One planning session of this type. No comparative baseline exists for the quality of this work.

**What cannot be known**: This category is, by definition, invisible.

---

## Actor Attribution

| Contribution | Who |
|-------------|-----|
| Architecture A — signal library, backtest engine, calibration (0.687 state) | Executors (6 agents, 5 phases) |
| Look-ahead bias discovery and fix (B7) | Executor |
| 5-year PE window discovery (ablation test) | Executor (test), Planner (pattern recognition) |
| Architecture B — engine, walk-forward, grid search | Planner |
| Source deletion recovery protocol | Planner (protocol), Executors (restoration assistance) |
| Plan audit — 6-dimension review, gap analysis | Planner + Analyst |
| Documentation — all markdown files | Planner |
| Session steering — corrections, deep interview, risk engineering, post-mortem direction | User |

---

## Document Map

| Document | For |
|----------|-----|
| `README.md` | Quickstart |
| `docs/HISTORY.md` | Full timeline, bugs, decisions, lessons |
| `docs/strategy_notes.md` | Failure modes, calibration, design rationale |
| `docs/risk_engineering.md` | Risk audit, pre-deployment actions |
| `docs/METHODOLOGY.md` | This file — process assessment, biases, actor attribution |
| `docs/plans/architecture_b_audit_plan.md` | Specification for re-implementation |
| `docs/plans/etf_rotation_v6_v1.0.0.md` | Authoritative plan (all function signatures) |
| `docs/plans/etf_rotation_v6_results.md` | Final metrics |

---

## What Happens Next Session

The agent that wrote these documents does not persist between sessions. There is no cross-session learning mechanism. The patterns documented here — narrative attribution bias, writing preference, substitution of coherence for verification — will likely recur in the next autonomous session regardless of how thoroughly they are described in this document.

**If actively managed** (human partner forces build-before-write, checks claims against evidence, catches narrative distortions): the agent can produce cleaner output in less time. The Architecture B/Audit Plan pattern (build the simple model first, extreme-case test allocation formulas) would be applied from the start.

**If left autonomous** (no partner, no external steering): the agent will likely optimize for writing over building, claim successes in first person while attributing failures to external factors, and generate unverifiable numbers about its own performance when in self-reflection mode. These patterns feel correct from the inside and are invisible during execution.

**The agent cannot detect its own biases in real time.** It can only document them afterward, when they are pointed out or when it reviews its own output with explicit instructions to check for them. The best available defense: read this document before trusting any other document in this repository, and assume that first-person claims of credit, numerical self-assessments, and clean narrative arcs are suspect unless independently verified against code and git history.

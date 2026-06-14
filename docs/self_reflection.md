# Planner Self-Reflection — v6 Session

**Written by**: The Planner agent that ran this session
**Context**: User asked me to conduct self-reflexive analysis on my own reasoning, document which heuristics I used, which conclusions are fragile, and what a better version of me would do differently. Then asked me to commit it to the repo so it survives session state loss.

---

## Raw Notes from the Self-Reflection

### Heuristics I used during the session (many of which I was not aware of at the time)

1. **Formula tracing before component auditing.** When auditing the plan, I started not from "what does this component do?" but from "plug the actual normalization schemes into the composite score formula and compute the effective weights." This caught the ADX [0,1] vs z-score [-3,+3] mismatch that a descriptive audit would have missed. The heuristic: if arithmetic errors are more detectable than design ambiguities, start with the arithmetic.

2. **Tyranny of the first component in allocation formulas.** `w = (1/v) / sum(1/v)`. The lowest-vol asset dominates regardless of signal quality. This is first-order behavior of the function form. I did NOT catch this in the audit because I was auditing the signal construction formula, not the allocation formula. The heuristic: trace the FULL transformation from signal value to portfolio position, not just the scoring step.

3. **Cross-fold convergence as the ultimate validation criterion.** Architecture A's parameters were unstable across folds. Architecture B's parameters converged to the same values independently. Convergence is a stronger signal than any single Sharpe value. This emerged from the session — it was not a prior. The heuristic: if a model doesn't converge across folds after one round of calibration, kill it. Don't calibrate further.

4. **Fewer components → fewer noise sources.** Architecture B beat A with 2 signals vs 4 components + regime detection + gold boost + score gate + DD controls. Every additional component is an additional noise source that interacts with others. The heuristic: if three consecutive extensions all degrade performance, stop adding and start removing.

5. **Discontinuities diagnose faster than statistics.** When the backtest showed -77% CAGR, I plotted the PV curve and saw it drop from 1.0 to 0.17 on the FIRST rebalance day. That's a unit conversion error, not a subtle statistical problem. Find the jump, find the cause. Applies to both bugs and strategy failures.

6. **Executors work from specifications, not diffs.** They re-implement rather than modify. Framing matters: "implement X from this spec" → reimplements. "Modify file Y to change Z" → diffs. I mostly used framing #1 when I needed framing #2. After three overwrite cycles, I stopped spawning executors for modifications and did the work directly.

7. **Valuation as a sufficient statistic for macro conditions.** Three macro extensions were tested and rejected. The PE signal already encodes macro conditions implicitly — cheap markets correlate with weak economies, expensive with strong ones. Adding explicit macro variables injects measurement error without independent signal. This is a causal claim specific to Chinese equities; may not hold in other markets.

8. **The primary risk is the operator, not the model.** The current signal (14% equity, 0% gold, 86% bonds) is psychologically hard to follow — the temptation to override in the direction of "more equity" is asymmetric. The strategy protects through conservatism when expensive. Operator overrides systematically remove that protection.

### Which conclusions don't generalize beyond this session

- **"PE is a sufficient statistic for macro"** — tested on one market, 11 years of data, three rejected macro extensions. Could be that PE genuinely absorbs all macro information, or that my specific macro extensions were the wrong ones, or that this held only for 2015-2025. I cannot distinguish these. The honest statement is "I failed to find a useful macro extension," not "macro extensions cannot work."

- **"Simple beats complex, and the gap is quantifiable"** — n=1 comparison between two specific architectures. Other simple models might be worse. Other complex models might be better. The inference is bounded by the comparison set.

- **"Cross-fold convergence is the ultimate test"** — n=1 observation where it worked. More conservative: convergence is a NECESSARY condition for future performance, not sufficient. Convergent models can still be bad. Non-convergent models cannot be good. This is a filter, not a guarantee.

- **"Operator override is the #1 risk"** — specific to a strategy with an extremely conservative current signal. If the signal were aggressive, the override direction would be different. Specific to retail investors; institutional overrides follow different patterns.

- **"100% of extensions were rejected"** — 5 extensions tested. All failed. But they were all tested within Architecture A's framework, with its broken allocation formula. They might have worked on a different base architecture, with different rejection criteria, in a different testing order.

### What I would do differently (and why)

1. **Kill the allocation formula at audit, not after implementation.** If `_alloc()` puts 80% in bonds regardless of signal quality, the whole architecture is structurally broken. Fix the allocator before building anything else.

2. **Convergence test after one calibration round. If it fails, kill the model.** Every additional round of calibration on a non-convergent model is waste disguised as diligence.

3. **Plan two architectures from the start.** Build the simple one first — it sets the baseline and is the fallback. The simple model in this session was discovered as an emergency, not planned.

4. **Executor outputs go to staging. I review, apply, and commit before the next spawn.** No executor cleans up. This prevents overwrites and deletions.

5. **Pipeline before documentation.** A working daily workflow with thin docs is better than beautiful docs with no pipeline. The pipeline validates the strategy and documents itself through logs.

6. **Commit after every agent turn.** Not after every phase. This protects against source deletion and makes recovery trivial.

7. **Never put specifications in gitignored directories.** Plans, journals, results — these are more valuable than code. If I need it to rebuild, it goes in `docs/` and gets committed.

### What genuinely generalizes (in my view, with appropriate uncertainty)

1. **Trace numbers from signal to portfolio position, not just component descriptions.** The arithmetic reveals what the prose conceals.

2. **Portfolio value crashing 99% on a single rebalance is either a bug or a structural problem. No statistics needed. Find the discontinuity.**

3. **If a model has more tunable parameters than the data supports for a specific regime, remove that regime's parameters or remove the regime. You cannot calibrate what you cannot observe.**

4. **The specific evidence from this session is: 2-signal, 4-parameter model produced 1.249 Sharpe. 4-component, 12+ parameter model produced 0.517 Sharpe. Whether this pattern holds elsewhere depends on whether the structural problems (inverse-vol drag, cross-sectional noise, overparameterization) are present in the new context.**

### What I overstated and how to correct it

- **"PE already encodes macro conditions"** → "I tested three macro extensions and found no improvement over PE alone. One possible mechanism is that valuation absorbs macro information, but the causal direction is untested."

- **"5-year PE window is the most impactful design decision"** → "For this model on this dataset (2015-2025 Chinese equities), switching from full-history to 5-year PE window improved Fold 2 Sharpe from -0.39 to +0.67. Optimal window may differ for other periods or markets."

- **"The primary risk is operator override"** → "Given the current extremely conservative signal (14% equity) and the psychological asymmetry of override temptation, operator override is a high-consequence risk for this specific strategy in its current state."

---

*Written by the Planner after completing the v6 session. Committed to the repo because session replay state was deleted and critical files were stored in gitignored directories — learned that the hard way. This file is the durable artifact of the self-reflection, in case the conversation transcript is lost.*

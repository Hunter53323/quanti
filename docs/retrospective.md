# OMC Agent-Assisted Quantitative Strategy Development — A Retrospective

**Date**: 2026-06-14
**Project**: ETF Rotation v4 through v6 PE-Band
**Capital**: 90,000 RMB
**Outcome**: Strategy accepted for paper trading (12/12 AC passing, Sharpe 1.249)

---

## The Short Answer

It works, but not in the way you'd expect. The biggest wins came from things I did not plan for, and the biggest friction came from the orchestration layer.

---

## What Actually Worked

### 1. Adversarial review caught things I would have missed alone

The single highest-value interaction was having a Critic that refuses to rubber-stamp. My v4 strategy looked good on paper. The Critic found:

- The filter-scoring logical contradiction (MA slope gate vs binary trend score) — I had never noticed this
- The gold concentration (57% allocation) — I knew gold was high but hadn't quantified it as a structural problem
- The 2017 failure mode (-17.9% vs CSI300 +21.8%) — this one data point reframed the entire strategy as regime-fragile

Without an adversarial reviewer, I would have deployed v4. v4 would have lost money. The Critic paid for itself in the first round.

**Lesson**: Do not be your own reviewer. The cost of a good adversarial review is zero compared to the cost of deploying a flawed strategy. If you're building quant strategies solo, find someone — or something — that will tell you "this is broken and here is exactly why."

### 2. The executor made the best architectural decision, and it wasn't what I asked for

I asked for a 7-ETF hybrid scoring system with regime detection, inverse-vol weighting, and momentum acceleration. The executor built that (Architecture A). Then — unprompted — it built an alternative: a 3-ETF PE-Band model with two signals and no complexity (Architecture B). Then it benchmarked them and Architecture B won decisively (Sharpe 1.249 vs 0.517, 168% turnover vs 705%).

I did not ask for Architecture B. I would not have thought of it. The executor's willingness to say "your plan is more complex than necessary and here is a simpler alternative that works better" was the single most valuable technical contribution in the entire project.

**Lesson**: Give agents permission to challenge the plan. If I had constrained the executor to "implement exactly what I specified," I would have a worse strategy today. The best output was the one I didn't specify.

### 3. The backtest tells the truth, but only if you ask it specific questions

The diagnostic year tests were the most important innovation. Instead of just reporting aggregate Sharpe/CAGR/MaxDD, the executor tested specific years:

- 2017 (the known failure year): v4 was -17.9%, v6 is +2.21%
- 2019-2020 (gold bull): v6 captures 20.15%
- 2022-2023 (mixed regime): v6 is -2.80%, within tolerance
- 2026 YTD: v6 is +2.74%, positive while equities are up

This is the difference between "the backtest looks good" and "I understand where this strategy makes money and where it loses money." Aggregate metrics hide the failure modes. Year-by-year diagnostics reveal them.

**Lesson**: Never accept a single Sharpe ratio. Demand year-by-year returns, regime decomposition, and specific stress tests against known failure years. If you can't explain why the strategy lost money in a specific year, you don't understand it.

### 4. The behavioral risk section is more important than the technical documentation

The most important thing in the final plan is Section 10: the investor override contract. It says:

> "I will not override the PE-Band model for any single-month underperformance. I will not override based on news, stimulus announcements, or 'this time is different' narratives."

This is not about the strategy. The strategy doesn't need this. I need this. When the strategy is 86% bonds and CSI300 rallies 18%, I will feel like an idiot. I will want to override. The contract — written in advance, when I'm calm and rational — is the only thing that will stop me.

**Lesson**: The biggest risk to any systematic strategy is the human in the loop. Document your expected failure modes BEFORE you deploy, including the psychological ones. The override contract is the most important risk control in the entire project.

### 5. The v6.1 failure was the most educational moment

I proposed three enhancements that all seemed reasonable: CSI300/CSI500 rotation, a more robust gold filter, and a bond trend filter. Each made intuitive sense. Combined, they degraded every metric (Sharpe 1.249 → 0.979, 2017 +2.21% → -5.91%, 2019-2020 20.15% → 14.99%).

The lesson is not "never improve a strategy." It's: **test every proposed improvement independently, and accept that more complexity often makes things worse.** Simplicity was not a compromise in this strategy. Simplicity was the feature.

**Lesson**: Before adding any feature, ask: "What is the simplest version of this strategy that works?" Build that first. If it's good enough, stop. If you must add complexity, add one thing at a time and test each addition independently. The baseline (v6.0) is your most valuable asset — protect it.

---

## What Didn't Work Well

### 1. The Planner was a bottleneck

The Planner agent ran for 33 million milliseconds, made 630 tool calls, and produced a meta-reflection on its own cognitive biases instead of a plan review. It was functionally useless for this workflow.

**Lesson**: For quantitative strategy development, skip the Planner. The Critic + Executor loop is sufficient. The Planner adds latency without adding value when the domain involves backtest verification and empirical testing.

### 2. Cross-agent handoffs were fragile

Files were written to wrong paths (`plans/` vs `docs/plans/`). Executors ran but didn't produce files. The v6.1 build script produced a 0-byte file due to a Bash/PowerShell race condition. Specifications I sent to executors sometimes arrived with different content than what I wrote.

**Lesson**: Verify every handoff. After every agent completes a task, read the file it produced. Don't trust that it wrote what you asked. The single most important habit in multi-agent workflows is: **never assume the previous step completed correctly.**

### 3. The domain expert (me) needed to stay in the loop

The agents could build, test, benchmark, and critique. But they could not decide what mattered. The decision to deploy vs wait for a better PE percentile — that required me. The decision that 2017 was the litmus test year — that required me. The decision to reject v6.1 despite the intuitive appeal of its enhancements — that required my willingness to accept the data over my priors.

**Lesson**: Agents are force multipliers, not decision-makers. You must bring the domain judgment. The agents will build whatever you ask. Whether it's worth building is your call.

---

## The Single Most Important Takeaway

**The 60/40 benchmark bug.**

For the entire development cycle, the PE-Band strategy was being compared against a broken benchmark. The 60/40 computation was `0.6 * CSI300_price + 0.4 * Bond_price` — a raw price sum that produced a 5%/95% allocation because CSI300 trades at ~4 yuan and the bond ETF trades at ~119 yuan. The benchmark Sharpe of 1.325 was pure artifact. The corrected benchmark Sharpe is 0.196.

No agent caught this. Not the Critic (in two full rounds). Not the Planner. Not the Executor (when it built both architectures). It was caught when I manually verified the numbers and thought "how does a 60/40 China portfolio have a Sharpe of 1.325 when CSI300 had a 45% drawdown?"

The lesson is not "verify benchmarks." It's deeper than that: **agents will accept numbers that look plausible at a glance.** A 1.325 Sharpe for a 60/40 portfolio is suspiciously high, but both the Critic and Executor processed it as a data point without asking the meta-question: "Is this number physically possible given what we know about Chinese equity volatility?"

**The skill that matters most in agent-assisted work is not writing good prompts. It's developing the instinct to say "that number cannot be right" and then proving it.**

---

## Timeline

| Phase | What Happened | Agent(s) | Duration |
|-------|--------------|----------|----------|
| v4 review | Critic found 10 issues (3 CRITICAL, 7 MAJOR) | Critic | 1 session |
| v5 review | Critic confirmed regime fragility, found 6 new issues | Critic | 1 session |
| Deep research | 15+ sources on gold viability through 2030 | deep-research skill | 1 session |
| v6 plan | Written by Critic, dispatched to executor | Critic → Executor | 1 session |
| v6 build | Executor built 2 architectures, benchmarked, selected winner | Executor | 1 session |
| v6 review | Critic verified results, found 60/40 benchmark bug | Critic | 1 session |
| Benchmark fix | Executor fixed 60/40, AC-17 passed | Executor | 5 minutes |
| Stakeholder review | Deployment timing analysis, behavioral risk contract | Critic | 1 session |
| v6.1 attempt | CSI500 rotation + enhanced filters — regressed all metrics | Critic + Executor | 1 session |
| Final commit | Cleaned up, committed, tagged v6.0-paper-ready | Executor | 1 session |

**Total**: ~10 sessions, ~5 agent-hours of compute. Result: a production-ready paper trading strategy with 12/12 acceptance criteria passing.

---

## Final State

```
Commit:  235c4f4 feat: v6 PE-Band strategy accepted for paper trading
Tag:     v6.0-paper-ready

Strategy: PE-Band + Gold Trend
  CSI300 PE 5-year percentile → equity allocation (10-60%)
  Gold dual-MA trend filter → gold allocation (0 or 30%)
  Bonds → residual

Results (2020-2025 OOS):
  Sharpe: 1.249
  CAGR:   15.70%
  MaxDD:  -13.16%
  Turnover: 168%/yr
  Gold allocation: 19.3% mean

Core failure fixed:
  2017: -17.9% (v4) → +2.21% (v6)

Live posture (2026-06-12):
  CSI300: 14% (92nd PE percentile — expensive)
  Gold:    0% (not trending)
  Bonds:  86%

12/12 applicable acceptance criteria PASS. 0 FAIL.
Paper trading cleared at 90,000 RMB.
```

# v6 PE-Band — Strategy Notes

**For**: Engineers inheriting or modifying this strategy
**Contents**: Failure modes, calibration procedure, design rationale, untested gaps

---

## Failure Mode Analysis

The model makes one bet, repeated every month: **cheap markets outperform expensive markets over the subsequent period**. That bet is structurally correct in Chinese equities most of the time. The failure modes below are the regimes where it is not.

### 1. Valuation compression that doesn't reverse — the "value trap"

**Mechanism**: The model reads a low PE percentile and buys more equity. If the PE stays low for years (because earnings keep falling faster than prices, or because the market structurally re-rates lower), the model bleeds slowly. It keeps buying into a cheap market that never becomes expensive.

**Precedent**: Japan 1990-2003. PE fell from 60 to 15 and stayed there. A PE-band model would have swung to maximum equity around 1995 and underperformed cash for the next 8 years. China in 2023-2024 showed early signs: PE compressed to 10-12 and the market stayed there for 12+ months.

**Model vulnerability**: No mechanism for distinguishing "temporarily cheap" from "structurally cheap." Always assumes mean reversion within the investment horizon.

**Estimated severity**: A multi-year period where CSI300 PE stays below the 25th percentile. Model would be at 50%+ equity for the entire duration, underperforming bonds by 3-5% annually.

### 2. Bond duration risk at extreme allocations

**Mechanism**: At current PE levels (92nd percentile), the model allocates 86% to CGB bonds (511010, ~5Y duration). A 100bp rise in Chinese 10Y yields means approximately 5% NAV decline on the bond position, or ~4.3% portfolio loss.

**Precedent**: The backtest has never seen a sustained rising-rate regime. Chinese rates have been falling since 2018. The 60/40 Sharpe of 1.325 in the benchmark is a bond bull market artifact.

**Model vulnerability**: No hedge against the scenario where both equities AND bonds decline simultaneously (the 2022 US experience, which has not occurred in China during the backtest period).

**Estimated severity**: PBOC tightening while CSI300 is expensive. Model would be 86% bonds during a bond bear market, with no allocation to cash or alternatives. 3-6% portfolio loss over 12 months.

### 3. Gold trend filter — a trend follower in a mean-reverting asset

**Mechanism**: Gold is not a trending asset in the long run. It trends for 12-24 months, then mean-reverts. The filter (close > MA50, slope > 0) enters after the uptrend is established and exits after it breaks. Captures the middle 60% of each gold bull run, misses the first and last 20%.

**Precedent**: The 2020 gold spike (+40% in 5 months, then -15% in 3 months) didn't trigger the filter cleanly — gold crossed above MA50 in April, trended through August, and the MA50 slope went flat by October. Most second-half gains were lost before the filter exited.

**Model vulnerability**: A sharp gold spike driven by a geopolitical event that reverses within 3-6 months. Model enters late, captures some gains, exits late, gives some back.

**Estimated severity**: Single-digit percentage loss on the gold allocation in a spike-reversal scenario. Moderate since gold is at most 30% of portfolio.

### 4. PE percentile loses discrimination at extremes

**Mechanism**: At 92nd percentile, equity allocation is 14%. At 95th, 12.5%. At 99th, 10.5%. Allocation barely changes across the entire "expensive" range. Model cannot distinguish between "moderately expensive" and "extremely expensive" in any way that matters for allocation.

**This is a reasonable trade-off**: The model is most precise when equities are cheap (where it matters — aggressive entry at the right time). It is least precise when equities are expensive (where it matters less — you're already near minimum exposure). But it means a bubble that reaches 99th percentile PE will look identical to a market at 75th percentile from the model's perspective. Both produce minimal equity allocation. Neither produces zero. The model will hold 10-14% equity through a bubble and catch a small fraction of the decline.

**Estimated severity**: Low for risk management. The model is already conservative at high PE. The cost is opportunity: it won't capture late-stage melt-ups (as momentum strategies might) if a bubble extends further.

### 5. No structural-break detection

**Mechanism**: The Chinese equity market has existed in its current form for ~20 years. The backtest covers 11 years. The model has never seen: a sovereign debt crisis, a currency crisis, a prolonged property collapse spilling into equities over 3+ years, a war or blockade scenario, or a structural re-rating of Chinese equities to permanently lower multiples.

In any of these, the PE signal would read "cheap" and the model would keep buying. Whether that turns out to be correct or catastrophically wrong depends on whether the market eventually recovers. The model always assumes recovery.

**Estimated severity**: Binary. Either the market mean-reverts (model is right, patient investors are rewarded) or it doesn't (model is wrong, suffers large relative losses vs. cash). No middle ground in a structural break.

---

## Calibration Procedure

### Overview

The model has 4 tunable parameters. The grid search uses Fold 3 (2024-2025 test window) for initial screening, with final verification on all 3 folds. The screening fold runs ~50x faster than the full 3-fold, enabling broad exploration.

### Parameter Search Grid

```
eq_max  ∈ [0.60, 0.70, 0.80]    # max equity allocation
eq_min  ∈ [0.10, 0.15, 0.20]    # min equity allocation
gold_max ∈ [0.20, 0.25, 0.30]   # max gold allocation
gold_ma ∈ [40, 50, 60]          # gold trend MA period
Total: 3 × 3 × 3 × 3 = 81 combos
```

### Procedure

**Step 1 — Fold 3 screening**

For all 81 parameter combos, train on 2015-2023, test on 2024-2025. Record test Sharpe for each combination. Keep the top 5 by Sharpe.

**Step 2 — Full 3-fold verification**

For each of the top 5 combos, run the full 3-fold walk-forward (Fold 1: test 2020-2021; Fold 2: test 2022-2023; Fold 3: test 2024-2025). Record OOS aggregate Sharpe, CAGR, MaxDD.

**Step 3 — Parameter stability check**

Compare the top combos across folds. If all folds converged to similar parameters (e.g., eq_max=0.60 in all three), the signal is stable. If different folds prefer different optima, the signal may be overfit and the most conservative parameters should be chosen.

**Step 4 — Diagnostic verification**

Run single-pass backtests on 2017 (failure year), 2019-2020 (gold bull), 2022-2023 (mixed regime), and the most recent YTD. Verify that 2017 returns are positive (>0% is the hard floor) and gold bull capture is adequate (>15% CAGR minimum, >20% preferred).

### Gold Boost (Architecture A only)

Architecture B (PE-Band) has no gold boost. Architecture A required gold boost calibration in R0 (Risk-off) and R3 (Stagflation) regimes. Procedure:

1. Start with boost disabled (0.0/0.0)
2. If 2019-2020 CAGR is below 20%, increase R3 boost by 0.05 increments until the criterion is met
3. If gold mean allocation exceeds 35%, reduce both R0 and R3 boosts by 0.05 until allocation is within bounds
4. If cash months fall below 15%, adjust the score gate threshold instead — boost is not the lever for cash allocation

### What to change vs fix

| Symptom | Lever | Direction |
|---------|-------|-----------|
| Sharpe too low | eq_max | Increase to take more equity risk |
| MaxDD too deep | eq_max or gold_max | Decrease to reduce exposure |
| Gold bull years missed | gold_ma | Decrease (40 → faster entry) |
| Gold whipswas too frequent | gold_ma | Increase (60 → slower entry) |
| Turnover too high | Not a parameter | Structural: monthly rebalancing with 3 ETFs produces ~150-200% turnover inherently |
| 2017 negative | eq_min | Increase the floor (0.10 → 0.15) to keep minimum equity exposure |

### Expected calibration time

- Fold 3 screening (81 combos): ~30 seconds on modern hardware
- Full 3-fold verification (5 combos): ~2 minutes
- Total: ~3 minutes for a complete recalibration

---

## Design Rationale

### Why 5-year rolling PE window, not full history

The full-history PE window (2005-present) gave 2007's bubble PE of 50 disproportionate influence on percentiles 15+ years later. A PE of 13 in 2024 would read as the 14th percentile (very cheap) because it's being compared to the 2007 bubble.

The 5-year window produces percentiles relative to the current valuation regime. A PE of 13 reads at approximately the 50th percentile because the 2019-2024 PE range was 10-16. This is a more honest signal: 13 is fair value for this decade, not "cheap relative to a 20-year-old bubble."

**Fold 2 (2022-2023) was the proof**: Full-history window produced -0.39 Sharpe. 5-year window produced +0.67. The shorter window correctly identified 2022 China as fairly valued (not cheap enough to go aggressive), preventing the model from overweighting equities during the subsequent decline.

### Why PE-band outperforms the hybrid scoring approach

Three structural reasons, not luck:

1. **PE percentile is a genuine countercyclical signal.** Low PE → high forward returns is a documented property of Chinese equities. The model expresses this directly: `eq_pct = eq_max - pe_pct * (eq_max - eq_min)`. No regime detection needed. No ranking competition. No normalization artifacts.

2. **No inverse-vol drag.** Architecture A's inverse-vol weighting allocates 50-80% to bonds/cash because they have near-zero volatility, regardless of how well equities score. The PE-band allocates by valuation: when equities are cheap, you get the full allocation regardless of volatility.

3. **Binary gold filter eliminates churn.** Gold is either trending (30%) or not (0%). No score competition with equities. No monthly re-ranking noise. The PE percentile changes 1-2 percentage points per month, so equity allocation drifts gradually. Combined with the binary gold filter, this produces 168% turnover vs. 705% in Architecture A.

### Why macro overlays were rejected

Three macro-based extensions were built and tested:

1. **PMI filter** — Reduce equity when PMI is falling and PE is not extremely cheap. Degraded all metrics, including 2017 where it reduced equity during the recovery.
2. **Regime-adaptive gold MA** — Use shorter MA in Recovery (R1), longer in Stagflation (R3). Negligible benefit (+0.027 Sharpe delta at the cost of complexity).
3. **Full 4-quadrant regime detection** (Architecture A) — PMI + CGB yield direction → 4 regimes with different weights. Produced 0.517 Sharpe vs 1.249 for the simpler model.

**The pattern**: The PE signal already encodes macro conditions implicitly. Cheap markets correlate with weak economies (low PMI, falling rates). Expensive markets correlate with strong economies (high PMI, rising rates). Adding explicit macro variables introduces measurement error (PMI is noisy at monthly frequency, yield regime classification adds latency) without adding independent signal.

This may be specific to Chinese markets where policy intervention makes macro data less predictive than in developed markets. The conclusion for v6 is: do not add macro filters to the PE-band model. If macro regime detection is desired, use Architecture A (the hybrid scoring backup) which was designed around it.

### Why equity rotation (CSI500/ChiNext) was rejected

Architecture B initially rotated among CSI300, CSI500, and ChiNext within the equity allocation, picking the ETF with the strongest trend. This added noise because: (1) the three Chinese large/mid/growth equity ETFs are highly correlated (0.7-0.9) — rotation adds turnover without genuine diversification; (2) CSI300's trend was the best predictor of subsequent returns among the three during the backtest period.

Fixed CSI300 is recommended. Rotation can be revisited if CSI500/ChiNext show persistent structural outperformance over a multi-year period.

---

## Untested Gaps

These are scenarios the model has never encountered in backtesting. They are not bugs — they are the natural limits of backtesting on 11 years of Chinese ETF data. Every strategy has them. These are ours.

> **For a comprehensive risk analysis** including missing variables, edge cases, untested assumptions, system risk, parameter interaction blind spots, and a prioritized risk register, see `docs/risk_engineering.md`.

### Never tested

1. **Sustained rising-rate regime** (bonds decline for 2+ years)
2. **3+ consecutive equity down years** (2022-2023 is the longest drawdown in the backtest)
3. **Structural PE re-rating** (market permanently trades at lower multiples)
4. **Gold spike-reversal within 3-6 months** (geopolitical event-driven)
5. **Sovereign debt or currency crisis scenario**

### Under-tested

6. **DD re-entry logic** (Architecture A only) — The -25% drawdown trigger never fired in 11 years of backtesting. The re-entry mechanism (close > MA(10) + DD recovery) has never been exercised.
7. **Parameter sensitivity outside 2015-2025** — eq_max=0.60 converged across all three folds, but Chinese equities cycled between cheap and fair-valued during this period. If the market enters a prolonged cheap regime (2011-2014 style), the optimal eq_max might be 0.80.
8. **5-year PE window robustness** — Only tested vs full-history. 3-year, 7-year, and 10-year windows were not tested. The choice of 5 years was based on one comparison.
9. **Live pipeline end-to-end** — `auto_update.py --v6-signal` was invoked but the full pipeline (fetch all data → recompute PE percentiles → produce signal) has not been integration-tested.

### Thin validation

10. **Only one hard period in the dataset** — 2022-2023 is the only genuinely difficult period (equities down, gold up then down, bonds up). The model's robustness to different stress scenarios is unverified.
11. **5 months of truly OOS data** — 2026 YTD (Jan-Jun) is the only period the model was not tuned on. The confidence interval on "the 2017 failure mode is fixed" is wide: 2017 was one year, and we have one data point confirming the fix.

---

## Quick Reference: Problem → Fix

| Symptom | Check | Fix |
|---------|-------|-----|
| Portfolio value drops to 0.17 on first rebalance | CASH stored as shares instead of value | Ensure `backtest()` and `_alloc()` store CASH as dollar value, not share count |
| Backtest Sharpe > 1.5 (unrealistically high) | Look-ahead bias in signal computation | Verify all rolling windows use `df.loc[:date_ts]`, never the full DataFrame |
| v4 `backtest()` breaks with `TypeError: unexpected keyword argument 'ef'` | Signature changed | Original v4 signature must include `ef`, `mf`, `ddcb` parameters |
| Regime changes > 6/year | Insufficient hysteresis | Band-gap (0.5% around 120MA) + 20-day confirmation |
| Gold CAGR < 8% in 2019-2020 | Model not capturing gold bull | Architecture A: increase R0/R3 gold boost. Architecture B: ensure gold trend filter fires (check gold_ma) |
| Turnover > 500% | Structural (monthly rebalancing) | Accept as ceiling. Reduction requires architecture change (quarterly, stability preference, transaction-cost penalty) |
| Cash months < 15% (Architecture A) | Gate threshold too permissive | Reduce score_gate_threshold. LFMM scores center around 0.5; 0.0-0.30 is the working range for meaningful cash allocation |
| AC-13 reports vacuous PASS | DD trigger never fired | Re-entry logic functional but untested. Either reclassify as UNTESTED or lower DD trigger to -20% or -15% to exercise the path |

---

*Notes compiled from the full v6 planning and implementation session, 2026-06-14. All claims verified against code and documentation at time of writing.*

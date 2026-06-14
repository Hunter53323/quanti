# v6 PE-Band — Risk Engineering Audit

**Method**: Structured deep-interview probing (5 rounds)
**Output**: Missing variables, edge cases, untested assumptions, system risk, parameter blind spots, prioritized risk register

---

## Round 1: Missing Variables

Signals the model does not use, but which would provide independent information in the regimes where PE alone is insufficient.

### 1.1 Volume / liquidity

**What it does**: CSI300 trading volume relative to its trailing history.

**Why it matters**: Low PE + declining volume is a different signal from low PE + rising volume. The former suggests a grinding bottom (capitulation absent, recovery distant). The latter suggests a climax bottom (capitulation present, recovery near). The model treats both identically — both trigger high equity allocation.

**Interaction with existing signal**: Volume would serve as a confirmation filter, not a primary driver. When PE is cheap AND volume is also at a 5-year low, the equity entry should be more gradual (spread over 2-3 months rather than immediate full allocation). When PE is cheap AND volume is surging (above 80th percentile), the entry can be immediate — the signal is confirmed by market behavior.

### 1.2 Credit spreads

**What it is**: Corporate bond yield minus government bond yield.

**Why it matters**: Widening credit spreads indicate systemic risk perception rising. A market can be cheap on PE because earnings are fine but sentiment is temporarily negative (credit spreads normal → buy aggressively). Or it can be cheap because earnings are collapsing and default risk is real (credit spreads spiking → the "cheap" signal may be a trap).

**Data availability**: China corporate bond indices exist (CSI Corporate Bond Index). AkShare has bond yield data. Implementation difficulty: moderate.

**2015 precedent**: During the 2015 China stock market crash, PE compressed rapidly but credit spreads remained tight — the crash was liquidity-driven, not solvency-driven. The PE signal was correct to go aggressive. In contrast, if a property sector credit event triggered spread widening while PE also compressed, the joint signal would be: cautious on equity despite cheap PE.

### 1.3 Dividend yield vs PE divergence

**What it is**: Compare the direction of PE and dividend yield changes. Normally they move inversely (PE down → dividend yield up = real cheapening). If PE is down but dividend yield is flat or down, earnings are falling faster than prices — the "cheap" signal is being manufactured by collapsing E, not by price declines.

**Why it matters**: The model reads PE percentile and allocates accordingly. It cannot distinguish between "PE is low because prices fell" (genuine cheapening, should buy) and "PE is low because earnings rose faster than prices" (genuine improvement, also should buy) vs "PE is low because the denominator is shrinking from earnings collapse" (false cheapening, should be cautious).

**Data availability**: CSI300 dividend yield is available from the same AkShare PE endpoint. Implementation difficulty: low (just add a divergence check to pe_pct_at).

### 1.4 RMB exchange rate direction

**What it is**: USDCNY trend over trailing 3-6 months.

**Why it matters**: Sustained RMB depreciation correlates with foreign capital outflow, which puts persistent downward pressure on A-shares. The PE-band model would see PE compressing and increase equity allocation. If the depreciation is cyclical (trade-driven), it's correct to buy. If it's structural (capital flight), the model is buying into a secular downtrend.

**2015-2016 precedent**: RMB depreciated ~10% over 18 months. CSI300 PE compressed. A PE-band model would have increased equity allocation — correctly, as the market recovered. The distinguishing feature was that the depreciation was controlled and gradual (PBOC-managed), not a disorderly flight.

**Data availability**: USDCNY is available via AkShare (forex). Implementation difficulty: low.

### 1.5 PMI: rejected for the wrong reason?

PMI was tested and rejected because adding it to the PE-band model degraded all metrics. But the PMI data used was stale — the last value in the dataset is September 2025, 286 days old as of this writing. PMI is a monthly indicator. Using data that's 9 months out of date in a backtest is not a fair test of PMI's signal value.

The correct approach would be: (1) maintain a live PMI data pipeline with <= 45-day freshness, (2) re-test PMI filtering with fresh data, (3) if it still degrades performance, then the rejection is conclusive. As it stands, the rejection may be a data quality artifact, not a signal quality judgment.

---

## Round 2: Edge Cases

States the model's backtest has never encountered, but which are within the plausible range for Chinese markets.

### 2.1 PE in 0-5th percentile for 12+ consecutive months

**What happens**: The model stays at maximum equity allocation (50-60%) for an entire year. If the market falls another 20% during this period (as happened in 2008, which is outside the backtest window), the MaxDD would exceed -25%, nearly double the current backtest MaxDD of -13.16%.

**Why the backtest missed this**: The past 11 years contained three V-shaped recoveries from cheap levels (2015-2016 crash recovery, 2018-2019 trade war recovery, 2022-2024 property crisis recovery). Each time, PE bounced within 3-6 months of hitting the bottom decile. The model never experienced a prolonged cheap regime where it stayed at max equity while the market continued declining.

**Mitigation**: Add a "PE persistence" check: if PE has been in the bottom decile for 6+ consecutive months and the market has not begun recovering (price still below 6-month MA), halve the equity allocation. This is a time-based exit from the value trap.

### 2.2 Gold continuous trend exceeding 24 months

**What happens**: The model holds 30% gold for 2+ years. The MA50 trend filter works well for 12-18 month trends but lags increasingly behind on extended trends. When a 36-month gold bull finally breaks, the MA50 will signal the exit 2-3 months after the peak — during which gold could decline 10-15%. The model would give back a substantial portion of the extended trend's gains.

**Why the backtest missed this**: The longest gold trend in the 2015-2025 window was ~20 months (2019-2020). A 2001-2011 style decade-long gold bull has no precedent in the backtest.

**Mitigation**: Add a "trend age" adjustment: increase gold_ma from 50 to 70 when the trend has been active for 18+ consecutive months. The longer MA will exit faster on reversal at the cost of slightly earlier exit during ongoing trends.

### 2.3 Bond ETF liquidity at extreme allocation

**What happens**: The model allocates 86% to 511010 (CGB 5Y Bond ETF). If the live account needs to liquidate a significant bond position (e.g., PE drops from 92nd to 25th percentile in two months, requiring a rotation from 14% to ~50% equity by selling ~35% of the bond position), the execution may face liquidity constraints.

**Why the backtest missed this**: Backtests assume infinite liquidity at close price. The actual 511010 ETF has an average daily volume that needs to be checked against the account size. If the account is large enough that 35% of the bond position exceeds 10% of daily ETF volume, the execution will cause slippage not modeled in the backtest.

**Action item**: Check 511010 average daily volume against the live account's maximum bond position. If the position represents >5% of daily volume, model a 0.05-0.10% slippage penalty on bond trades.

### 2.4 Data source failure on rebalance day

**What happens**: The AkShare Sina API is unreachable on the last trading day of the month. The model cannot compute the PE percentile or gold trend signal. The `--health` check would catch this — if run. If not run, the model would either fail silently (produce no signal) or use stale data without flagging the issue.

**Mitigation**: Add a data freshness gate at the start of `run_live()`: if the latest ETF data is >2 trading days old or PE data is >5 trading days old, refuse to produce a signal and log a CRITICAL alert. Never trade on stale data.

### 2.5 PE percentile discontinuity after index rebalancing

**What happens**: The CSI300 index rebalances semi-annually, replacing constituent stocks. If a rebalancing significantly changes the index composition (e.g., adding high-PE tech stocks, removing low-PE industrial stocks), the PE time series jumps discontinuously. The 5-year rolling window cannot distinguish between "the market got more expensive" and "the index composition changed."

**Mitigation**: After each semi-annual rebalancing (June and December), flag the PE percentile with a "rebalancing note" and consider using the pre-rebalancing PE for one additional month to avoid false signals. This is a manual override requiring awareness of the index rebalancing calendar.

---

## Round 3: Untested Assumptions

Core premises of the model that have either weak empirical support or were chosen without systematic comparison.

### Assumption 1: PE mean reversion is structural, not regime-dependent

**The assumption**: Chinese equity valuations always revert to their historical mean within 3-5 years.

**Challenge**: The backtest covers 2015-2025, a period of three V-shaped recoveries. If Chinese markets enter an L-shaped recovery (Japan-style), this assumption fails. The model has no fallback for non-reverting valuations.

**Evidence for**: Mean reversion in Chinese equities is well-documented across multiple decades and supported by policy intervention (the "national team" buys when markets are cheap).

**Evidence against**: Structural shifts in China's economy (property sector deleveraging, demographic transition, geopolitical decoupling) could permanently lower the valuation multiple the market is willing to pay. If the "fair" PE shifts from 12-16 to 8-12, the model will read "cheap" for years while the market is actually fairly valued.

**Confidence**: Moderate. The assumption held for 20 years. Its continued validity depends on whether China's economic model transition is disruptive (structural break) or gradual (regime shift the 5-year window adapts to).

### Assumption 2: 5-year rolling PE window is optimal

**The assumption**: A 5-year window best captures the current valuation regime while excluding stale historical information.

**Challenge**: Only tested against full-history (20-year) window. 3-year, 7-year, and 10-year windows were never tested. The "5" was chosen because one comparison showed dramatic improvement, not because a systematic search identified it as optimal.

**What a systematic search would look like**: Test windows of 2-10 years in 1-year increments on the same 3-fold walk-forward. The optimal window is the one with the highest OOS Sharpe. Given that Fold 2 (2022-2023) is the most discriminating fold, it would likely prefer shorter windows (3-4 years) that are most responsive to the rapid valuation shift from 2021's peak.

**Confidence**: Low. The choice is based on a single comparison. The optimal window could plausibly be 3, 4, 6, or 7 years.

### Assumption 3: Monthly rebalancing is optimal

**The assumption**: Checking signals and rebalancing every month captures the PE signal without excessive trading.

**Challenge**: Never tested against quarterly, bi-monthly, or threshold-based rebalancing. PE changes ~1-2 percentile per month. The difference between rebalancing monthly vs quarterly is at most a 4-6 percentile change in the signal — which translates to a 2-3% portfolio allocation shift. The reduction in turnover (from ~168% to ~56%) would more than offset any signal degradation.

**What a test would look like**: Compare monthly vs quarterly rebalancing on the same 3-fold walk-forward. The quarterly variant would have lower turnover, marginally less responsive allocations, and potentially similar or better Sharpe due to reduced transaction costs.

**Confidence**: Low-medium. The choice of monthly rebalancing was inherited from Architecture A without being questioned for Architecture B.

### Assumption 4: 511010 accurately tracks the CGB 5Y index

**The assumption**: The ETF price faithfully reflects the underlying bond index with minimal tracking error.

**Challenge**: Bond ETF tracking error can widen during periods of market stress or low liquidity. The 511010 ETF has ~3,200 rows of history (since 2013), suggesting adequate data, but the tracking error relative to the CSI ChinaBond 5Y Treasury Note Index has not been verified.

**Action item**: Compare 511010 daily returns against the index daily returns for the full history. If tracking error exceeds 0.5% annualized, model the difference as an additional cost.

### Assumption 5: AkShare PE data is accurate and column-stable

**The assumption**: `ak.stock_index_pe_lg(symbol="沪深300")` returns data where column 6 (0-indexed) is consistently the TTM PE value, and this will not change in future API versions.

**Challenge**: The column mapping is hardcoded (`df.iloc[:, 6]`). AkShare is a community-maintained library. API changes happen without notice. If the column structure changes, the model will silently read the wrong data and produce incorrect PE percentiles.

**Mitigation**: Add a validation check after fetching PE data: verify that PE values are in a plausible range (5-60 for CSI300 historically). If the fetched values fall outside this range, log a CRITICAL error and refuse to update the PE file.

---

## Round 4: System Risk

Risks that emerge from the interaction of components, not from any single component's failure.

### 4.1 Correlation explosion in crisis scenarios

In normal conditions, the three ETFs have low or negative correlation. In crisis conditions, correlations can converge to +1 or -1:

**Scenario A — Global liquidity crisis / RMB devaluation**:
- CSI300 ↓ (foreign capital outflow)
- CGB Bonds ↓ (capital flight from all RMB assets)
- Gold ↑ (global safe haven, but the model holds 0% if not trending)
- Result: equity AND bonds decline simultaneously, gold benefit missed because trend filter was off before the shock. Portfolio loss: -6% to -10%.

**Scenario B — China-specific credit event**:
- CSI300 ↓ (domestic risk-off)
- CGB Bonds ↑ (flight to sovereign safety)
- Gold → (unclear direction)
- Result: the bond allocation provides a partial hedge, offsetting approximately half the equity loss. Portfolio loss: -3% to -5%.

**Scenario C — Global risk-on with China underperformance**:
- CSI300 → (flat or slightly down)
- CGB Bonds → (flat, rates stable)
- Gold ↑ (global reflation trade)
- Result: gold benefit missed if trend filter is off. Opportunity cost: +5% to +10% vs a strategy with permanent gold allocation.

**The common thread**: The model's defensive posture relies on bond-equity negative correlation. When that correlation breaks (Scenario A), the model has no second layer of defense.

### 4.2 Data pipeline single point of failure

The `--fetch` pipeline has four sequential steps, each depending on the previous:

1. AkShare Sina API (ETF data) → 7 ETF parquet files
2. AkShare legulegu.com (PE data) → csi300_pe.parquet
3. AkShare macro APIs (PMI, CGB yield) → macro parquet files
4. reload_data() → in-memory T, pe_raw structures

If step 1 fails, all 7 ETFs retain their last successful fetch. The model can still produce signals. If step 2 fails, PE data is not updated. The model uses the previous successful PE data — which may be stale but is better than nothing. If step 3 fails, PMI/CGB data is not updated. Architecture B does not use this data, so the failure is invisible.

The most critical failure is step 2: if PE data is stale, the model's primary signal degrades. A 7-day-old PE value is acceptable (PE changes slowly). A 30-day-old PE value during a rapidly falling market would produce dangerously wrong signals (the model thinks PE is high when it has actually compressed significantly).

### 4.3 Execution timing gap

**PE signal**: Changes ~1-2 percentile per month. The 18-hour gap between signal computation (close of last trading day) and execution (open of next trading day) is negligible. The PE percentile on day T and day T+1 are nearly identical.

**Gold trend signal**: MA50 crossover can flip in a single day. If gold closes exactly at MA50 on the signal computation day, the trend signal is borderline. By the next morning, gold could have moved 1% in either direction, changing the signal. This is an edge case that occurs perhaps 2-3 times per year, but when it does, the model can produce a signal that is already stale by execution time.

**Mitigation**: Check the distance between gold price and MA50 on signal computation day. If the distance is <0.5% (i.e., gold is within 0.5% of its 50-day MA), flag the gold signal as "borderline" and consider delaying the gold portion of the rebalance by one day.

### 4.4 Model vs operator drift

The most common failure mode of systematic strategies in live trading: the operator overrides the model.

With Architecture B's current signal (14% equity, 0% gold, 86% bonds), the temptation to override is asymmetric: the model is saying "do almost nothing" (86% in bonds yielding ~2.5%). If the operator believes a policy catalyst is imminent, the override direction will always be "increase equity allocation" — never "decrease it further."

This creates a one-sided risk: the model protects against downside (by being conservative when PE is high), but the operator's overrides systematically add upside exposure without compensating downside protection. Over time, the live returns diverge from the backtest returns — and the divergence is always in the direction of higher risk and larger drawdowns.

**Mitigation**: Log every instance where the live allocation differs from the model signal. Review quarterly. If the cumulative impact of overrides is negative (as it almost always is in systematic strategies), institute a hard rule: no overrides without a written justification reviewed after 3 months.

---

## Round 5: Parameter Interaction Blind Spots

Interactions that the independent-parameter grid search did not explore.

### 5.1 eq_max × gold_max interaction

The grid search varies eq_max and gold_max independently. But their sum (eq_max + gold_max) is the maximum risk-on allocation. At eq_max=0.80 + gold_max=0.30, the model could allocate 110% to risk assets — but the allocation formula constrains eq_pct + gold_pct ≤ 1.0. The effective maximum is always ≤1.0, but the *interaction* between the two parameters determines which asset gets priority when the constraint binds.

At eq_max=0.80 + gold_max=0.30: when PE is cheap AND gold is trending, the model allocates eq_max - pp*(0.70) to equity and 0.30 to gold. At pp=0 (extremely cheap), equity gets 0.80 and gold gets 0.20 (not 0.30) because 0.80 + 0.30 > 1.0. The allocation formula silently caps the sum, but the cap behavior is not documented and not tested across the full parameter space.

### 5.2 eq_min hard floor behavior

eq_min=0.10 is a hard floor: at PE=100th percentile, equity is exactly 10%. At PE=101st percentile (conceptually — if the market exceeded its historical maximum PE), equity would still be 10%. There is no provision for "beyond historical extreme."

If Chinese equities reached a PE of 18 (historically high for the 2019-2024 period), the 5-year percentile would be close to 100. The model would allocate 10% equity. If PE then went to 20, the model would still allocate 10%. If PE went to 25 (a true bubble), still 10%.

The model has no mechanism for reducing equity below the floor in genuine bubble conditions. This is a deliberate design choice (the model always maintains some equity exposure), but the floor value (10%) was chosen as part of the grid search, not from a risk-management perspective. A crash from PE=25 to PE=15 with 10% equity allocation would cost approximately 4% of portfolio — manageable. But a crash from PE=25 to PE=8 (a 2008-style event) would cost approximately 6% of portfolio with 10% equity. The floor is not calibrated to worst-case drawdown.

### 5.3 gold_ma boundary sensitivity

At gold_ma=40, the trend filter triggers when gold is above its 40-day MA with positive slope. At gold_ma=60, it triggers at 60-day MA. In most market conditions, these produce similar signals — if gold is trending, it's above both MAs.

But there is a specific regime where they diverge: when gold has been in a mild uptrend for 40-50 days, then flattens. The 40-day MA will still have positive slope (the early part of the uptrend is still in the window), while the 60-day MA may have flattened (the early part has dropped out). This produces a 0% vs 30% gold allocation difference from a single parameter value.

The grid search did not specifically test for this boundary regime. The fact that all three folds converged to gold_ma=50 is reassuring but does not eliminate the sensitivity: at ma=50, the boundary between "trending" and "not trending" is a knife-edge that can be crossed by a single day's price movement.

**Mitigation**: Add a confirmation requirement: the gold trend signal must have been in its current state for at least 5 trading days before it triggers a rebalance. This prevents single-day whipsaws without changing the filter's overall sensitivity.

---

## Risk Register (Prioritized)

Ranked by probability × impact, with mitigation status.

| # | Risk | Probability | Impact | Mitigation | Status |
|---|------|-----------|--------|------------|--------|
| 1 | Bond duration loss from rate hike | Medium | High (4-5% portfolio loss at current allocation) | None. Model has no rate sensitivity overlay. | Not mitigated |
| 2 | PE data pipeline failure | Medium | High (wrong signal produced) | `--health` check catches staleness; column validation not implemented | Partially mitigated |
| 3 | Correlation breakdown (equity+bonds both decline) | Low | High (6-10% portfolio loss) | None. Model relies on bond-equity negative correlation. | Not mitigated |
| 4 | PMI stale data (286 days old) | High (certain) | None (PMI not currently used) | PMI filter already rejected. Data quality issue irrelevant to current model. | Accepted |
| 5 | Operator override of conservative signal | Medium | Medium (cumulative underperformance) | None. No override tracking mechanism. | Not mitigated |
| 6 | Gold signal stale by execution time (MA50 boundary) | Low | Low (<1% impact on isolated event) | Flag borderline signals; delay gold rebalance by 1 day when within 0.5% of MA | Not implemented |
| 7 | PE percentile discontinuity (index rebalancing) | Low | Medium (incorrect allocation for 1 month) | Manual check of PE around index rebalance dates | Not implemented |
| 8 | Value trap (prolonged cheap regime) | Low | Very High (strategy basis fails) | PE persistence check: if in bottom decile 6+ months with no recovery, halve equity | Not implemented |
| 9 | eq_max × gold_max interaction unmodeled | Low | Low (edge case, bounded by formula cap) | Documented. Formula silently caps at 1.0 which is correct behavior. | Accepted |
| 10 | 5-year PE window suboptimal | Unknown | Medium (strategy underperforms vs alternative window) | Never tested against 3/7/10-year alternatives | Not tested |

---

## Recommended Pre-Deployment Actions

1. **Check 511010 daily volume** against expected maximum bond position. Model slippage if >5% of daily volume.
2. **Add PE value validation** (range 5-60) to `fetch_pe_data()`. Reject and alert if values are outside plausible range.
3. **Add data freshness gate to `run_live()`**: refuse to produce signal if ETF data >2 days old or PE data >5 days old.
4. **Add gold signal boundary flag**: if gold close is within 0.5% of MA50, mark signal as borderline.
5. **Test quarterly rebalancing** vs monthly on the same 3-fold walk-forward (estimate: 3 minutes of runtime).
6. **Test PE windows 3, 4, 6, 7, 10 years** on Fold 3 screening (estimate: 5 minutes of runtime).
7. **Implement quarterly override review**: log every instance where live allocation differs from model signal.

---

*Risk engineering audit completed 2026-06-14. All findings are structural analyses, not bugs. The model is well-specified and its failure modes are predictable. The primary risk is not model error — it is the operator overriding a conservative signal during a market that the model correctly identifies as expensive.*

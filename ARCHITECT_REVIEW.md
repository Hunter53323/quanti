# Architectural Review: Quantitative Trading System (Greenfield)

## Verdict: ITERATE

The plan has a coherent shape and makes reasonable technology choices for the Chinese market, but contains critical architectural omissions that would surface as painful rewrites around phases 3-4. Specific issues to fix: missing state management architecture, no data validation pipeline, no order state machine, and a framework duality (Backtrader/vnpy) that needs resolution before a single line of code is written.

---

## Steelman Antithesis

**This plan will almost certainly lose money, not because the code will fail, but because the economics do not work at 100,000 RMB with a crowded, well-known strategy.**

Here is the strongest case against it:

**1. Capital is structurally insufficient for the primary strategy.** Convertible bond "dual-low" rotation typically requires holding 20-50 positions simultaneously to achieve meaningful diversification. Chinese convertible bonds trade at face values of 100 RMB with minimum lots of 10 (1,000 RMB minimum per position). In practice, bonds meeting dual-low criteria often trade at 105-130 RMB, meaning 1,050-1,300 RMB per lot. A bare-minimum portfolio of 20 positions consumes 21,000-26,000 RMB in position sizing alone, with no room for position sizing logic, no reserves for the ETF secondary strategy, and zero cash buffer. Add the ETF trend-following allocation and the system is either chronically under-diversified (concentration risk) or forced into positions too small to overcome transaction costs.

**2. The alpha has been arbitraged away.** The dual-low convertible bond strategy was popularized on Chinese retail forums (Jisilu, Xueqiu) around 2019-2021. By 2024-2026, it is one of the most widely replicated strategies among retail algo traders in China. When thousands of traders run the same screens on the same free data sources (Tushare/AkShare) and submit orders through the same broker APIs, any remaining edge is in execution quality and speed -- precisely the dimensions where a retail trader on MiniQMT has no advantage over institutional participants with colocated servers and direct exchange connectivity.

**3. Transaction costs at 100K scale are punishing.** A typical rotation strategy on convertible bonds might see 50-200% annual turnover. At 0.02%-0.05% commission per side (0.04%-0.10% round-trip), plus the 0.001% stamp duty on sell side (though convertible bonds are exempt from stamp duty, unlike stocks), plus slippage on illiquid bonds, the cost drag can exceed 2-4% annually. Against a 6-10% conservative return, costs alone consume 30-60% of alpha before considering the risk-free alternative (3-4% from money market funds or bonds with zero effort and zero drawdown risk).

**4. Convertible bond liquidity risk is acute and silent.** The dual-low screen inherently selects for bonds that are out of favor. Many such bonds have daily trading volumes below 1 million RMB. A 100K position in a bond with 500K daily volume means your own orders represent 20% of market activity. In a drawdown, you will not exit at your modeled price. Backtesting against daily close prices systematically overstates realizable returns because it assumes you can transact at the close -- which you cannot for illiquid names.

The steelman conclusion: **the expected risk-adjusted return, net of costs and execution slippage, is likely negative or indistinguishable from zero.** The 6-10% "conservative" estimate is not conservative; it is optimistic relative to the capital constraint and strategy crowding.

---

## Tradeoff Tensions

### Tension 1: Backtrader (research velocity) vs. vnpy (path to production)

This is the most consequential architectural decision the plan glosses over by listing both as "Backtrader / vnpy." They are fundamentally different systems with incompatible assumptions:

| Concern | Backtrader | vnpy |
|---------|-----------|------|
| Primary purpose | Research/backtesting | Live trading platform |
| Live execution | None (requires rewrite) | Native (connects to QMT, CTP, etc.) |
| Learning curve | Moderate | Steep (event-driven engine, complex object model) |
| Strategy portability | Locked into Backtrader API | Runs same code in backtest and live |
| Ecosystem | Python-only, broad community | Chinese-focused, tight broker integration |

**The tension:** If you prototype in Backtrader (faster research cycles), you will face a full strategy rewrite when moving to live trading in phases 4-5. If you commit to vnpy from day one, your research velocity will be slower because vnpy's backtesting is more constrained and its learning curve steeper. If you try both (Backtrader for research, vnpy for live), you now maintain two implementations of the same strategy logic, introducing divergence risk.

This is not a "pick whichever" decision -- it is the central architectural constraint and the plan treats it as an implementation detail.

### Tension 2: Strategy diversification vs. development complexity

Convertible bond dual-low and ETF trend following are poorly correlated strategies (good for portfolio construction, bad for development effort). They require:
- Different data sources (bond fundamentals + pricing vs. ETF price/volume)
- Different signal generation (cross-sectional ranking vs. time-series trend detection)
- Different execution models (limit orders on illiquid bonds vs. market/limit on liquid ETFs)
- Different risk models (credit risk + liquidity risk vs. trend reversal risk)

At 100K capital and with one developer, the overhead of maintaining two independent strategy stacks may exceed the diversification benefit. The development effort split means neither strategy receives full attention, increasing the probability of bugs in both.

---

## Architectural Gaps

The following critical components are missing entirely from the plan:

### 1. State Management and Persistence (Critical -- will block phase 3+)

No mention of where strategy state lives. What happens when:
- The process crashes mid-trading-day with open positions?
- The machine reboots for Windows Update during market hours?
- A partially filled order needs to be resumed after restart?

A production trading system needs a state machine with durable persistence. At minimum: position snapshots, order journal (submitted, acknowledged, partially filled, filled, cancelled, rejected), and strategy parameters. Without this, every restart is a manual reconciliation exercise against broker records.

### 2. Data Quality and Validation Pipeline

The plan mentions Tushare/AkShare as data sources but includes no validation architecture:
- How are missing data points detected and handled?
- How are corporate actions (bond redemptions, calls, conversions) reconciled against price data?
- How are data source discrepancies resolved (Tushare and AkShare can disagree on some fields)?
- What is the schema for normalized storage? (Raw ingestion, cleaned, and feature-engineered layers?)

Free data APIs have known issues: Tushare rate-limits aggressively on free tiers, AkShare scrapes publicly available data that can change format without notice. Without a validation layer, garbage data silently produces garbage signals.

### 3. Order State Machine and Position Management

"QMT/MiniQMT" is listed as the execution layer, but the plan says nothing about the order lifecycle:
- Order submission, acknowledgment, partial fill, full fill, rejection, cancellation states
- Timeout and retry logic (MiniQMT can silently drop order acknowledgments)
- Position reconciliation (what the strategy thinks it holds vs. what the broker says it holds)
- Pre-trade risk checks (do you have enough capital? is the position within limits? has the bond been called/redeemed?)

MiniQMT is a COM-based API that runs in-process. If your Python process crashes, you lose order state unless you explicitly persist it.

### 4. Error Recovery and Circuit Breakers

No failure modes are modeled:
- What triggers a strategy halt? (Consecutive losses? Execution failures? Data feed interruption?)
- How does the system degrade gracefully? (Stop new orders but manage existing positions? Liquidate everything?)
- What is the operator notification path? (WeChat? SMS? Email? And what if that channel also fails?)

### 5. Backtest-Reality Gap Management

The plan mentions "overfitting" as a risk but does not propose any architectural mitigations:
- Walk-forward analysis framework
- Out-of-sample period design
- Slippage and commission models calibrated to actual broker experience
- Survivorship bias handling (convertible bonds that were called, defaulted, or delisted)
- Regime detection to suppress trading in unfavorable conditions

### 6. Monitoring Architecture

"Logging + alerts" is too vague to be architectural. Missing:
- What metrics are collected? (P&L, positions, order latency, fill rate, slippage, drawdown, Sharpe, turnover)
- What is the alert escalation policy? (Warning vs. critical; who gets notified when?)
- Where do logs live? (Local files? Structured logging with rotation? Remote collection?)
- How do you audit a trading decision post-hoc? (Signal log, order log, fill log, all with timestamps for reconstruction)

### 7. Testing Strategy

No testing approach is described:
- Unit tests for signal calculation correctness
- Integration tests against historical data with known outcomes
- Paper trading as a test environment (how do you validate that paper fills match expected behavior?)
- Edge case tests: delisted bonds, called bonds, trading halts, price limits

### 8. Regulatory and Compliance

Chinese securities regulation regarding automated/programmatic trading is not addressed:
- Are there reporting requirements for algorithmic trading accounts?
- Does the broker (via QMT) require specific approvals or risk disclosures?
- Is the strategy classified as "high frequency" under CSRC guidelines (which imposes additional requirements)?

---

## Synthesis and Suggestions

### 1. Resolve the framework tension decisively

**Recommendation: Commit to vnpy from day one, accept the slower research velocity.** Rationale: The cost of rewriting a Backtrader strategy for live execution (and then debugging the behavioral differences between two implementations) exceeds the cost of learning vnpy's event-driven model upfront. The strategy code runs identically in backtest and live modes in vnpy's engine, eliminating the single largest source of production bugs (backtest-live divergence).

Trade-off acknowledged: Research iteration will be slower in weeks 1-6. Mitigate by building a lightweight Jupyter notebook harness around vnpy's data structures for rapid prototyping before committing strategies to the full framework.

### 2. Capital allocation architecture redesign

**Before writing any code, model the capital constraint explicitly:**

```
Total capital: 100,000 RMB
├── Primary strategy (CB dual-low): 70,000 RMB (70%)
│   └── Max 25 positions × avg 2,800 RMB = 70,000 RMB
├── Secondary strategy (ETF trend): 20,000 RMB (20%)
│   └── Max 3 ETFs × avg 6,700 RMB = 20,100 RMB
└── Cash buffer: 10,000 RMB (10%) -- never deployed
```

This constrains the architecture: the position sizer must enforce a hard maximum of 25 bond positions and must reject any position that would violate the allocation split. If 25 positions is insufficient for dual-low diversification, the primary strategy itself is infeasible at this capital level and should be replaced with a pure ETF rotation strategy using the full 90,000 RMB.

### 3. Mandatory architectural components before phase 3

Add the following to the phase plan before any backtesting begins (between phases 2 and 3):

- **Phase 2.5a: Data validation pipeline** (1 week)
  - Schema definition, null/missing detection, corporate action reconciliation, source comparison
  - Store raw ingested data separately from cleaned data

- **Phase 2.5b: State persistence module** (1 week)
  - SQLite-based position/order journal
  - Serialization format for strategy parameters
  - Crash-recovery procedure (replaying journal from last checkpoint)

- **Phase 2.5c: Order state machine** (concurrent with phase 3)
  - Finite state machine for order lifecycle
  - Position reconciliation against broker
  - Pre-trade risk checks (capital, position limits, instrument validity)

### 4. Monitoring specification

Define the monitoring architecture concretely:

- **Metrics collection**: In-process metrics registry (Prometheus model, but simpler -- a dict of gauges/counters pushed to structured logs)
- **Log format**: JSON-structured logs with fields: timestamp, level, module, event_type, order_id, symbol, quantity, price, reason
- **Alert channels**: WeChat Work webhook (primary, since this is China-focused) + local console (fallback)
- **Alert rules**: Circuit breaker on 3+ consecutive execution failures OR drawdown exceeding 5% in a single day

### 5. Consider a narrower initial scope

Given the capital and developer constraints, the architecture should support starting with a single strategy and adding the second later. Design the strategy interface to be polymorphic from the start. Start with ETF trend following only (simpler execution, higher liquidity, lower capital per position, easier to validate) for phases 3-5. Add convertible bonds only after the full pipeline (data, state, monitoring, execution) is proven with the simpler strategy.

---

## Windows-Specific Concerns

Since the environment is Windows 11 with PowerShell:

### 1. MiniQMT compatibility (positive)
MiniQMT's COM API is Windows-native, so this is actually a good fit. Full QMT relies on Windows GUI components. This is an advantage of the Windows environment, not a drawback.

### 2. vnpy installation on Windows
vnpy's Windows installation requires:
- Visual C++ Build Tools (for compiling Cython extensions)
- TA-Lib -- notoriously painful to compile from source on Windows; use the prebuilt wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/ or `pip install TA-Lib-cp3xx-cp3xx-win_amd64.whl`
- This should be documented as a setup prerequisite, not discovered during phase 1

### 3. Process management for long-running trading
On Linux, you would use systemd. On Windows, options are:
- **Windows Service** (via pywin32 or NSSM wrapper) -- reliable but harder to debug
- **Scheduled Task** (trigger on startup) -- simpler but harder to monitor
- **Terminal session** (just keep PowerShell open) -- easiest but fragile (session ends on logout)

Recommendation: Start with a terminal session during paper trading (phase 4), migrate to NSSM-wrapped Windows Service before live trading (phase 5). Document this migration path upfront.

### 4. Shell scripts and automation
The plan should specify `.ps1` scripts (PowerShell) rather than `.sh` scripts for process automation. Common pain points:
- `activate venv` in PowerShell is `.\venv\Scripts\Activate.ps1` (not `source venv/bin/activate`)
- Execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` must be run once
- `tail -f` equivalent is `Get-Content -Wait -Tail 30` (for log monitoring)

### 5. Encoding issues with Chinese market data
Chinese financial data often uses GBK/GB2312 encoding. Python 3 defaults to UTF-8, but data files from Chinese sources may not. Explicitly set `encoding='gbk'` or `encoding='gb2312'` when reading data files, and add encoding validation to the data pipeline. On Windows, `locale.getpreferredencoding()` may return `'cp936'` (the Windows GBK variant), which can mask encoding bugs that would surface on Linux CI.

### 6. File path handling
Python on Windows handles forward slashes in paths (`C:/study/AIWorkspace/quanti/data/`) correctly since Python 3.6+. Use `pathlib.Path` exclusively (never string concatenation) and this becomes a non-issue. But this convention must be enforced from the start to avoid mixed `\\` and `/` bugs.

---

## Summary of Required Changes Before Implementation

| Priority | Issue | Effort | Phase to add |
|----------|-------|--------|-------------|
| Critical | Resolve Backtrader vs. vnpy (choose vnpy) | Decision only | Before phase 1 |
| Critical | Model capital constraint explicitly | 2-4 hours | Before phase 1 |
| Critical | Add state persistence architecture | 1 week | Between phases 2 and 3 |
| Critical | Add data validation pipeline | 1 week | Between phases 2 and 3 |
| High | Add order state machine design | Concurrent with phase 3 | Phase 3 |
| High | Specify monitoring architecture | 2-3 days | Phase 4 |
| High | Start with single strategy (ETF only) | Decision only | Before phase 3 |
| Medium | Document Windows prerequisites | 1 day | Phase 1 |
| Medium | Define testing strategy | 2 days | Phase 2 |
| Low | Regulatory due diligence | 1 day | Before phase 5 |

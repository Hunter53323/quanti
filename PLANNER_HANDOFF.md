# Planner Handoff: Quantitative Trading System Architecture

## Instructions for the Planner

This document contains the mandatory inputs and acceptance criteria for the Planner phase. Every item below was identified as a gap in the original plan by the Architect review (ARCHITECT_REVIEW.md). The revised plan MUST address all of them explicitly.

---

## 1. Architectural Decisions to Codify (No Defaults Allowed)

The following decisions were left ambiguous in the original plan. The revised plan MUST state a clear choice for each, with a one-paragraph rationale.

| Decision | Options | Architect Recommendation |
|----------|---------|--------------------------|
| Backtesting/Live framework | Backtrader vs. vnpy vs. both | vnpy (single codebase for backtest and live) |
| Initial strategy scope | ETF-only vs. CB-only vs. both | ETF trend following only; add CB after pipeline validation |
| Process management (Windows) | Terminal session vs. Scheduled Task vs. NSSM service | Terminal session for paper trading (Phase 4), NSSM service for live (Phase 5) |

---

## 2. New Phases to Insert

The original plan had 6 phases. The revised plan MUST insert the following phases between Phase 2 (Data Pipeline) and Phase 3 (Strategy Backtesting). Renumber accordingly.

### Phase 2.5a: Data Validation Pipeline (1 week)

**Why this matters**: Tushare/AkShare are free APIs with known issues -- rate limiting, schema drift, missing data, encoding problems (GBK/GB2312 vs UTF-8). Without a validation layer, garbage data silently produces garbage signals.

**Required deliverables**:
- [ ] Normalized storage schema: raw ingestion layer, cleaned layer, feature-engineered layer (SQLite or Parquet)
- [ ] Null/missing data detection with configurable thresholds (reject if > X% missing)
- [ ] Corporate action reconciliation (bond redemptions, calls, conversions) against price data
- [ ] Data source cross-validation when both Tushare and AkShare provide the same field (flag discrepancies)
- [ ] Encoding validation: explicit charset handling for all ingestion sources (`gbk`, `gb2312`, `utf-8`)
- [ ] Rate-limit handling with exponential backoff for Tushare free tier
- [ ] Data freshness check: alert if expected daily data has not arrived by configurable cutoff time

### Phase 2.5b: State Persistence Module (1 week)

**Why this matters**: When the process crashes mid-trading-day (Windows Update, power loss, Python exception), you must know what you own and what orders are in flight. Without durable state, every restart is a manual reconciliation against broker records.

**Required deliverables**:
- [ ] SQLite-based position journal with schema: `timestamp, symbol, side, quantity, price, order_id, status`
- [ ] Order journal with full lifecycle: `submitted, acknowledged, partial_filled, filled, cancelled, rejected`
- [ ] Strategy parameter serialization (JSON) -- save all tunable parameters with each signal generation run
- [ ] Crash-recovery procedure: on startup, replay journal from last checkpoint, reconcile positions against broker
- [ ] Checkpoint mechanism: periodic snapshot of full portfolio state (position vector + cash + pending orders)
- [ ] Retention policy for journals (keep N days of tick-level detail, aggregate beyond that)

### Phase 2.5c: Order State Machine Design (runs concurrent with Phase 3)

**Why this matters**: MiniQMT is a COM-based in-process API. If your Python process crashes, you lose order state. An explicit state machine makes the order lifecycle auditable and recoverable.

**Required deliverables**:
- [ ] Finite state machine diagram for order lifecycle: `NEW -> SUBMITTED -> ACKNOWLEDGED -> PARTIAL_FILLED -> FILLED | REJECTED | CANCELLED`
- [ ] Timeout and retry logic: what happens when an order is submitted but never acknowledged?
- [ ] Position reconciliation: periodic comparison of local position book vs. broker position book
- [ ] Pre-trade risk checks: capital sufficiency, position limits, instrument validity (not called/redeemed/delisted)
- [ ] Idempotency: submitting the same order intent twice must not result in a double fill

---

## 3. Existing Phases to Harden

### Phase 1 (Environment Setup) -- add Windows-specific prerequisites

- [ ] Document Visual C++ Build Tools installation (required for vnpy Cython extensions)
- [ ] Document TA-Lib Windows installation (prebuilt wheel, not source compilation)
- [ ] Document PowerShell execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- [ ] Document virtual environment activation: `.\venv\Scripts\Activate.ps1` (not `source venv/bin/activate`)
- [ ] Set project convention: use `pathlib.Path` exclusively (never string concatenation for paths)

### Phase 2 (Data Pipeline) -- add testing and schema definition

- [ ] Define data schemas BEFORE writing ingestion code (not discovered during implementation)
- [ ] Add unit tests for data ingestion against known-correct sample data
- [ ] Add encoding edge case tests (GBK files, mixed-encoding files, BOM handling)

### Phase 3 (Strategy Backtesting) -- add backtest-reality gap mitigations

- [ ] Walk-forward analysis framework (not just simple train/test split)
- [ ] Out-of-sample period design with explicit date boundaries
- [ ] Slippage model calibrated to actual broker experience (not fixed 0.1% assumption)
- [ ] Commission model matching the actual broker rate schedule
- [ ] Survivorship bias handling: include delisted/called/redeemed bonds in historical universe
- [ ] Regime detection: suppress trading signals in detected unfavorable market regimes

### Phase 4 (Paper Trading) -- add monitoring specification

- [ ] Define metrics to collect: P&L, positions, order latency, fill rate, slippage, drawdown, Sharpe, turnover
- [ ] Define log format: JSON-structured with fields `timestamp, level, module, event_type, order_id, symbol, quantity, price, reason`
- [ ] Define alert channels: WeChat Work webhook (primary), local console (fallback)
- [ ] Define alert escalation: warning (non-critical anomaly) vs. critical (requires immediate attention)
- [ ] Define circuit breaker rules: halt on 3+ consecutive execution failures OR single-day drawdown > 5%
- [ ] Post-hoc audit trail: every trading decision must be reconstructable from logs alone

### Phase 5 (Small Capital Live) -- add regulatory compliance

- [ ] Verify broker QMT account has algorithmic trading approval
- [ ] Check CSRC reporting requirements for automated trading
- [ ] Confirm strategy is NOT classified as "high frequency" (which triggers additional obligations)
- [ ] Migrate from terminal session to NSSM-wrapped Windows Service

---

## 4. Capital Allocation Model (Mandatory)

The revised plan MUST include this explicit capital allocation as a system constraint:

```
Total capital: 100,000 RMB
├── Strategy allocation: 90,000 RMB (90%)
│   └── Phase 3-5: ETF trend only (full 90,000 RMB)
│   └── Phase 6+: Add CB dual-low when pipeline proven (70,000 CB + 20,000 ETF)
└── Cash buffer: 10,000 RMB (10%) -- never deployed, always in money market
```

The position sizer must enforce hard limits derived from this model.

---

## 5. Polymorphic Strategy Interface (Mandatory)

The architecture MUST define a strategy interface that allows adding new strategies without modifying the execution engine. Design this BEFORE Phase 3:

```
BaseStrategy (ABC)
  ├── generate_signals(market_data) -> list[Signal]
  ├── size_positions(signals, capital, portfolio) -> list[Order]
  └── risk_check(orders, portfolio) -> list[Order]

ETF_Trend_Strategy(BaseStrategy)    # Phase 3-5
CB_DualLow_Strategy(BaseStrategy)   # Phase 6+ (future)
```

---

## 6. Testing Strategy (Mandatory, added to Phase 2)

- [ ] Unit tests for signal calculation correctness (known input -> expected output)
- [ ] Integration tests: backtest on historical data with known benchmark outcomes
- [ ] Paper trading validation: compare paper fills against real market prints for the same orders
- [ ] Edge case tests: delisted instruments, called bonds, trading halts, price limits, dividend events
- [ ] State recovery tests: simulate crash at each order lifecycle state, verify correct recovery

---

## 7. Acceptance Criteria for the Planner Output

The revised plan is ACCEPTABLE when it:

1. Explicitly resolves all 3 architectural decisions in Section 1 with rationale
2. Contains phases 2.5a, 2.5b, 2.5c with the deliverables listed above
3. Hardens existing phases 1-5 with the checklist items above
4. Includes the capital allocation model as a numbered constraint
5. Defines the strategy interface before Phase 3 work begins
6. Includes testing strategy as a first-class concern in Phase 2
7. Adds 3-5 weeks to the original 12-16 week timeline (acknowledging the cost of doing it right)
8. Specifies *.ps1 scripts for all automation (PowerShell, not Bash)

---

## 8. Timeline Impact (Transparency)

The original plan was 12-16 weeks. Adding the gaps identified above adds 3-5 weeks:

- Phase 2.5a (data validation): +1 week
- Phase 2.5b (state persistence): +1 week
- Phase 2.5c (order state machine): concurrent with Phase 3, no net addition
- Hardened Phase 1 (Windows prerequisites): +0.5 week
- Hardened Phase 2 (testing + schemas): +0.5 week
- Hardened Phase 3 (backtest mitigations): +1 week
- Hardened Phase 4 (monitoring specification): +0.5 week
- Hardened Phase 5 (regulatory + process migration): +0.5 week

**Revised estimate: 15-21 weeks.**

---

Generated by Architect (ralplan consensus review)
Input for: Planner phase
Date: 2026-06-10

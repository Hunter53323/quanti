# Quantitative Trading System -- Revised Implementation Plan

**Capital**: 100,000 RMB | **Environment**: Windows 11, PowerShell 7+, Python 3.12+  
**Revised Timeline**: 15-21 weeks | **Status**: Draft for review  
**Generated**: 2026-06-10 | **Based on**: Architect Review (ARCHITECT_REVIEW.md) + Planner Handoff (PLANNER_HANDOFF.md)

---

## Architectural Decisions (Resolved)

### AD-1: Framework -- vnpy (not Backtrader)

**Choice**: vnpy.  
**Rationale**: The primary risk in DIY quant trading is the backtest-to-live gap. vnpy runs the identical strategy code in both modes via its event-driven engine, eliminating the need to rewrite strategies. Backtrader has no live execution path and would require a full re-implementation in phases 4-5. The cost of vnpy's steeper learning curve is paid once; the cost of dual implementations compounds with every strategy change forever.

**Mitigation for research velocity**: A lightweight Jupyter notebook harness wraps vnpy data structures during Phase 3 so signals can be prototyped interactively before committing strategy code to the full engine.

### AD-2: Initial strategy scope -- ETF trend following only

**Choice**: ETF trend following only for Phases 3-5. Convertible bond dual-low deferred to Phase 6+.  
**Rationale**: (a) ETF execution is simpler (liquid markets, no credit risk, reliable fills). (b) The full pipeline (data, state, monitoring, execution) must be proven with a low-risk strategy before adding complexity. (c) At 100K capital, the dual-strategy split would leave both strategies under-capitalized. ETF trend following uses the full 90,000 RMB allocation and avoids the concentration risk of under-diversified convertible bond portfolios.

### AD-3: Process management -- Terminal (Phase 4) then NSSM Service (Phase 5+)

**Choice**: Interactive terminal during paper trading; NSSM-wrapped Windows Service for live.  
**Rationale**: Paper trading requires active observation. Debugging a Windows Service obscures logs and makes interactive inspection harder. The migration from terminal to service is a deliberate milestone that validates operational readiness for unattended execution.

---

## Capital Allocation Model (Hard Constraint)

```
Total: 100,000 RMB
├── Trading capital: 90,000 RMB (90%)
│   └── Phase 3-5: ETF Trend Following (full 90K)
│   └── Phase 6+: Split 70K CB + 20K ETF when CB strategy added
└── Cash buffer: 10,000 RMB (10%) -- money market fund, never deployed
```

**Enforcement**: The position sizer rejects any order that would breach the 90K cap or the per-strategy sub-allocation. Buffer is held in a separate account or a designated money-market ETF (e.g., 511880).

---

## Strategy Interface (Polymorphic, Defined Before Phase 3)

All strategies implement this interface. The execution engine only depends on the abstract base class.

### Shared Types (`quanti/types.py`)

These types are shared across `data/`, `strategy/`, `state/`, and `execution/` modules. They are defined ONCE in `quanti/types.py` and imported everywhere else.

```python
# File: quanti/types.py
from dataclasses import dataclass, field
from datetime import datetime

@dataclass(frozen=True)
class Bar:
    """Single price bar for any instrument."""
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class MarketData:
    """Container passed to strategy.generate_signals()."""
    bars: dict[str, list[Bar]]               # symbol -> latest N bars
    index_bars: dict[str, list[Bar]]         # index data for regime detection
    timestamp: datetime

@dataclass
class Position:
    """Single position held in the portfolio."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float

@dataclass
class Portfolio:
    """Snapshot of current portfolio state."""
    positions: dict[str, Position]           # symbol -> position
    cash: float                               # available (unallocated) cash
    total_capital: float                      # cash + market value of all positions
    timestamp: datetime
```

### Strategy Base Class (`quanti/strategy/base.py`)

```python
# File: quanti/strategy/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from quanti.types import MarketData, Portfolio

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Signal:
    symbol: str
    side: OrderSide
    strength: float       # 0.0 to 1.0, for position sizing priority
    reason: str           # human-readable, logged for audit

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int         # in shares/lots
    price: float | None   # None = market order
    order_type: str       # "limit" | "market"
    signal_ref: str       # back-reference to signal for audit

class BaseStrategy(ABC):
    name: str

    @abstractmethod
    def generate_signals(self, market_data: "MarketData") -> list[Signal]:
        """Produce signals from market data. Pure computation, no side effects."""
        ...

    @abstractmethod
    def size_positions(
        self, signals: list[Signal], capital: float, portfolio: "Portfolio"
    ) -> list[Order]:
        """Convert signals to sized orders respecting capital constraints."""
        ...

    @abstractmethod
    def risk_check(self, orders: list[Order], portfolio: "Portfolio") -> list[Order]:
        """Filter or modify orders based on risk rules. May drop or reduce orders."""
        ...
```

Concrete implementations: `ETFTrendStrategy(BaseStrategy)` in Phase 3, `CBDualLowStrategy(BaseStrategy)` in Phase 6+.

---

## Phase Plan

### Phase 1: Environment Setup (Week 1-1.5)

**Goal**: Reproducible development environment, all dependencies install cleanly.

- [ ] **1.1** Install Python 3.12+ (windows installer, add to PATH)
- [ ] **1.2** Install Visual C++ Build Tools (required for vnpy Cython extensions)
  - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - Select "Desktop development with C++" workload
- [ ] **1.3** Install TA-Lib from prebuilt wheel (DO NOT compile from source)
  - Source: https://www.lfd.uci.edu/~gohlke/pythonlibs/
  - `pip install TA_Lib-0.4.28-cp312-cp312-win_amd64.whl`
- [ ] **1.4** Create virtual environment and install core packages
  - `python -m venv venv`
  - `.\venv\Scripts\Activate.ps1`
  - `pip install vnpy akshare tushare pandas numpy loguru pytest python-dotenv`
- [ ] **1.5** Set PowerShell execution policy
  - `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- [ ] **1.6** Create project structure (see below)
- [ ] **1.7** Create `activate.ps1` convenience script
- [ ] **1.8** Git init, `.gitignore` (venv/, data/raw/, *.pyc, .omc/, .env)
- [ ] **1.9** Create `.env.template` with Tushare token placeholder (never commit real token)
- [ ] **1.10** Smoke test: `python -c "import vnpy; print('OK')"`

**Project structure after Phase 1**:
```
quanti/
├── activate.ps1
├── .env.template
├── .gitignore
├── quanti/
│   ├── __init__.py
│   ├── main_live.py              # Entry point for live trading (Phase 5+)
│   ├── main_paper.py             # Entry point for paper trading (Phase 4)
│   ├── types.py                  # Shared types: MarketData, Portfolio
│   ├── config/
│   │   └── settings.py           # All tunable parameters, loaded from .env
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py             # Data schemas (dataclasses)
│   │   ├── ingestion/            # Per-source fetchers (tushare_, akshare_)
│   │   ├── validation.py         # Phase 2.5a
│   │   └── storage.py            # SQLite/Parquet read/write
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseStrategy ABC (defined below)
│   │   └── etf_trend.py          # Phase 3
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── broker.py             # QMT/MiniQMT adapter
│   │   ├── order_manager.py      # Order state machine (Phase 2.5c)
│   │   └── risk.py               # Pre-trade risk checks
│   ├── state/
│   │   ├── __init__.py
│   │   ├── journal.py            # Position/order journal (Phase 2.5b)
│   │   └── recovery.py           # Crash recovery logic (Phase 2.5b)
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── engine.py             # vnpy backtest wrapper
│   └── monitor/
│       ├── __init__.py
│       ├── metrics.py            # Metrics registry
│       ├── logger.py             # JSON structured logging
│       └── alerts.py             # WeChat Work webhook + console
├── scripts/                      # PowerShell automation (NOT inside Python package)
│   ├── ingest_daily.ps1
│   ├── run_backtest.ps1
│   ├── run_paper.ps1
│   ├── run_live.ps1
│   ├── reconcile.ps1
│   └── healthcheck.ps1
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py
│   ├── test_signals.py
│   ├── test_order_state.py
│   ├── test_state_recovery.py
│   └── fixtures/                # Known-correct sample data
├── notebooks/                   # Jupyter research harness
│   └── signal_prototyping.ipynb
└── data/
    ├── raw/                     # Untouched ingestion output
    ├── clean/                   # Validated, normalized
    └── features/                # Strategy-ready
```

---

### Phase 2: Data Pipeline (Week 2-4, expanded from original 2 weeks)

**Goal**: Reliable daily ingestion of ETF price/volume data, validated and stored.

- [ ] **2.1** Implement `quanti/data/schema.py` -- define `ETFDailyBar`, `ETFMinuteBar`, `IndexDailyBar` as frozen dataclasses
- [ ] **2.2** Implement Tushare ingestion for ETF daily bars (primary source)
- [ ] **2.3** Implement AkShare ingestion for ETF daily bars (fallback source)
- [ ] **2.4** Add rate-limit handling: exponential backoff, configurable retry count
- [ ] **2.5** Implement `quanti/data/storage.py` -- SQLite for metadata, Parquet for time-series
- [ ] **2.6** Write daily ingestion script: `.\scripts\ingest_daily.ps1`
- [ ] **2.7** **Testing**: Unit test ingestion against known fixtures, edge case tests for missing data, encoding smoke test (GBK file)
- [ ] **2.8** Schedule daily ingestion via Windows Task Scheduler (run at 16:30, after market close data availability)

---

### Phase 2.5a: Data Validation Pipeline (Week 5)

**Goal**: Garbage data never reaches the strategy layer.

- [ ] **2.5a.1** Implement null/missing detection in `quanti/data/validation.py`
  - Configurable threshold: reject instrument if > X% of fields missing (default 5%)
  - Interpolation for single-day gaps vs. rejection for multi-day gaps
- [ ] **2.5a.2** Implement encoding validation: detect charset on read, normalize to UTF-8
  - Handle GBK, GB2312, UTF-8, UTF-8-BOM
- [ ] **2.5a.3** Implement cross-source comparison (Tushare vs. AkShare)
  - Flag discrepancies > 1% in OHLCV fields; log warning
  - Default to Tushare when both available, AkShare as fallback
- [ ] **2.5a.4** Implement data freshness check
  - Alert if expected daily data not received by 17:00 on trading days
  - Track last ingested date per instrument
- [ ] **2.5a.5** Implement corporate action reconciliation (ETF dividends, splits)
  - Fetch dividend/split calendar from AkShare
  - Validate that adjusted prices account for known events
- [ ] **2.5a.6** Three-layer storage: raw (immutable, as-received) -> clean (validated, normalized) -> features (strategy-ready derived columns)
- [ ] **2.5a.7** **Testing**: Feed deliberately corrupted data through pipeline, verify rejection/marking

---

### Phase 2.5b: State Persistence Module (Week 6)

**Goal**: Survive a crash at any point and know exactly what you own.

- [ ] **2.5b.1** Implement `quanti/state/journal.py` -- SQLite-based dual journal

  **Position journal schema**:
  ```sql
  CREATE TABLE position_journal (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,        -- ISO 8601
      symbol TEXT NOT NULL,
      side TEXT NOT NULL,             -- 'buy' | 'sell'
      quantity INTEGER NOT NULL,
      price REAL NOT NULL,
      order_id TEXT,
      status TEXT NOT NULL,           -- 'pending' | 'confirmed'
      reason TEXT                     -- signal or manual override
  );
  ```

  **Order journal schema**:
  ```sql
  CREATE TABLE order_journal (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      order_id TEXT NOT NULL UNIQUE,
      symbol TEXT NOT NULL,
      side TEXT NOT NULL,
      quantity INTEGER NOT NULL,
      price REAL,
      order_type TEXT NOT NULL,       -- 'limit' | 'market'
      status TEXT NOT NULL,           -- state machine status
      filled_qty INTEGER DEFAULT 0,
      avg_fill_price REAL,
      retry_count INTEGER DEFAULT 0,
      last_error TEXT
  );
  ```

- [ ] **2.5b.2** Implement checkpoint mechanism: periodic snapshot (every N minutes, configurable) of full portfolio state serialized as JSON alongside journal
- [ ] **2.5b.3** Implement `quanti/state/recovery.py`
  - On startup: read last checkpoint, replay journal entries since checkpoint timestamp
  - Reconcile computed positions against broker (call QMT position query)
  - Flag discrepancies for operator review
  - Restore pending orders to order manager
- [ ] **2.5b.4** Implement retention policy: 90 days tick-level journal, aggregate beyond that
- [ ] **2.5b.5** **Testing**: Simulate crash at each order lifecycle state, verify correct recovery
  - Crash before submit: no position, no order
  - Crash after submit, before ack: reconcile against broker, exactly one order
  - Crash after partial fill: recover fill quantity, resubmit remainder
  - Crash after full fill: recover, no duplicate order

---

### Phase 3: Strategy Backtesting (Week 7-10, expanded from original 2-3 weeks)

**Goal**: ETF trend-following strategy with rigorous validation and walk-forward analysis.

- [ ] **3.1** Implement `ETFTrendStrategy` in `quanti/strategy/etf_trend.py`
  - Signal logic: Moving average crossover + ADX trend filter on major ETFs (CSI 300, CSI 500, ChiNext 50)
  - Configurable parameters loaded from `quanti/config/settings.py`
- [ ] **3.2** Implement `quanti/backtest/engine.py` -- vnpy backtest wrapper
  - Feed: clean data from Phase 2.5a (not raw)
  - Slippage model: configurable basis points, NOT hardcoded
  - Commission model: match broker rate schedule (typically 0.025% per side for ETFs)
- [ ] **3.3** Walk-forward analysis framework
  - In-sample period: 2019-2023 (4 years)
  - Walk-forward windows: 1-year train, 3-month test, roll forward
  - Record parameter stability across windows (detect overfitting: parameters that oscillate wildly)
- [ ] **3.4** Out-of-sample period: 2024-2025 (2 years, never looked at during development)
- [ ] **3.5** Survivorship bias handling: use ETF universe from each historical point-in-time (not current constituent list)
- [ ] **3.6** Regime detection: suppress signals when VIX/VXFXI > 90th percentile or ADX < 20 (no trend)
- [ ] **3.7** Metrics output: CAGR, Sharpe, max drawdown, Calmar, win rate, profit factor, annual turnover
- [ ] **3.8** Save all backtest parameters + results as JSON for audit trail
- [ ] **3.9** **Testing**: Unit test signal calculation against known input/output pairs
- [ ] **3.10** **Testing**: Integration test -- backtest against benchmark buy-and-hold CSI 300 ETF

**Acceptance gate (go/no-go for Phase 4)**:
- Walk-forward Sharpe > 0.5 AND out-of-sample Sharpe > 0.3
- Max drawdown < 25% in any single walk-forward window
- Annual turnover < 400% (cost constraint)
- IF results fail these thresholds: iterate on parameters OR abandon ETF trend, try alternative strategy

---

### Phase 2.5c: Order State Machine (Week 7-8, concurrent with Phase 3)

**Goal**: Every order's lifecycle is defined, persisted, and recoverable.

- [ ] **2.5c.1** Implement `quanti/execution/order_manager.py` with explicit state machine

  **States**: `NEW -> SUBMITTED -> ACKNOWLEDGED -> PARTIAL_FILLED -> FILLED | REJECTED | CANCELLED`

  **State transitions**:
  ```
  NEW          --(submit)--> SUBMITTED
  SUBMITTED    --(ack)-----> ACKNOWLEDGED
  SUBMITTED    --(reject)--> REJECTED
  SUBMITTED    --(timeout)-  -> REJECTED (with retry, max 3 attempts)
  ACKNOWLEDGED --(fill)----> PARTIAL_FILLED
  PARTIAL_FILLED --(fill)--> PARTIAL_FILLED (incremental)
  PARTIAL_FILLED --(fill)--> FILLED
  ACKNOWLEDGED --(cancel)--> CANCELLED
  PARTIAL_FILLED --(cancel)--> CANCELLED (partial fill recorded)
  ```
  Every transition writes to `order_journal` table.

- [ ] **2.5c.2** Implement MiniQMT adapter in `quanti/execution/broker.py`
  - Connect/login, subscribe to order/trade callbacks, disconnect/cleanup
- [ ] **2.5c.3** Implement pre-trade risk checks in `quanti/execution/risk.py`
  - Capital sufficiency: order cost <= available cash
  - Position limit: post-order position <= strategy allocation cap
  - Instrument validity: instrument is actively trading, not suspended
  - Duplicate check: same symbol + same side + within 60 seconds = potential duplicate, flag
  - **Checks run BEFORE order submission, every time**
- [ ] **2.5c.4** Implement position reconciliation (runs every 5 minutes during live)
  - Query local position book vs. broker position book
  - Differences > 1 share/lot: log warning, use broker as source of truth
- [ ] **2.5c.5** **Testing**: State machine unit tests for every transition
- [ ] **2.5c.6** **Testing**: Integration test with simulated QMT callbacks

---

### Phase 4: Paper Trading (Week 11-14, expanded from original 1-2 weeks)

**Goal**: Live market data, real-time signals, simulated fills. Prove the pipeline end-to-end without risking capital.

- [ ] **4.1** Deploy paper trading environment: vnpy paper trading module with live data feed
- [ ] **4.2** Implement `quanti/monitor/metrics.py` -- metrics registry
  - Tracked: daily/MTD/YTD P&L, position count, order latency (submit-to-ack, ack-to-fill), fill rate, slippage vs. signal price, drawdown from peak, rolling Sharpe (20-day), daily turnover
- [ ] **4.3** Implement `quanti/monitor/logger.py` -- JSON-structured logging
  - Format: `{"ts": "ISO8601", "level": "INFO|WARN|ERROR|CRITICAL", "module": "...", "event": "...", "order_id": "...", "symbol": "...", "qty": N, "price": N.N, "reason": "..."}`
  - Output: rotating file (10MB x 5 files) + stdout
- [ ] **4.4** Implement `quanti/monitor/alerts.py`
  - WeChat Work webhook for critical alerts (drawdown, execution failure, data gap)
  - Console output for warnings
  - Escalation: warning (non-critical, logged) vs. critical (immediate WeChat notification)
- [ ] **4.5** Implement circuit breakers
  - **Rule 1**: 3+ consecutive execution failures -> halt new orders, manage existing positions, alert CRITICAL
  - **Rule 2**: Single-day realized loss > 2% of capital -> halt new orders, alert CRITICAL
  - **Rule 3**: Data feed gap > 5 minutes during market hours -> halt new orders, alert WARNING
  - Operator must manually re-enable after circuit breaker trip
- [ ] **4.6** Run paper trading for minimum 4 weeks (not 1-2)
- [ ] **4.7** **Validation**: Compare paper fill prices against actual traded prices from the same day (assess fill realism)
- [ ] **4.8** Post-hoc audit: verify that every trading decision can be reconstructed from logs
- [ ] **4.9** Create `.\scripts\run_paper.ps1`

**Acceptance gate (go/no-go for Phase 5)**:
- System ran 20 consecutive trading days without crash
- All circuit breaker rules tested (simulated), all triggered correctly
- Post-hoc audit of 5 random trading decisions: all fully reconstructable from logs
- Paper P&L within +/- 20% of backtest expectation for same period

---

### Phase 5: Small Capital Live Trading (Week 15-18, expanded from original 2-3 weeks)

**Goal**: Real money, minimum position sizes, prove operational reliability.

- [ ] **5.1** Open broker account with QMT support, confirm algorithmic trading approval
- [ ] **5.2** Verify: CSRC reporting requirements checked, account not classified as "high frequency"
- [ ] **5.3** Migrate from terminal session to NSSM-wrapped Windows Service
  - Install NSSM (https://nssm.cc/)
  - `nssm install QuantiTrading C:\study\AIWorkspace\quanti\venv\Scripts\python.exe C:\study\AIWorkspace\quanti\quanti\main_live.py`
  - Configure: auto-restart on crash, redirect stdout/stderr to log files
  - Test: stop service, verify state recovery on restart
- [ ] **5.4** Deploy with minimum viable position sizes (100 shares of cheapest qualifying ETF)
- [ ] **5.5** Run for minimum 3 weeks, progressively increase position sizes as confidence grows
- [ ] **5.6** Daily review: check P&L vs. expected, verify no unexpected orders, reconcile positions
- [ ] **5.7** Create `.\scripts\run_live.ps1`

**Acceptance gate (go/no-go for Phase 6)**:
- 15 consecutive trading days with real money, zero unhandled errors
- Actual slippage <= 2x the modeled slippage (if worse, recalibrate model)
- Actual fill rate > 95%
- No circuit breaker trips caused by system bugs (only market conditions)
- P&L within +/- 30% of paper trading for same period

---

### Phase 6: Gradual Scaling + Second Strategy (Week 19+)

**Goal**: Compound capital, add convertible bond strategy when pipeline is proven.

- [ ] **6.1** Increase position sizes proportional to capital growth
- [ ] **6.2** Implement `CBDualLowStrategy` in `quanti/strategy/cb_dual_low.py`
  - Reuse existing pipeline: data validation, state persistence, order manager, monitoring
- [ ] **6.3** Implement convertible bond data ingestion (fundamentals + pricing via Tushare/AkShare)
- [ ] **6.4** Re-validate full pipeline with CB data (corporate actions: redemptions, calls, conversions)
- [ ] **6.5** Paper trade CB strategy separately for 4 weeks before combining
- [ ] **6.6** Split allocation: 70K CB + 20K ETF (maintaining 10K buffer)

---

## Testing Strategy (Cross-Cutting)

| Test Category | Phase | What | Tool |
|---------------|-------|------|------|
| Unit: data validation | 2.5a | Corrupt/missing data -> correct rejection | pytest |
| Unit: signal calculation | 3 | Known input -> expected output | pytest |
| Unit: order state machine | 2.5c | Every state transition | pytest |
| Integration: backtest | 3 | Backtest result vs. known benchmark | pytest |
| Integration: state recovery | 2.5b | Crash simulation at each state | pytest |
| Paper validation | 4 | Paper fill vs. market print | manual + script |
| Edge case: delisted/suspended | 3 | Historical instruments that delisted | pytest |
| Edge case: dividends/splits | 2.5a | Forward-adjusted price correctness | pytest |
| Edge case: encoding | 2.5a | GBK/GB2312/UTF-8/BOM files | pytest |
| Edge case: trading halt | 4 | Simulated halt during position holding | paper trading |
| Operational: circuit breaker | 4 | Trigger each rule, verify halt | paper trading |
| Operational: crash recovery | 5 | Kill process during trading, recover | live (off-hours) |

---

## Monitoring Architecture

**Metrics (collected every signal cycle)**:
- P&L: realized, unrealized, daily, MTD, YTD
- Positions: count, exposure %, per-symbol weight
- Orders: submit-to-ack latency, ack-to-fill latency, fill rate %, reject rate %
- Risk: current drawdown %, rolling 20-day Sharpe, daily turnover %
- Data: last ingest timestamp, data freshness (minutes since last bar)

**Logging (JSON-structured, every event)**:
```
{"ts":"2026-08-15T09:31:05+08:00","level":"INFO","module":"order_manager","event":"order_ack","order_id":"20260815-0003","symbol":"510300","qty":1000,"price":3.850}
{"ts":"2026-08-15T09:31:12+08:00","level":"WARN","module":"data.validation","event":"missing_data","symbol":"159915","field":"volume","date":"2026-08-14","action":"interpolated"}
{"ts":"2026-08-15T09:45:00+08:00","level":"CRITICAL","module":"monitor.alerts","event":"circuit_breaker","rule":"consecutive_failures","count":3,"action":"halt_new_orders"}
```

**Alert rules**:

| Condition | Level | Channel |
|-----------|-------|---------|
| Execution failure (single) | WARNING | Console |
| 3+ consecutive execution failures | CRITICAL | WeChat + Console |
| Single-day drawdown > 2% | CRITICAL | WeChat + Console |
| Data feed gap > 5 min in market hours | WARNING | Console |
| Data feed gap > 15 min in market hours | CRITICAL | WeChat + Console |
| Position mismatch (local vs. broker) | WARNING | Console |
| Fill price slippage > 0.5% | WARNING | Console |

---

## Scripts (PowerShell, all `.ps1`)

| Script | Purpose | Schedule |
|--------|---------|----------|
| `.\scripts\ingest_daily.ps1` | Fetch and validate daily data | 16:30 trading days (Task Scheduler) |
| `.\scripts\run_backtest.ps1` | Run walk-forward backtest | On demand |
| `.\scripts\run_paper.ps1` | Start paper trading session | Manual (Phase 4) |
| `.\scripts\run_live.ps1` | Start live trading (via NSSM) | Auto-start with Windows (Phase 5+) |
| `.\scripts\reconcile.ps1` | Manual position reconciliation | On demand |
| `.\scripts\healthcheck.ps1` | System health: data freshness, process alive, P&L | 09:00 daily |

---

## Timeline Summary

| Phase | Name | Duration | Cumulative |
|-------|------|----------|------------|
| 1 | Environment Setup | 1.5 weeks | Week 1.5 |
| 2 | Data Pipeline | 3 weeks | Week 4.5 |
| 2.5a | Data Validation | 1 week | Week 5.5 |
| 2.5b | State Persistence | 1 week | Week 6.5 |
| 3 | Strategy Backtesting | 3 weeks | Week 9.5 |
| 2.5c | Order State Machine | concurrent with 3 | Week 9.5 |
| 4 | Paper Trading | 4 weeks | Week 13.5 |
| 5 | Small Capital Live | 3 weeks | Week 16.5 |
| 6 | Scaling + CB Strategy | ongoing | Week 17+ |

**Total to live trading with real money**: ~16.5 weeks (within the 15-21 week estimate).

---

## Acceptance Criteria Checklist (from PLANNER_HANDOFF.md Section 7)

- [x] 1. All 3 architectural decisions resolved with rationale (AD-1, AD-2, AD-3)
- [x] 2. Phases 2.5a, 2.5b, 2.5c included with deliverables
- [x] 3. Existing phases 1-5 hardened with checklist items
- [x] 4. Capital allocation model included as hard constraint
- [x] 5. Strategy interface defined before Phase 3
- [x] 6. Testing strategy as first-class concern
- [x] 7. Timeline revised to 15-21 weeks (16.5 weeks estimated)
- [x] 8. All scripts specified as .ps1 (PowerShell)


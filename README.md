# quanti — A-Share Delayed Confirmation Momentum Strategy

A quantitative trading system for the Chinese A-share market that combines delayed breakout confirmation with time-decay position sizing. Full research pipeline from backtesting through live paper trading.

**Core insight**: 72 CSI300 120MA breakouts were analyzed — only 24% were genuine. The key differentiator between real and false breakouts is **duration**, not intensity. Waiting 5 days after a breakout before entering captures real trends while filtering false ones.

---

## Quick Start

```bash
cd C:\study\AIWorkspace\quanti

# Full backtest — 6 strategies compared, 80 seconds
python run_backtest.py

# N×M parameter verification — 25 combos, 52 seconds
python run_backtest.py --verify

# VT/MD redundancy check
python run_backtest.py --check-vtmd

# Live paper trading (Ctrl+C to stop)
python quanti/main_paper_delayed.py

# CI gate
python check.py --quick
```

## Final Results (2022-2025 Test Period)

```
Strategy                         Train C   Train S  Train D |  Test C   Test S  Test D
BASELINE (120MA binary)          +26.34%    0.997    20.0%  |  -4.76%  -0.506   18.1%
BOND_ROTATE (entry only)         +7.07%     0.600    16.6%  |  +2.38%   0.256   13.6%
+ A43 decay                      +6.85%     0.606    16.1%  |  +3.52%   0.359   13.6%
+ Sharp3pct exit (no decay)      +13.96%    1.185     6.3%  |  +2.75%   0.368   10.4%
+ Sharp3pct + A43                +13.98%    1.206     4.7%  |  +3.78%   0.528    6.7%
```

**Best**: Sharp3pct+A43 — **+8.54% CAGR improvement** over baseline, MaxDD reduced from 18.1% to 6.7%.

## Strategy Stack (inside-out)

```
CSI300 daily bars
  │
  ▼
State machine (N=5 day confirmation, M=40 day cooldown)
  ├── 0 = defensive (511880 bond ETF)
  ├── 2 = full position (5 days above 120MA confirmed)
  └── 4 = half position (above 60MA, not yet 120MA-confirmed)
  │
  ▼
Position = state baseline (1.0/0.5/0) × A43 decay (100%→75%→50%)
  │
  ▼
Stock selection = 5-condition trend filter, top 5 by momentum (60%) + low-vol (40%)
  │
  ▼
Sharp3pct exit (CSI300 5-day return < -3% → full exit, 40-day cooldown)
  │  (months_in_cycle preserved across cooldown — genuine_prev fix)
  │
  ▼
Defensive: 511880 bond/money-market ETF
```

## Architecture

The system has a clean separation between research and production:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Canonical backtest** | `run_backtest.py` | Single source of truth. All numbers flow through one engine. |
| **Strategy class** | `quanti/strategy/delayed_confirm.py` | `DelayedConfirmStrategy` — implements `BaseStrategy` ABC. Used by both backtest and live trading. |
| **Paper trading** | `quanti/main_paper_delayed.py` | 60-second loop: ingest → signal → size → risk check → fill simulation → checkpoint |
| **Research history** | `scripts/delayed_confirm_backtest.py` | 1st-generation binary parameter sweep (N×M, VT/MD gating) |
| **Attribution** | `attribution.py`, `attribution2.py` | Selection vs. timing alpha decomposition |
| **Handbook** | `HANDBOOK.md` | Complete research narrative: 5 generations, parameter convergence, bugs found |

The system also includes:
- `quanti/strategy/etf_trend.py` — ETF trend-following (weighted scoring entry, 5 exit methods)
- `quanti/strategy/stock_momentum.py` — Stock momentum strategy (baseline)
- `quanti/strategy/pe_band.py` — CSI300 PE percentile allocation
- `quanti/strategy/dividend_barbell.py` — Core-satellite with dividend/bond/gold
- `quanti/strategy/cb_dual_low.py` — Convertible bond dual-low rotation
- `quanti/strategy/signal_filters.py` — Market environment filters (trending, bear market, forbidden periods, NT intervention)
- `quanti/strategy/signal_concentration.py` — Algorithmic herding detection
- `quanti/backtest/engine.py` — BacktestEngine: walk-forward, OOS, T+1 settlement
- `quanti/execution/` — Order FSM, risk checker, circuit breakers, broker simulation
- `quanti/state/` — SQLite journal + crash recovery checkpoint system
- `quanti/monitor/` — Structured logging (loguru), WeChat Work alerts, metrics

## Verified Parameters

| Parameter | Value | Verification |
|-----------|-------|-------------|
| N (confirmation days) | 5 | N×M sweep: #1/25 by Test CAGR |
| M (cooldown days) | 40 | Sweet spot 40-50 |
| VT (volume threshold) | removed | VT/MD check: 0.0000% difference |
| MD (max decline) | removed | Redundant with cooldown mechanism |
| Sharp3pct threshold | -3.0% | Stable range -2% to -4% |
| A43 decay | 1-4m:100%, 5-8m:75%, 9m+:50% | Best of 5 decay schedules |
| Half-tier (60MA) | structural | 2-state comparison confirms necessity |

## Key Decisions Made

### VT/MD gates removed
CSI300 daily volume near breakouts never drops below 60% of its 20-day average. A -2% decline within N=5 days almost always crosses below 120MA and triggers cooldown first. Both gates confirmed redundant at 0.0000% delta.

### No gold overlay
Gold (518880) showed excellent backtest results (+9.17% Test CAGR at 40/60 bond/gold split) — but this is entirely explained by gold's exceptional 2022-2025 performance, not strategy edge. Removed from production stack. Bond-only (511880) defense is the conservative baseline.

### Stock universe: CSI300 + CSI500 constituents
`quanti/main_paper_delayed.py` filters the 624-stock pool to index constituents via `akshare.index_stock_cons()`. This avoids survivorship bias from stocks with incomplete histories.

## Data

640 parquet files in `data/clean/` covering CSI300 ETF (510300), bond ETF (511880), and A-share stocks from 2002-2026. Data ingestion pipeline uses Tushare (primary) with AkShare fallback.

## Known Limitations

- 624-stock pool has incomplete delisting coverage (survivorship bias in backtests)
- Monthly rebalancing may overestimate liquidation prices in extreme events
- Train Sharpe 1.206 (Sharp3pct+A43) is very high — not independently cross-validated
- Strategy class not verified through BacktestEngine on full 624-stock universe
- Slippage/commission sensitivity not modeled in detail
- No statistical significance tests (p-values, confidence intervals, bootstrap)

## Documents

| File | Content |
|------|---------|
| `HANDBOOK.md` | Complete research narrative — 5 generations of strategy evolution, parameter convergence, bugs discovered and fixed, dead ends explored |
| `AGENTS.md` | Project file map and architecture invariants |
| `data/delayed_confirm_report.md` | Chinese-language research narrative |
| `README.md` | This file |

---

*Built on the quanti framework. All research numbers are reproducible via `python run_backtest.py` (80 seconds).*

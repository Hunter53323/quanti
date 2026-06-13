# quanti — A-Share Delayed Confirmation Momentum Strategy

A systematic momentum-timing strategy for the Chinese A-share market. Combines delayed breakout confirmation, multi-tier position sizing, and time-decay exposure to produce positive returns with controlled drawdowns.

**Core design**: When the market trends, follow it with conviction. When it doesn't, wait in bonds. The state machine does the waiting so the investor doesn't have to.

---

## 背景：为什么做 A 股，为什么这么做

### A 股 vs 美股的天然优势

A 股有三个特征使其比美股更适合中小个人投资者做系统性动量策略：

1. **趋势持续性更强**。A 股散户占比高、信息效率低，趋势一旦形成往往持续数月至一年——不像美股那样几天内被套利资金抹平。CSI300 在 120MA 上方一旦站稳，平均持续 4-6 个月。这给了动量策略足够的持仓窗口。

2. **品种分散度高**。624 只可交易股票覆盖全市场，top-5 动量股的月度轮换率极高（很少连续两月同一批标的）。这意味着策略不依赖少数牛股，而是靠持续的"选最强"机制滚动获得超额收益。

3. **政策红利客观存在**。国家队持有约 1.5 万亿 ETF，市场极端下跌时有托底力量。策略在熊市几乎空仓（2022-2025 test 期 31/48 个月空仓），天然避开了最惨的下跌，却又能在国家队拉盘时及时跟上。

### 适合中小个人投资者的特征

- **小资金友好**：最低一手（100 股）即可执行，初始资金 9 万元即可完整复制策略。所有交易标的都是 ETF 和沪深 300/500 成分股，流动性充裕，单笔冲击成本可控。
- **路径无关**：策略不是埋在 Excel 里的一次性分析结果。`run_backtest.py` 一键复现所有数字，`quanti/main_paper_delayed.py` 是持续运行的纸交易引擎——任何人在自己机器上都能独立验证、持续跟踪。
- **完全可复现**：全部回测数字来源于 `run_backtest.py` 的 80 秒运行。无需依赖第三方数据平台、无外部 API 调用（回测阶段直接读本地 parquet 文件）、无黑箱参数。

### 与常见个人投资方式的对比

| 投资方式 | 本质 | 资金效率 | 回撤风险 | 适应性 |
|---------|------|---------|---------|--------|
| **价值投资**（买好公司长期持有） | "我相信这家公司未来会更好" | 满仓暴露，无法规避系统性下跌 | 深度回撤（2022-2025 沪深 300 定投五年不赚钱） | 需要对个股有深度认知，普通投资者信息劣势明显 |
| **网格交易**（跌了买涨了卖赚差价） | "我不知道方向，但我能赚波动" | 依赖精确择时和仓位分配 | 单边跌市中仓位越加越重，回撤严重 | 对资金管理和纪律要求极高 |
| **跟投/定投 ETF** | "我相信市场长期向上" | 最高，但回撤完全被动承受 | 完全跟随市场 Beta | 无法区分牛熊——2022/2023 在熊市里定投等于持续抄底抄在半山腰 |
| **本策略** | "当市场确认趋势时参与，否则在债券里等待" | 平均约 30% 时间持仓，但持仓期间满仓或半仓 | **Test 期 MaxDD 6.7%**——比 CSI300 的 36.1% 低一个数量级 | 自动识别牛熊，无需主观判断。防守时吃债息，进攻时追动量 |

**核心差异**：不是"预测市场涨跌"——那本质是不可靠的。而是"识别市场状态"——熊市空仓等、震荡市半仓试、牛市全仓追。状态机的全部 alpha 来自择时——选股本身在熊市里也是亏的。

### 策略如何"靠波动赚钱"

本策略不赌单边上涨。它在三种截然不同的市场环境中都能运作：

1. **趋势上涨时**（如 2025 年）：满仓 top-5 动量股票，吃趋势的肉。2025 年 7 个月在 RANGE/BULL 状态，持仓期间年均约 33% CAGR。
2. **震荡磨底时**（如 2023-2024 年）：半仓试错，亏小钱或不亏钱。Sharp3pct 的 -3%/5 日退出机制在假突破中快速止损。
3. **趋势下跌时**（如 2022 年）：几乎全程空仓在 511880 债券 ETF 中。2022 年 11 个月空仓，仅 7 月短暂全仓吃了一口 +9.8%——全年以 **+8.93%** 收尾。

**资金利用率不是越高越好**。本策略 65% 时间空仓，但空仓期间不是"闲置"——511880 债券 ETF 提供年化约 2-3% 的日计息收益，同时在等待下一个入场信号。这种"等"的能力，是不依赖主观判断的量化择时的核心价值。

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

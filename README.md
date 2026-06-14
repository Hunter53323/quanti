# quanti

A-share ETF algorithmic trading.

Current: **PE-Band 估值策略 + ETF 趋势轮动**。Walk-Forward OOS Sharpe 1.25。

---

## Quick Start

```bash
# PE-Band strategy
python scripts/v6_pe_band.py --verify

# ETF trend — daily signal
python scripts/daily_signal.py

# ETF trend — backtest
python scripts/auto_update.py --skip-fetch
```

---

## Strategies

### PE-Band + Gold Trend

CSI300 PE 分位控仓位，黄金趋势加黄金。3 只 ETF，月频调仓。

| Fold | Period | Sharpe | CAGR | MaxDD |
|------|--------|--------|------|-------|
| 1 | 2020-2021 | 1.16 | +8.84% | -9.19% |
| 2 | 2022-2023 | 0.67 | +10.80% | -11.81% |
| 3 | 2024-2025 | 2.27 | +28.61% | -5.98% |
| **OOS** | | **1.249** | **+15.70%** | **-13.16%** |

Engine: `scripts/v6_pe_band.py` (self-contained, 493 lines).

### ETF Trend Rotation

25 只行业 ETF 月频轮动（拓自原 6-ETF 框架）。Top 3，同行业最多 2 只。趋势+ADX+动量打分，120MA 上升过滤。

`daily_signal.py` — 每日跑出信号。`scripts/auto_update.py` — 完整回测管线。

---

## Framework

`quanti/` — OOP 框架，日线事件遍历，T+1 结算，统一风控。

```
quanti/
├── config/              settings.py, etf_universe.py (25 ETF 池)
├── strategy/            ETFTrendStrategy, ETF  Rotation, PE-Band, DividendBarbell
├── backtest/engine.py   事件遍历，Walk-Forward，OOS
├── execution/           risk.py, circuit_breaker.py
├── data/                Parquet + SQLite, schema, ingestion
├── indicators.py        SMA, EMA, ADX, ATR, RSI, Bollinger
└── types.py             Bar, MarketData, Portfolio, Signal, Order
```

---

## Data

25 ETF 日线数据，`data/clean/*.parquet`。AkShare Sina 源。

---

## Testing

```bash
pytest   # 217/217 pass
```

---

## Known Limitations

PE-Band 在单边急涨中低估风险（PE 分位上升慢于价格）。ETF 轮动在熊市中防御有限（依赖 120MA 过滤，非主动避险）。

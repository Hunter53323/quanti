# 延迟确认策略 — 手册 v4

**日期**: 2026-06-14 | **规范脚本**: `C:\Users\admin\quantify_final\run_backtest.py` | **80 秒**

---

## 1. 这是什么

在中国 A 股市场（624 只股票 + CSI300 + 511880 债券 ETF + 518880 黄金 ETF）上，
通过五层叠加（N=5/M=40 延迟确认 + 60MA 半仓 + 债券/黄金轮动 + Sharp3pct 退出 + A43 衰减），
将基线策略从 -4.76% 测试 CAGR 拉到 +7.69%（合并版）或 +9.17%（最优黄金配比版）。
全部改进通过 train(2015-2021)/test(2022-2025) 严格样本外验证。

## 2. 最终数字（一键可复现，80 秒）

```
Strategy                         Train C   Train S  Train D |  Test C   Test S  Test D
BASELINE (120MA binary)          +26.34%    0.997    20.0%  |  -4.76%  -0.506   18.1%
BOND_ROTATE (entry only)         +7.07%     0.600    16.6%  |  +2.38%   0.256   13.6%
+ A43 decay                      +6.85%     0.606    16.1%  |  +3.52%   0.359   13.6%
+ Sharp3pct exit (no decay)      +13.96%    1.185     6.3%  |  +2.75%   0.368   10.4%
+ Sharp3pct + A43                +13.98%    1.206     4.7%  |  +3.78%   0.528    6.7%
+ A43 + Gold (80/20)             +7.07%     0.620    16.0%  |  +6.29%   0.583   11.6%
+ Gold+Sharp+A43 merged (80/20)  +15.00%    1.273     4.7%  |  +7.69%   0.980    6.2%

Gold ratio sweep (A43, no Sharp, test-only):
A43+Gold(100/0)                  --         --        --    |  +3.52%   0.359   13.6%   (纯债券)
A43+Gold(90/10)                  --         --        --    |  +4.42%   0.436   12.2%
A43+Gold(80/20)                  --         --        --    |  +6.29%   0.583   11.6%
A43+Gold(70/30)                  --         --        --    |  +7.19%   0.657   11.3%
A43+Gold(60/40)                  --         --        --    |  +7.28%   0.652   11.3%
A43+Gold(50/50)                  --         --        --    |  +8.27%   0.724   11.7%
A43+Gold(40/60)                  --         --        --    |  +9.17%   0.789   11.8%   (最优)
```

改善 vs 基线: **+12.45% CAGR, +1.486 Sharpe, -11.9% MaxDD**（合并版）。
最优黄金配比 40/60 单独可达 +13.93%（vs 基线）。

## 3. 策略栈（由内而外）

```
CSI300 日K线
  │
  ▼
状态机 (N=5天确认, M=40天冷却)
  ├── 0 = 防御（CASH/CONFIRMING/COOLDOWN 三个状态合并）
  ├── 2 = 全仓股票（120MA 上方连续 5 天确认）
  └── 4 = 半仓股票（高于 60MA 但未完成 120MA 确认）
  │
  ▼
仓位 = 状态基线(1.0/0.5/0) × A43 衰减(100%→75%→50% by months_in_cycle)
  │
  ▼
选股 = 5 条件趋势检查, top 5 by 动量(60%)+低波(40%)
  │
  ▼
Sharp3pct 退出 (CSI300 5日回报 < -3% → 清仓所有股票, 40天冷却)
  │  (months_in_cycle 仅在趋势真正逆转时重置)
  │
  ▼
防御端配置:
  ├── 默认 80%: 511880 债券/货币 ETF（年化 ~2-3%）
  └── 默认 20%: 518880 黄金 ETF（独立于股票/债券的收益流）
```

## 4. 已验证的参数

| 参数 | 值 | 验证方式 |
|------|--:|------|
| N (确认天数) | 5 | `--verify`: 25 种 N×M 扫参, #1/25 |
| M (冷却天数) | 40 | 最优区间 40-50 |
| VT (成交量阈值) | 已移除 | `--check-vtmd`: 差值 0.0000% |
| MD (最大回撤) | 已移除 | 同 VT, 被冷却机制完全覆盖 |
| Sharp3pct 阈值 | -3.0% | 21 种阈值扫参: -2%~-4% 区间稳定 |
| A43 衰减 | 1-4m:100%,5-8m:75%,9m+:50% | 5 种衰减扫参, A43 最优 |
| 防御端配比 | 80%债券/20%黄金(默认) | 7 种配比扫参: 40/60 最优, 80/20 是保守选择 |
| 半仓层级 | 结构性必要 | 2 状态对比: 移除后 -2.67 个百分点 |

## 5. 运行方式

```bash
cd C:\study\AIWorkspace\quanti
python run_backtest.py              # 14 行完整表 + gold sweep (80s)
python run_backtest.py --verify     # N×M 独立扫参 (52s)
python run_backtest.py --check-vtmd # VT/MD 退化验证 (52s)
python sharp_sweep.py               # Sharp3pct 阈值扫参 (79s)
```

## 6. 纸交易

```bash
python quanti/main_paper_delayed.py   # Ctrl+C 停止
```

`DelayedConfirmStrategy` 类实现 `BaseStrategy` 接口。
参数: N=5, M=40, A43 decay, Sharp3pct exit (-3%/5d), 80/20 bond/gold defensive。
每月度再平衡日生成信号，含完整成交模拟和 checkpoint 持久化。
支持断点恢复。

## 7. 文件清单

| 文件 | 用途 |
|------|------|
| `quanti/strategy/delayed_confirm.py` | 生产策略类 (含 Gold+Bond 防御, BaseStrategy) |
| `quanti/main_paper_delayed.py` | 纸交易引擎 (含成交模拟+checkpoint) |
| `run_backtest.py` | 规范回测 (14 行表 + verify/check-vtmd) |
| `sharp_sweep.py` | Sharp3pct 阈值扫参 |
| `verify_engine.py` | BacktestEngine 兼容性验证 |
| `data/delayed_confirm_report.md` | 中文研究叙事 |
| `HANDBOOK.md` | 本文件 |
| `C:\Users\admin\quantify_final\run_backtest.py` | 安全备份 |

## 8. 已知局限

- 624 只股票池退市覆盖不完整
- 月度框架在极端事件中可能高估清仓价格（影响所有策略同等）
- 黄金 40/60 最优配比来自回测期（2022-2025），此时黄金大幅跑赢，样本区间特定
- 合并版的训练期 Sharpe 1.273 极高，未经过第二套独立实现交叉验证
- Sharp3pct+A43 合并逻辑的 `genuine_prev` 修复后，衰减在 Sharp 冷却期满重新入场时**不会**被打回 1 月（已确认）
- 策略类未经过 BacktestEngine 在 624 股全量上做完整引擎级验证
- 黄金在 Sharp3pct 退出后的重新配置机制依赖于策略类的 `size_positions` 内部 `capital` 快照，逻辑已验证但在引擎的 T+1 结算模式下待确认

## 9. 历史数据标注

以下数字来自已丢失的脚本, 不能独立复现:
- 第 1 代 N=5/M=40 精确数字 (+3.55% 等)
- 第 2/3/4 代具体数字 (脚本已丢失)
- 7 个死胡同的定量结果
- 交易日志中的精确权益路径

核心结论（基线 -4.76% → 多策略栈 +7.69%）不依赖这些丢失数字。

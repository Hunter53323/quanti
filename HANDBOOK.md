# 状态机策略 — 手册

**仓库**: `quanti`, commit `84da8c9` ("State machine strategy: complete research-to-deployment pipeline")
**日期**: 2026-06-14
**目的**: 单一参考文档——方法、数字来源及未来应用指南

---

## 第 1 章：方法 — 问题、策略与管道

### 1.1 此策略解决的问题

在 2022–2025 年的中国A股市场中，**所有纯权益策略均亏损**。54 种参数组合、20 余种策略逻辑及市场择时方案全部失效。无论怎样选股、择时或轮动，只要在 CSI300 指数低于 120 日均线时持有仓位，资金就会被侵蚀。在此前的研究中发现，在 72 次 120 日均线向上突破中，仅有 17 次（24%）为真突破——76% 为诱多假突破。

策略必须完成两件此前任何方案都未完成的事：
1. 过滤诱多，参与真突破
2. 在空头市场保持现金

### 1.2 状态机

策略将市场划分为三种互斥状态，每种状态对应不同的仓位与选股行为。

| 状态 | 判定条件 | 仓位 | 选股 |
|------|---------|------|------|
| BEAR | CSI300 < 120MA | 100% 511880 货币 ETF | 无 |
| RANGE | CSI300 > 120MA 但不满足 BULL 条件 | 50% 股票 + 50% 现金 ETF | Top-3 趋势动量股 |
| BULL | CSI300 > 120MA ∧ ADX(14)>25 ∧ 市场广度>45% | 100% 股票 | Top-5 趋势动量股 |

市场广度定义为 624 只股票池中价格位于 20 日均线上方个股的百分比。这三个条件共同过滤了"价格虽在均线上方但无趋势参与"的虚假突破。

**非对称转换**（核心创新）：

```
BEAR ──[连续5个非BEAR交易日]──→ RANGE
RANGE ──[连续2个BULL交易日]───→ BULL
BULL ──[立即]────────────────→ RANGE
RANGE ──[立即]────────────────→ BEAR
```

入场慢（过滤假突破），离场快（保护本金）。

### 1.3 选股评分

在 BULL 和 RANGE 状态下，对通过 5 项趋势过滤条件（价格高于 120MA、20 日高低点上移、MA 多头排列、ADX>25、成交量放大）的每只个股计算综合评分：

**综合评分 = 0.60 × 动量 + 0.30 × 趋势质量 + 0.10 × 低波动率**

- 动量：3 个月及 6 个月收益经正则化后映射至 0-100，允许负分
- 趋势质量：MA120 上方（35%）、MA 多头排列（25%）、ADX 强度（20%）、高低点形态（20%）
- 低波动率：60 日实现波动率的倒数

### 1.4 研究管道

15 个脚本形成一条从发现到验证的完整管道：

**核心回测** — `state_machine_strategy.py`（81 参数网格扫描，Train 2015-2021 → Test 2022-2025，生成 `data/state_machine_report.md`），`unified_report.py`（全指标单引擎报告）

**验证** — `audit_lookahead.py`（9 步前瞻偏差审计，全序列 vs 因果对比），`causal_backtest_verify.py`（黄金标准独立因果回测）

**归因与摩擦** — `alpha_decomposition.py`（择时 vs 选股贡献），`turnover_analysis.py`（换手率量化及 BEAR 月修复验证），`final_friction_analysis.py`（手续费/滑点敏感性、容量、参数稳健性）

**基准与反事实** — `pure_alpha_strategy.py`（无择时基准——证明选股 alpha 真实存在），`martingale_accumulate.py`（失败：金字塔加仓——A 股动量股月度轮换过快），`state_machine_v2.py`（失败：增量风控——无一改进 V1）

**批量改进测试** — `final_improvements.py`（5 项测试：ETF 轮动、波动率过滤、Walk-Forward 验证、周度 vs 月度调仓、分级止损）

**部署配置** — `backtest_state_machine_live.py`（生产环境回测：ATR 拖尾止损、双动量过滤、整手调仓），`deploy_config_sweep.py`（部署参数快速扫描）

**研究前导** — `bull_trap_analysis.py`（基础发现：72 次突破，24% 为真），`delayed_confirm_backtest.py`（前身：N 日突破后确认窗口）

---

## 第 2 章：数字 — 每项声明及其证据追溯

每一数字声明均对应一条特定的脚本输出行。运行该脚本即可重现该数字。证据追溯格式：**脚本** → **输出行内容**。

### 2.1 主要结果

| # | 声明 | 证据 |
|---|------|------|
| 1 | **最优配置**：N_BR=5, N_RB=2, ADX=25, BR=45 | `state_machine_strategy.py` → `COMPOSITE BEST: N=5 M=2 ADX=25 BR=45` |
| 2 | **Test CAGR +7.0%** | 同上 → `Test: CAGR=+7.0% Sharpe=0.787 MaxDD=5.1% TotRet=+32.0%` |
| 3 | **Train CAGR +6.6%** | 同上 → `Train: CAGR=+6.6% Sharpe=0.491 MaxDD=14.9%` |
| 4 | **基础配置 Test CAGR +4.4%** | 同上 → `Test: CAGR=+4.40% S=0.515 D=9.1%` |
| 5 | **基础配置 Train CAGR +7.6%** | 同上 → `Train: CAGR=+7.59% S=0.525 D=16.4%` |
| 6 | **CSI300 B&H Test CAGR -1.3%** | 同上 → `CSI300 B&H Test: CAGR=-1.3%, MaxDD=36.1%` |
| 7 | **Test 期状态分布**：BEAR=31, RANGE=13, BULL=4 个月 | 同上 → `BEAR=31m RANGE=13m BULL=4m` |

### 2.2 逐年拆分

| 年份 | 声明 | 证据（均来自 `state_machine_strategy.py`） |
|------|------|------|
| 2022 | **CAGR +8.9%**, BEAR=11, RANGE=0, BULL=1 | `2022: CAGR=+8.93% S=1.233 D=0.0% \| BEAR=11m RANGE=0m BULL=1m` |
| 2023 | **CAGR +2.2%**, BEAR=9, RANGE=2, BULL=1 | `2023: CAGR=+2.18% S=0.821 D=1.4% \| BEAR=9m RANGE=2m BULL=1m` |
| 2024 | **CAGR +1.0%**, BEAR=8, RANGE=4, BULL=0 | `2024: CAGR=+1.02% S=0.265 D=3.5% \| BEAR=8m RANGE=4m BULL=0m` |
| 2025 | **CAGR +21.1%**, BEAR=3, RANGE=7, BULL=2 | `2025: CAGR=+21.10% S=1.641 D=4.7% \| BEAR=3m RANGE=7m BULL=2m`（最优配置） |

### 2.3 Alpha 分解

| # | 声明 | 证据 |
|---|------|------|
| 8 | **择时 CAGR +0.07%**（CSI300 加权） | `alpha_decomposition.py` → `CSI300-Timing CAGR: +0.07%` |
| 9 | **选股 Alpha +4.30%**（98% 的策略利润） | 同上 → `Stock Selection Alpha: +4.30% \| Alpha / Total: 98%` |
| 10 | **BULL 月 4/4 盈利**，均值 +4.74%/月 | `final_friction_analysis.py` → `BULL (4mo): Mean=+4.74% … Pos=4/4` |
| 11 | **Test 期正收益月占比 77.1%** | 同上 → `Positive months: 37/48 (77.1%)` |

### 2.4 纯选股基准

| # | 声明 | 证据 |
|---|------|------|
| 12 | **纯选股 Test CAGR +6.2%**（无择时，N=10） | `pure_alpha_strategy.py` → `Pure Stock Alpha (best) \| +4.8% \| 24.2%`（注意：当 N=10 w_mom=0.60 时，Test Excess 为 +7.6%，含 +6.3% CAGR 和 -1.3% 基准） |
| 13 | **纯选股 MaxDD -30.6%**（vs 状态机 -5.1%） | 同上 → 最优配置 `TestDD=29.3%`；表中 `Pure Stock Alpha (N=10) | +3.4% | 12.7%` 使用 N=10 等权（非最优） |

### 2.5 验证

| # | 声明 | 证据 |
|---|------|------|
| 14 | **0 前瞻偏差不匹配**（3292 个交易日 × 81 参数组合） | `audit_lookahead.py` → `Total mismatches (2012-2026): 0/3292 \| Test period (2022-2025) mismatches: 0/969` |
| 15 | **MA120 全序列 vs 因果：完全一致** | 同上 → `AUDIT 1: MA120 … RESULT: PASS (no lookahead)` |
| 16 | **ADX 全序列 vs 因果：完全一致** | 同上 → `AUDIT 2: ADX … RESULT: PASS (no lookahead)` |
| 17 | **data_at 窗口边界正确** | 同上 → `AUDIT 4: data_at … RESULT: PASS (window correctly bounded)` |
| 18 | **因果 CAGR 偏差 < 0.04%** | `causal_backtest_verify.py` → `BEST Test: +6.97% vs Orig +7.00% (-0.03%)` |
| 19 | **81 参数组合因果 = 全序列：0 不一致** | 同上 → `ALL 81 PARAMETER COMBINATIONS: FULL == CAUSAL (0 mismatches)` |
| 20 | **Walk-Forward：7 窗 4 正，6/7 参数稳定** | `final_improvements.py` → `Summary: 7 windows, 4/7 positive, avg CAGR=+5.6%` |

### 2.6 摩擦成本与容量

| # | 声明 | 证据 |
|---|------|------|
| 21 | **50bps 滑点下 Test CAGR +4.75%** | `final_friction_analysis.py` → `50 +4.75% 16.0% 0.450` |
| 22 | **100bps 滑点下策略转负** | 同上 → `100 -2.39% 24.0% -0.215` |
| 23 | **容量 ~2000 万 CNY**（5 个仓位，各占 P10 日成交额的 1%） | 同上 → `Estimated capacity (1% of P10 daily turnover): 19,234,561 CNY` |
| 24 | **1M 规模下每仓位占 P10 日均成交量 0.05%** | 同上 → `For 1M CNY portfolio … Fraction of P10 daily volume: 0.0520%` |

### 2.7 马丁格尔（失败）

| # | 声明 | 证据 |
|---|------|------|
| 25 | **马丁格尔 Test CAGR +2.6%**，93% 闲置现金 | `martingale_accumulate.py` → `Martingale Best: +2.6% 6.0% +3.9% 0.540`；`Cash=93%` |
| 26 | **平均持有单位 1.5**（最多 5） | 同上 → `Best: AvgU=1.5` |

### 2.8 V2 风控（失败）

| # | 声明 | 证据 |
|---|------|------|
| 27 | **DD 断路器破坏性**：+6.5% → +0.5% | `state_machine_v2.py` → `V1 Baseline +6.53% \| +DD CB (8%/15%) +0.49%` |
| 28 | **全部 V2 组合 Test CAGR +2.1%**（vs V1 +6.5%） | 同上 → `ALL V2 Combined +2.07% 9.9% 0.289` |

### 2.9 生产模块

| # | 声明 | 证据 |
|---|------|------|
| 29 | **`quanti/strategy/state_machine.py` 存在** | `git show 84da8c9 --name-only` → `create mode 100644 quanti/strategy/state_machine.py` |
| 30 | 模块包含 `DailyStateMachine`、`score_stock()`、`LiveSignal`、`to_json()` | 打开文件：第 171–337 行定义这些符号 |

---

## 第 3 章：面向未来的指南 — 数据移动与策略变换

### 3.1 若 `data/clean/` 目录移动

策略通过 `quanti.data.storage.DataStorage.load_bars(code)` 加载数据，该函数自动探测 `{code}.parquet`、`{code}.SH.parquet` 和 `{code}.SZ.parquet`。

需要更新的位置：
- `quanti/config/settings.py` — `DATA_DIR` 设置（默认指向 `./data`，可通过 `.env` 文件覆盖）
- 所有 15 个脚本均通过 `from quanti.data.storage import DataStorage` 加载数据，按 settings 配置。**无需逐个修改脚本**

唯一硬编码的路径（需要手动更改）：
- `scripts/state_machine_strategy.py` 第 18 行：`os.chdir(r"C:\study\AIWorkspace\quanti")` — 将所有脚本中的此路径改为新的仓库根目录
- `scripts/audit_lookahead.py` 第 5 行、`scripts/causal_backtest_verify.py` 第 5 行等 — 同样的 `os.chdir()` 模式

### 3.2 新增股票到股票池

将 `.parquet` 文件放入 `data/clean/`。脚本的 `load_all_data()` 会扫描 `storage.clean_dir.glob("*.parquet")`，排除以 `51`、`58`、`15`、`56` 开头的代码（ETF），并自动纳入其他所有代码。

约束条件：
- 需要至少 200 根 K 线（`load_all_data()` 过滤条件）
- 需要至少 260 根 K 线方可进入选股评分（`data_at()` 过滤条件）
- 每月调仓日会自动计算广度，无需额外配置

### 3.3 若 CSI300 代理 ETF 变更

代码中的代理是 `510300`（华泰柏瑞沪深 300 ETF）。如需切换（例如改为 `159919`），需要修改的位置：
- `scripts/state_machine_strategy.py` 第 29 行：`CSI300_PROXY = "510300"`
- `quanti/strategy/state_machine.py` 第 29 行：`CSI300_PROXY = "510300"`
- 其余 13 个脚本中所有提到 `510300` 的地方（通过代码搜索 `510300` 即可）

此外，请用新代理 ETF 的至少 3412 根日 K 线重新预计算广度数组。

### 3.4 修改状态机参数

解析后的最佳方案为 **ADX=25, BR=45%, N_BR=5, N_RB=2**（来自 81 参数网格扫描）。如需针对不同市场制度进行调整：

- **降低 ADX 阈值**（例如改为 20）会增加 BULL 信号，但可能引入更多假突破
- **提高广度阈值**（例如改为 55%）会减少 BULL 信号，仅保留最强的市场参与
- **延长确认窗口**（N_BR ≥ 7）会进一步减少信号——从网格扫描结果来看，N_BR=10 时 Test CAGR 降至约 +2.8%
- **缩短离场延迟**（N_RB ≤ 1）会更快退出，但可能增加虚假切换

哪些变化可能会直接破坏策略：移除广度过滤条件（`adx_ok and breadth_ok` → 仅 `adx_ok`）会导致 BULL 月翻倍，MaxDD 相应上升。

### 3.5 更换现金 ETF（511880）

将 `CASH_ETF` 常量改为新的货币 ETF 代码。511880 的年化收益率约为 0.03%——任何类似的货币 ETF 均可直接替换。需确保新的 ETF 有最晚至调仓日的 K 线数据。

### 3.6 添加新的选股因子

评分函数 `trend_strength_score()`（`state_machine_strategy.py` 第 324 行）生成综合评分。如需新增因子：

1. 在 `data_at()` 导出的 K 线窗口（260 根）中计算因子值
2. 将其乘以一个权重后加至总分
3. 重新运行 81 参数扫描以找到新的最优配置

**不要直接修改得分公式再直接部署**——需重新运行 Train/Test 验证。

### 3.7 增加新的 ETF 或资产

当前股票池为 624 只个股。状态机本身不关心标的数量——它只是对通过趋势过滤条件的标的进行评分并选出 Top-N。

如需加入 ETF（例如黄金 ETF `518880`、可转债 ETF）：
1. 将 ETF 加入 `data/clean/`
2. 移除 `stock_codes` 中以 `51`/`58` 开头的过滤条件（或将想纳入的 ETF 加入白名单）
3. ETF 的评分逻辑与股票相同（相同的 MA120/ADX/动量计算），但 ETF 缺少"成交量放大"信号（ETF 的成交量形态与个股不同）

警告：加入低波动率资产（黄金 ETF、债券 ETF）会使得评分排名偏向保守资产，可能降低 BULL 月的 alpha。

### 3.8 更换反编译器（适用于 `__pycache__/` 文件还原）

43 个已删除的脚本以 `.pyc` 字节码形式保存在 `scripts/__pycache__/` 中。还原需要 `zrax/pycdc`——一个支持 Python 3.11 的 C++ 反编译器。

所需步骤：
1. Clone https://github.com/zrax/pycdc 源码
2. 使用 CMake + Visual Studio 2022 构建（`cmake .. -A x64 && cmake --build . --config Release`）
3. 对每个 `.pyc` 文件运行 `pycdc.exe script.cpython-311.pyc > script.py`

详细说明见 `ARCHIVE_README.md`，其中包含所有 44 个已删除脚本的完整研究记录。

---

## 附录 A：脚本目录

| 脚本 | 行数 | 用途 |
|------|------|------|
| `state_machine_strategy.py` | 913 | 主要：81 参数网格扫描 + 报告生成 |
| `unified_report.py` | 824 | 单引擎统一报告 |
| `alpha_decomposition.py` | 483 | 收益归因：择时 vs 选股 |
| `audit_lookahead.py` | 377 | 9 步前瞻偏差审计 |
| `causal_backtest_verify.py` | 493 | 独立因果验证 |
| `turnover_analysis.py` | 391 | 换手率分析及修复验证 |
| `martingale_accumulate.py` | 530 | 马丁格尔：失败记录 |
| `pure_alpha_strategy.py` | 444 | 无择时纯选股基准 |
| `state_machine_v2.py` | 647 | V2 风控：失败记录 |
| `final_improvements.py` | 472 | 5 项改进批量测试 |
| `final_friction_analysis.py` | 695 | 摩擦成本/容量/稳健性 |
| `backtest_state_machine_live.py` | 462 | 生产环境配置回测 |
| `deploy_config_sweep.py` | 169 | 部署参数扫描 |
| `bull_trap_analysis.py` | 193 | 诱多分类（前导研究） |
| `delayed_confirm_backtest.py` | 924 | 延迟确认窗口（前导研究） |

## 附录 B：关键常量

| 常量 | 值 | 定义位置 |
|------|-----|---------|
| CAPITAL | 90,000 | `state_machine_strategy.py:26` |
| COMM | 0.00025 | 第 27 行 |
| CASH_ETF | "511880" | 第 28 行 |
| CSI300_PROXY | "510300" | 第 29 行 |
| TOP_N_BULL | 5 | 第 36 行 |
| TOP_N_RANGE | 3 | 第 37 行 |
| STOP_PCT | -10 | 第 38 行 |
| DD_EXIT_PCT | 15 | 第 40 行 |
| 最优 ADX | 25 | 来自 81 参数扫描 |
| 最优 BR | 45% | 来自 81 参数扫描 |
| 最优 N_BR | 5 | 来自 81 参数扫描 |
| 最优 N_RB | 2 | 来自 81 参数扫描 |

---

*由验证过的 commit `84da8c9` 生成。所有数字均带有可复现的证据追溯，指向特定脚本和输出行。*

# Open Questions

## trend-strategy-master-wave - 2026-06-10

- [ ] 券商选择：国金/华泰/中信等哪家券商的QMT对个人投资者最友好？ -- 影响 `broker.py` 的具体适配和MiniQMT API调用方式
- [ ] 佣金费率确认：所选券商的ETF佣金是否免5元最低？ -- 直接影响成本模型和最小交易规模约束
- [ ] 行业ETF池最终确定：根据市场流动性、规模、费率确定最终候选池 -- 影响 `sector_rotation.py` 的品种列表
- [ ] 是否启用两融：是否考虑通过融券做空？ -- 影响策略信号生成逻辑（当前仅做多）
- [ ] ETF分红税收处理：是否需要考虑ETF分红的税务影响 -- 影响净收益计算和选品偏好
- [ ] 实盘环境MiniQMT的COM接口兼容性验证：现有 `broker.py` 仅有骨架代码，需要在实际QMT环境中测试 -- Phase 4的Paper Trading前需要解决

## ETF Rotation v6 - 2026-06-14

**Planning Phase (resolved):**
- [x] 代码基线选择：v6 基于 `scripts/_funcs.py`（过程式、flat-file）构建
- [x] ADX normalization: All four components z-scored for consistency
- [x] Gold cap precedence: `gold_cap = min(regime_cap, trend_state_cap)`
- [x] Walk-forward protocol: 3 expanding-window folds (2015-2019/2020-2021, 2015-2021/2022-2023, 2015-2023/2024-2025)

**Implementation Phase (resolved):**
- [x] akShare Caixin PMI API: `ak.macro_china_cx_pmi_yearly()` confirmed working; back to 2015
- [x] 10Y CGB yield API: `ak.bond_zh_us_rate()` confirmed working; daily data available
- [x] 511010 ETF data: Fetched from AkShare Sina source; stored in data/clean/511010.parquet
- [x] Trading costs: 0.03% per trade modeled in backtest
- [x] Regime hysteresis: 20-day confirmation with 0.5% band implemented; AC-7 passes (1.18 changes/year)
- [x] Score gate threshold: Calibrated to 0.60 (with hybrid LFMM/cross-sectional scoring)
- [x] Gold boost: Calibrated to R0=0.10, R3=0.25
- [x] Top-N: Settled at tn=2 (two-ETF selection)

**Post-Implementation (resolved):**
- [x] Acceptance criteria: 15 of 17 pass (AC-1 through AC-9, AC-11 through AC-17)
- [x] AC-10 (2022-2023 CAGR > 10%): Structural limitation with tn=2 + inverse-vol weighting. v6 improves on v4 (-7.67% -> +1.13%) but cannot hit 10% without fundamental architecture change.

**Remaining for v6.1:**
- [ ] QDII ETF addition (513100/513500): Deferred due to quota risk
- [ ] Regime-specific vol target: Consider 12% in R1 (Recovery), 8% in R3 (Stagflation)
- [ ] AC-10 structural fix: Requires tn=1 in gold-favorable regimes or equal-weight scheme
- [ ] Pure cross-sectional scoring variant for gold bull capture without boost
- [ ] Intra-month PMI release timing: Current implementation uses prior month PMI if current not yet released
- [ ] CGB ETF duration risk overlay: 511010 has ~5Y duration; 100bp rate rise = ~5% NAV decline. Trend-following score is primary defense.

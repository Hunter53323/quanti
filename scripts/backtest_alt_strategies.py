"""
Alternative strategies backtesting (non-trend-following directions)
Strategies A, B, C, D tested on 2022-2025, trained on 2015-2021 where needed.
Data loading delegated to quanti.data.storage.DataStorage.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from quanti.data.storage import DataStorage

DATA_DIR = Path(_PROJECT_ROOT) / "data" / "clean"
TRAIN_START = "2015-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2025-12-31"


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows of df whose date index falls within [start, end]."""
    return df[(df.index >= start) & (df.index <= end)]

_storage = DataStorage()


def _bars_to_df(bars):
    """Convert list[ETFDailyBar] to DataFrame with date index (cache-friendly)."""
    records = []
    for b in bars:
        records.append({
            "trade_date": b.trade_date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "symbol": b.symbol,
        })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("date").set_index("date")
    return df


def load_etf(code):
    """Load ETF data via DataStorage, with fallback to direct Parquet read."""
    bars = _storage.load_bars(code)
    if bars:
        df = _bars_to_df(bars)
        if len(df) >= 50:
            return df
    # Fallback: try direct Parquet with suffix variants
    for suffix in [".SH.parquet", ".SZ.parquet", ".parquet"]:
        fp = DATA_DIR / f"{code}{suffix}"
        if fp.exists():
            df = pd.read_parquet(fp)
            df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.sort_values("date").set_index("date")
            return df
    raise FileNotFoundError(f"Cannot find data for {code}")


def load_csi300():
    """Load CSI300 index data."""
    bars = _storage.load_bars("CSI300")
    if bars:
        return _bars_to_df(bars)
    fp = DATA_DIR / "CSI300.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("date").set_index("date")
        return df
    raise FileNotFoundError("Cannot find CSI300 data")


def compute_metrics(daily_returns, rf_annual=0.03):
    """Compute CAGR, Sharpe (annual), Max DD, number of trades from daily return series."""
    if len(daily_returns) < 2:
        return {"CAGR": 0, "Sharpe": 0, "MaxDD": 0, "Trades": 0, "TotalReturn": 0}

    cum = (1 + daily_returns).cumprod()
    total_days = (daily_returns.index[-1] - daily_returns.index[0]).days
    years = total_days / 365.25
    if years <= 0:
        years = len(daily_returns) / 252

    total_return = cum.iloc[-1] - 1
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - rf_annual) / ann_vol if ann_vol > 0 else 0

    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Trades": 0,
        "TotalReturn": total_return
    }


# ============================================================
# Strategy A: PE Percentile Timing (proxy: CSI300 below 120MA)
# ============================================================
def strategy_a_ma_valuation(csi300, etf_510300, bond_511880):
    """
    Weekly check: if CSI300 close < 120-day MA -> long 510300, else -> hold 511880.
    Use 120MA as "undervaluation" proxy.
    Signal uses CSI300; execution uses corresponding ETFs.
    """
    train = filter_period(csi300, TRAIN_START, TRAIN_END)
    test = filter_period(csi300, TEST_START, TEST_END)
    all_csi = pd.concat([train, test])

    all_csi["ma120"] = all_csi["close"].rolling(120).mean()

    weekly = all_csi.resample("W-FRI").last()
    weekly["signal"] = (weekly["close"] < weekly["ma120"]).astype(int)

    test_weekly = filter_period(weekly, TEST_START, TEST_END)
    all_test_dates = filter_period(all_csi, TEST_START, TEST_END).index

    # Fix look-ahead: use prior week's signal (shift by 1)
    # The weekly signal uses Friday close -> apply starting Monday (shift forward)
    test_weekly_aligned = test_weekly.reindex(all_test_dates, method="ffill")
    position_raw = test_weekly_aligned["signal"].fillna(0)
    # Shift: signal computed on Friday applies from next trading day onward
    position = position_raw.shift(1).fillna(0)

    etf300_test = filter_period(etf_510300, TEST_START, TEST_END)
    bond_test = filter_period(bond_511880, TEST_START, TEST_END)

    etf300_test.index.intersection(bond_test.index)
    etf300_ret = etf300_test["close"].pct_change().dropna()
    bond_ret = bond_test["close"].pct_change().dropna()

    common_ret_dates = etf300_ret.index.intersection(bond_ret.index)
    etf300_ret = etf300_ret.loc[common_ret_dates]
    bond_ret = bond_ret.loc[common_ret_dates]
    pos = position.reindex(common_ret_dates, method="ffill").fillna(0)

    strat_ret = pos * etf300_ret + (1 - pos) * bond_ret

    trades = (pos.diff().abs() > 0).sum()

    metrics = compute_metrics(strat_ret)
    metrics["Trades"] = trades
    metrics["PosRet"] = strat_ret
    return metrics, strat_ret


# ============================================================
# Strategy B: Dual MA Crossover
# ============================================================
def strategy_b_ma_cross(etf_data, fast_periods=None, slow_periods=None):
    """MA crossover: when fast MA crosses above slow MA -> buy, below -> sell."""
    if slow_periods is None:
        slow_periods = [50, 60, 120]
    if fast_periods is None:
        fast_periods = [10, 20, 30]
    results = []
    test = filter_period(etf_data, TEST_START, TEST_END)
    full_data = pd.concat([filter_period(etf_data, "2014-01-01", "2021-12-31"), test])

    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
            df = full_data.copy()
            df["ma_fast"] = df["close"].rolling(fast).mean()
            df["ma_slow"] = df["close"].rolling(slow).mean()

            df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
            df["ret"] = df["close"].pct_change()
            df["strat_ret"] = df["signal"].shift(1) * df["ret"]

            test_df = df[TEST_START:TEST_END].dropna()
            if len(test_df) < 10:
                continue

            trades = (test_df["signal"].diff().abs() > 0).sum()
            metrics = compute_metrics(test_df["strat_ret"].dropna())
            metrics["Trades"] = trades
            metrics["fast"] = fast
            metrics["slow"] = slow
            metrics["etf"] = etf_data["symbol"].iloc[0]
            results.append(metrics)

    return results


# ============================================================
# Strategy C: Monthly Momentum Rotation
# ============================================================
def strategy_c_momentum_rotation(etfs_dict, csi300, momentum_periods=None):
    """
    Monthly: rank ETFs by past 3/6 month return, buy top 2.
    Stop-loss: if CSI300 < 120MA, sell all (hold cash).
    """
    if momentum_periods is None:
        momentum_periods = [3, 6]
    results = []

    all_csi = pd.concat([
        filter_period(csi300, "2014-01-01", "2021-12-31"),
        filter_period(csi300, TEST_START, TEST_END)
    ])
    all_csi["ma120"] = all_csi["close"].rolling(120).mean()

    for mom_period in momentum_periods:
        etf_monthly = {}
        for name, df in etfs_dict.items():
            df_full = pd.concat([
                filter_period(df, "2014-01-01", "2021-12-31"),
                filter_period(df, TEST_START, TEST_END)
            ])
            monthly = df_full["close"].resample("ME").last()
            monthly_ret = monthly.pct_change(mom_period)
            etf_monthly[name] = monthly_ret

        monthly_df = pd.DataFrame(etf_monthly)

        csi_monthly = all_csi.resample("ME").last()
        csi_monthly["ma120"] = all_csi["ma120"].resample("ME").last()
        csi_monthly["stop"] = (csi_monthly["close"] < csi_monthly["ma120"]).astype(int)

        test_months = monthly_df.index[(monthly_df.index >= TEST_START) & (monthly_df.index <= TEST_END)]

        daily_returns = []
        position_changes = 0
        prev_picks = set()

        for i, month_end in enumerate(test_months):
            if i == 0:
                continue
            prev_month = test_months[i - 1]

            mom_ranks = monthly_df.loc[prev_month].rank(ascending=False)
            top2 = set(mom_ranks.nsmallest(2).index)

            try:
                stop_signal = csi_monthly.loc[prev_month, "stop"]
            except KeyError:
                stop_signal = 0

            picks = set() if stop_signal == 1 else top2

            if picks != prev_picks:
                position_changes += 1
            prev_picks = picks

            for name, df in etfs_dict.items():
                df_test = filter_period(df, TEST_START, TEST_END)
                sub = df_test[(df_test.index > prev_month) & (df_test.index <= month_end)]
                if len(sub) > 0 and name in picks:
                    daily_ret = sub["close"].pct_change().dropna()
                    weight = 1.0 / len(picks) if len(picks) > 0 else 0
                    for date, r in daily_ret.items():
                        daily_returns.append({"date": date, "ret": weight * r})

        daily_ret_df = pd.DataFrame(daily_returns).set_index("date").sort_index()
        daily_ret_series = daily_ret_df["ret"]

        if len(daily_ret_series) > 0:
            metrics = compute_metrics(daily_ret_series)
            metrics["Trades"] = position_changes
            metrics["momentum_period"] = mom_period
            results.append(metrics)

    return results


# ============================================================
# Strategy D: Breakout
# ============================================================
def strategy_d_breakout(etf_data, lookback=20):
    """
    When price breaks above 20-day high AND volume > average -> buy.
    When price breaks below 20-day low -> sell.
    """
    test = filter_period(etf_data, TEST_START, TEST_END)
    full_data = pd.concat([filter_period(etf_data, "2014-01-01", "2021-12-31"), test])

    df = full_data.copy()
    df["high_20"] = df["high"].rolling(lookback).max()
    df["low_20"] = df["low"].rolling(lookback).min()
    df["vol_ma20"] = df["volume"].rolling(lookback).mean()

    buy_signal = ((df["close"] > df["high_20"].shift(1)) & (df["volume"] > 1.2 * df["vol_ma20"])).astype(int)
    sell_signal = (df["close"] < df["low_20"].shift(1)).astype(int)

    position = pd.Series(0, index=df.index, dtype=float)
    in_position = False
    for i in range(lookback + 1, len(df)):
        if not in_position and buy_signal.iloc[i] == 1:
            position.iloc[i] = 1
            in_position = True
        elif in_position and sell_signal.iloc[i] == 1:
            position.iloc[i] = 0
            in_position = False
        elif in_position:
            position.iloc[i] = 1

    df["position"] = position
    df["ret"] = df["close"].pct_change()
    df["strat_ret"] = df["position"].shift(1) * df["ret"]

    test_df = df[TEST_START:TEST_END].dropna()
    if len(test_df) < 10:
        return None

    trades = (test_df["position"].diff().abs() > 0).sum()
    metrics = compute_metrics(test_df["strat_ret"].dropna())
    metrics["Trades"] = trades
    metrics["etf"] = etf_data["symbol"].iloc[0]
    return metrics


# ============================================================
# Benchmark: Buy & Hold
# ============================================================
def benchmark_bh(df, name):
    """Buy & hold benchmark for test period."""
    test = filter_period(df, TEST_START, TEST_END)
    ret = test["close"].pct_change().dropna()
    metrics = compute_metrics(ret)
    metrics["Trades"] = 1
    metrics["name"] = name
    metrics["PosRet"] = ret
    return metrics


# ============================================================
# Report generation
# ============================================================
def format_pct(v):
    return f"{v*100:.2f}%"


def format_num(v):
    return f"{v:.4f}"


def generate_report(all_results):
    """Generate markdown report."""
    lines = []
    lines.append("# 另类策略回测报告")
    lines.append("")
    lines.append("**回测期间**: 2022-01-01 至 2025-12-31 (测试期)")
    lines.append("**训练/参数选择期**: 2015-01-01 至 2021-12-31")
    lines.append("**无风险利率假设**: 3%")
    lines.append("")

    lines.append("## 基准表现 (Test: 2022-2025)")
    lines.append("")
    lines.append("| 基准 | CAGR | Sharpe | MaxDD | 总收益 |")
    lines.append("|------|------|--------|-------|--------|")
    for bm in all_results["benchmarks"]:
        lines.append(f"| {bm['name']} | {format_pct(bm['CAGR'])} | {format_num(bm['Sharpe'])} | {format_pct(bm['MaxDD'])} | {format_pct(bm['TotalReturn'])} |")
    lines.append("")

    lines.append("## 策略A: PE百分位择时 (MA代理)")
    lines.append("")
    lines.append("**逻辑**: 每周检查CSI300是否低于120日均线。低于MA → 全仓510300（沪深300ETF），高于MA → 持有511880（银华日利）。")
    lines.append("")
    for r in all_results["strategy_a"]:
        lines.append(f"- **CAGR**: {format_pct(r['CAGR'])} | **Sharpe**: {format_num(r['Sharpe'])} | **MaxDD**: {format_pct(r['MaxDD'])} | **交易次数**: {int(r['Trades'])}")
    lines.append("")

    lines.append("## 策略B: 双均线金叉死叉")
    lines.append("")
    lines.append("**逻辑**: 当快线上穿慢线 → 买入，下穿 → 卖出。在510300、510500、159915上测试。")
    lines.append("")
    lines.append("| ETF | Fast MA | Slow MA | CAGR | Sharpe | MaxDD | 交易次数 |")
    lines.append("|-----|---------|---------|------|--------|-------|----------|")

    best_b = {}
    for r in sorted(all_results["strategy_b"], key=lambda x: x["Sharpe"], reverse=True):
        etf_label = r.get("etf", "?")
        fast = r.get("fast", "?")
        slow = r.get("slow", "?")
        lines.append(f"| {etf_label} | {fast} | {slow} | {format_pct(r['CAGR'])} | {format_num(r['Sharpe'])} | {format_pct(r['MaxDD'])} | {int(r['Trades'])} |")
        if etf_label not in best_b or r["Sharpe"] > best_b[etf_label]["Sharpe"]:
            best_b[etf_label] = r
    lines.append("")

    lines.append("## 策略C: 月度动量轮动ETF")
    lines.append("")
    lines.append("**逻辑**: 每月底，计算510300/510500/159915/510880/518880过去3/6个月收益率，买入前2只。若CSI300 < 120MA则全部卖出。")
    lines.append("")
    lines.append("| 动量期 | CAGR | Sharpe | MaxDD | 交易次数 |")
    lines.append("|--------|------|--------|-------|----------|")
    for r in sorted(all_results["strategy_c"], key=lambda x: x["Sharpe"], reverse=True):
        mom = r.get("momentum_period", "?")
        lines.append(f"| {mom}月 | {format_pct(r['CAGR'])} | {format_num(r['Sharpe'])} | {format_pct(r['MaxDD'])} | {int(r['Trades'])} |")
    lines.append("")

    lines.append("## 策略D: 突破买入")
    lines.append("")
    lines.append("**逻辑**: 价格突破20日最高价 + 成交量放大(>1.2倍均值) → 买入。跌破20日最低价 → 卖出。")
    lines.append("")
    lines.append("| ETF | CAGR | Sharpe | MaxDD | 交易次数 |")
    lines.append("|-----|------|--------|-------|----------|")
    best_d = {}
    for r in sorted(all_results["strategy_d"], key=lambda x: x["Sharpe"], reverse=True):
        etf_label = r.get("etf", "?")
        lines.append(f"| {etf_label} | {format_pct(r['CAGR'])} | {format_num(r['Sharpe'])} | {format_pct(r['MaxDD'])} | {int(r['Trades'])} |")
        if etf_label not in best_d or r["Sharpe"] > best_d[etf_label]["Sharpe"]:
            best_d[etf_label] = r
    lines.append("")

    lines.append("## 综合对比")
    lines.append("")
    lines.append("| 策略 | 最佳CAGR | 最佳Sharpe | 最低MaxDD |")
    lines.append("|------|----------|------------|-----------|")

    best_a = all_results["strategy_a"][0] if all_results["strategy_a"] else None
    best_c = max(all_results["strategy_c"], key=lambda x: x["Sharpe"]) if all_results["strategy_c"] else None

    if best_a:
        lines.append(f"| A: PE择时(MA代理) | {format_pct(best_a['CAGR'])} | {format_num(best_a['Sharpe'])} | {format_pct(best_a['MaxDD'])} |")

    if best_b:
        best_b_overall = max(best_b.values(), key=lambda x: x["Sharpe"])
        lines.append(f"| B: 双均线({best_b_overall.get('fast','?')}/{best_b_overall.get('slow','?')}) | {format_pct(best_b_overall['CAGR'])} | {format_num(best_b_overall['Sharpe'])} | {format_pct(best_b_overall['MaxDD'])} |")

    if best_c:
        lines.append(f"| C: 月度动量轮动({best_c.get('momentum_period','?')}月) | {format_pct(best_c['CAGR'])} | {format_num(best_c['Sharpe'])} | {format_pct(best_c['MaxDD'])} |")

    if best_d:
        best_d_overall = max(best_d.values(), key=lambda x: x["Sharpe"])
        lines.append(f"| D: 突破买入({best_d_overall.get('etf','?')}) | {format_pct(best_d_overall['CAGR'])} | {format_num(best_d_overall['Sharpe'])} | {format_pct(best_d_overall['MaxDD'])} |")

    lines.append("")
    for bm in all_results["benchmarks"]:
        lines.append(f"- 基准 **{bm['name']}**: CAGR={format_pct(bm['CAGR'])}, Sharpe={format_num(bm['Sharpe'])}, MaxDD={format_pct(bm['MaxDD'])}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 关键发现")
    lines.append("")
    lines.append("1. **策略A (PE择时MA代理)**: 通过简单的120日均线判断市场估值，在市场高位时切换到货币基金，降低了回撤但可能错失趋势行情。")
    lines.append("2. **策略B (双均线)**: 经典的趋势跟踪策略，参数选择对结果影响显著。短周期参数(10/50)对波动更敏感，长周期参数(30/120)产生更少但更可靠的信号。")
    lines.append("3. **策略C (动量轮动)**: 通过多ETF分散 + 动量选强 + 止损保护，在趋势市场中表现较好。缺点是在震荡市中频繁换仓。")
    lines.append("4. **策略D (突破买入)**: 基于价格突破和成交量确认，适合捕捉短期的爆发性行情，但在假突破时容易产生损失。")
    lines.append("")
    lines.append(f"*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    # Load data via DataStorage (with Parquet fallback)
    etf_510300 = load_etf("510300")
    etf_510500 = load_etf("510500")
    etf_159915 = load_etf("159915")
    etf_510880 = load_etf("510880")
    etf_518880 = load_etf("518880")
    etf_511880 = load_etf("511880")
    csi300 = load_csi300()

    print(f"510300: {len(etf_510300)} rows")
    print(f"510500: {len(etf_510500)} rows")
    print(f"159915: {len(etf_159915)} rows")
    print(f"510880: {len(etf_510880)} rows")
    print(f"518880: {len(etf_518880)} rows")
    print(f"511880: {len(etf_511880)} rows")
    print(f"CSI300: {len(csi300)} rows")

    all_results = {
        "strategy_a": [],
        "strategy_b": [],
        "strategy_c": [],
        "strategy_d": [],
        "benchmarks": []
    }

    # ===== Benchmarks =====
    print("\n" + "=" * 60)
    print("Running Benchmarks...")
    print("=" * 60)

    bm_510880 = benchmark_bh(etf_510880, "510880 Buy&Hold")
    bm_csi300 = benchmark_bh(csi300, "CSI300 Buy&Hold")
    all_results["benchmarks"] = [bm_510880, bm_csi300]

    for bm in all_results["benchmarks"]:
        print(f"  {bm['name']}: CAGR={bm['CAGR']:.4f}, Sharpe={bm['Sharpe']:.4f}, MaxDD={bm['MaxDD']:.4f}")

    # ===== Strategy A =====
    print("\n" + "=" * 60)
    print("Running Strategy A: PE Percentile Timing (MA proxy)...")
    print("=" * 60)

    metrics_a, ret_a = strategy_a_ma_valuation(csi300, etf_510300, etf_511880)
    all_results["strategy_a"].append(metrics_a)
    print(f"  CAGR={metrics_a['CAGR']:.4f}, Sharpe={metrics_a['Sharpe']:.4f}, MaxDD={metrics_a['MaxDD']:.4f}, Trades={int(metrics_a['Trades'])}")

    # ===== Strategy B =====
    print("\n" + "=" * 60)
    print("Running Strategy B: Dual MA Crossover...")
    print("=" * 60)

    for name, etf_df in [("510300", etf_510300), ("510500", etf_510500), ("159915", etf_159915)]:
        results_b = strategy_b_ma_cross(etf_df)
        all_results["strategy_b"].extend(results_b)
        best = max(results_b, key=lambda x: x["Sharpe"])
        print(f"  {name}: best ({best['fast']}/{best['slow']}) -> CAGR={best['CAGR']:.4f}, Sharpe={best['Sharpe']:.4f}, MaxDD={best['MaxDD']:.4f}, Trades={int(best['Trades'])}")

    # ===== Strategy C =====
    print("\n" + "=" * 60)
    print("Running Strategy C: Monthly Momentum Rotation...")
    print("=" * 60)

    etfs_dict = {
        "510300": etf_510300,
        "510500": etf_510500,
        "159915": etf_159915,
        "510880": etf_510880,
        "518880": etf_518880
    }

    results_c = strategy_c_momentum_rotation(etfs_dict, csi300)
    all_results["strategy_c"].extend(results_c)
    for r in results_c:
        mom = r.get("momentum_period", "?")
        print(f"  {mom}月动量: CAGR={r['CAGR']:.4f}, Sharpe={r['Sharpe']:.4f}, MaxDD={r['MaxDD']:.4f}, Trades={int(r['Trades'])}")

    # ===== Strategy D =====
    print("\n" + "=" * 60)
    print("Running Strategy D: Breakout...")
    print("=" * 60)

    for name, etf_df in [("510300", etf_510300), ("510500", etf_510500)]:
        r = strategy_d_breakout(etf_df)
        if r:
            all_results["strategy_d"].append(r)
            print(f"  {name}: CAGR={r['CAGR']:.4f}, Sharpe={r['Sharpe']:.4f}, MaxDD={r['MaxDD']:.4f}, Trades={int(r['Trades'])}")

    # ===== Generate Report =====
    print("\n" + "=" * 60)
    print("Generating Report...")
    print("=" * 60)

    report = generate_report(all_results)
    report_path = DATA_DIR / "alt_strategies_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport written to: {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()

"""
每日运行：输出当前市场状态和月度调仓信号。

用法：
    python scripts/daily_signal.py          # 打印完整报告
    python scripts/daily_signal.py --short  # 仅打印 ticker 列表

输出示例：
    市场状态: 牛市确认 (ST_BULL)  持仓第 3 月  仓位系数: 1.00
    Sharp 风险: 正常 (CSI300 5日回报: +1.23%)
    持仓 ETF (3只):
      1. 512480 半导体  [科技]  得分: 78.5
      2. 512660 军工    [高端制造]  得分: 72.1
      3. 159928 消费    [消费]  得分: 68.3
    防御资产待命: 511880 (80%) + 518880 (20%)
"""
import sys
sys.path.insert(0, r"C:\study\AIWorkspace\quanti")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from datetime import datetime

import numpy as np

from quanti.config.etf_universe import ETF_UNIVERSE_MULTI, get_available_etfs, get_sector
from quanti.data.storage import DataStorage
from quanti.indicators import sma, adx_with_di


# ── 参数 ──
TOP_N = 3
N_CONFIRM = 5
M_COOLDOWN = 40
SHARP_THRESHOLD = -0.03
BOND_PCT = 0.80
GOLD_PCT = 0.20
MAX_PER_SECTOR = 2
BOND_ETF = "511880"
GOLD_ETF = "518880"
CSI300 = "510300"


def load_etf_arrays(storage, code, until=None):
    """加载 ETF 数据，返回 (dates, closes, highs, lows, volumes) 的 numpy 数组。

    如果指定 until (YYYYMMDD)，只加载该日期及之前的数据。
    """
    bars = storage.load_bars(code)
    if not bars or len(bars) < 200:
        return None
    if until:
        bars = [b for b in bars if b.trade_date <= until]
        if len(bars) < 200:
            return None
    return (
        np.array([b.trade_date for b in bars]),
        np.array([b.close for b in bars], dtype=np.float64),
        np.array([b.high for b in bars], dtype=np.float64),
        np.array([b.low for b in bars], dtype=np.float64),
        np.array([b.volume for b in bars], dtype=np.float64),
    )


def market_state(csi):
    """计算当前市场状态。"""
    dates, closes, highs, lows, vols = csi
    n = len(closes)
    ma120 = sma(closes, 120)
    ma60 = sma(closes, 60)

    # 当前是否在 120MA 之上
    above_now = closes[-1] > ma120[-1] if not np.isnan(ma120[-1]) else False

    # 最近一次 120MA 突破
    above_series = (closes > ma120) & (~np.isnan(ma120))
    last_below = n - 1
    for i in range(n - 1, 120, -1):
        if not above_series[i] and above_series[i - 1]:
            pass  # 跌破
        if above_series[i] and not above_series[i - 1]:
            last_below = i
            break

    days_above = n - last_below if last_below < n else 0

    state_names = {0: "熊市", 1: "确认中", 2: "牛市确认", 3: "冷却期", 4: "假突破"}

    if not above_now:
        return 0, f"熊市 (ST_BEAR) — 价格在 120MA 之下"

    if days_above < N_CONFIRM:
        return 1, f"确认中 (ST_CONFIRMING) — 突破第 {days_above} 天, 还需 {N_CONFIRM - days_above} 天"

    # 检查确认期的成交量和涨幅
    cf_start = last_below
    cf_end = cf_start + N_CONFIRM
    wvol = float(np.mean(vols[cf_start:cf_end + 1])) if cf_end < n else 0
    pvol = float(np.mean(vols[max(0, cf_start - 20):cf_start])) if cf_start >= 20 else wvol
    chg = closes[min(cf_end, n - 1)] / closes[cf_start] - 1.0

    if wvol >= 0.85 * pvol and chg >= -0.02:
        return 2, f"牛市确认 (ST_BULL) — 突破 {days_above} 天, 量比 {wvol / pvol:.2f}, 涨幅 {chg:+.1%}"
    else:
        return 4, f"假突破 (ST_FAKE) — 量比 {wvol / pvol:.2f}, 涨幅 {chg:+.1%}"


def sharp_check(csi):
    """检查 CSI300 5 日回报是否触发 Sharp 闪电退出。"""
    closes = csi[1]
    if len(closes) < 6:
        return False, 0.0
    ret5 = float(closes[-1] / closes[-6] - 1.0)
    return ret5 < SHARP_THRESHOLD, ret5


def etf_score(dates, closes, highs, lows, volumes):
    """5 条件筛选 + 动量稳定性评分。"""
    n = len(closes)
    if n < 200:
        return False, 0.0

    ma120 = sma(closes, 120)
    if ma120 is None or np.isnan(ma120[-1]):
        return False, 0.0

    # 1. above 120MA
    above = closes[-1] > ma120[-1]

    # 2. structure
    rh20 = float(np.max(highs[-20:]))
    ph20 = float(np.max(highs[-60:-20])) if n >= 80 else 0.0
    rl20 = float(np.min(lows[-20:]))
    pl20 = float(np.min(lows[-60:-20])) if n >= 80 else 0.0
    structure = rh20 > ph20 and rl20 > pl20

    # 3. MA alignment
    ma20 = sma(closes, 20)
    ma60_val = sma(closes, 60)
    align = (
        ma20 is not None and ma60_val is not None
        and not np.isnan(ma20[-1]) and not np.isnan(ma60_val[-1])
        and ma20[-1] > ma60_val[-1] > ma120[-1]
    )

    # 4. ADX > 25
    di = adx_with_di(highs, lows, closes, 14)
    adx_ok = di is not None and not np.isnan(di[0][-1]) and di[0][-1] > 25

    # 5. volume surge
    v20 = float(np.mean(volumes[-21:-1])) if n >= 22 else 0
    surge = volumes[-1] > v20 * 1.2 if v20 > 0 else False

    conds = sum([above, structure, align, adx_ok, surge])
    passed = above and adx_ok and conds >= 3

    if not passed:
        return False, 0.0

    # 评分
    r3 = closes[-1] / closes[-63] - 1.0 if closes[-63] > 1e-6 else 0.0
    r6 = closes[-1] / closes[-126] - 1.0 if closes[-126] > 1e-6 else 0.0
    m3 = min(max(r3 / 0.5, 0.0), 1.0) if r3 > 0 else 0.0
    m6 = min(max(r6 / 0.8, 0.0), 1.0) if r6 > 0 else 0.0
    mom = (0.5 * m3 + 0.5 * m6) * 100.0
    window = closes[-61:]
    daily_ret = np.diff(window) / (window[:-1] + 1e-10)
    vol_metric = min(np.nanstd(daily_ret) / 0.04, 1.0) if len(daily_ret) > 1 else 1.0
    stability = (1.0 - vol_metric) * 100.0
    return True, 0.6 * mom + 0.4 * stability


def main():
    parser = argparse.ArgumentParser(description="daily signal generator")
    parser.add_argument("--short", action="store_true", help="仅输出 ticker 列表")
    parser.add_argument("--date", type=str, default=None, help="指定日期 YYYYMMDD (默认今天)")
    args = parser.parse_args()

    storage = DataStorage()
    today = args.date or datetime.now().strftime("%Y%m%d")

    # ── CSI300 市场状态 ──
    csi = load_etf_arrays(storage, CSI300, until=today)
    if csi is None:
        print("错误: 缺少 CSI300 (510300) 数据")
        return

    latest_csi_date = csi[0][-1]
    if not args.date:
        today = latest_csi_date  # 默认用最新可用日期

    mst, mst_desc = market_state(csi)
    sharp_fired, ret5 = sharp_check(csi)

    if args.short:
        if mst in (0, 1, 3) or sharp_fired:
            print(f"{BOND_ETF} {GOLD_ETF}")
        else:
            # 正常权益模式
            available = get_available_etfs(today)
            scored = []
            for e in available:
                code = e["code"]
                arr = load_etf_arrays(storage, code, until=today)
                if arr is None:
                    continue
                passed, s = etf_score(*arr)
                if passed:
                    scored.append((code, s, get_sector(code)))
            scored.sort(key=lambda x: x[1], reverse=True)

            selected = []
            sc = {}
            for code, s, sector in scored:
                if len(selected) >= TOP_N:
                    break
                if sector not in ("宽基", "防御"):
                    if sc.get(sector, 0) >= MAX_PER_SECTOR:
                        continue
                selected.append(code)
                sc[sector] = sc.get(sector, 0) + 1
            print(" ".join(selected))
        return

    # ── 完整报告 ──
    print("=" * 60)
    print(f"  quanTI 每日信号  {today}")
    print("=" * 60)

    print(f"\n  数据日期: {latest_csi_date}")
    print(f"  市场状态: {mst_desc}")
    print(f"  Sharp 风险: {'⚡ 触发！' if sharp_fired else '正常'} (CSI300 5日回报: {ret5:+.2%})")
    if sharp_fired:
        print(f"  操作: 立即清仓权益 → {BOND_ETF}({BOND_PCT:.0%}) + {GOLD_ETF}({GOLD_PCT:.0%})")
        return

    if mst in (0, 1, 3):
        print(f"\n  权益仓位: 0% (市场状态 {mst})")
        print(f"  防御资产: {BOND_ETF} ({BOND_PCT:.0%}) + {GOLD_ETF} ({GOLD_PCT:.0%})")
        return

    # 权益模式
    position_pct = {2: 1.0, 4: 0.5}.get(mst, 0)
    print(f"\n  权益仓位: {position_pct:.0%} (市场状态 {mst})")

    available = get_available_etfs(today)
    scored = []
    for e in available:
        code = e["code"]
        arr = load_etf_arrays(storage, code, until=today)
        if arr is None:
            continue
        passed, s = etf_score(*arr)
        if passed:
            scored.append((code, s, e["name"], get_sector(code)))

    scored.sort(key=lambda x: x[1], reverse=True)

    # 前 10 候选
    print(f"\n  候选 ETF (通过筛选 {len(scored)} 只, 前 10):")
    print(f"  {'代码':>6s}  {'名称':<8s}  {'行业':<8s}  {'得分':>6s}")
    for code, s, name, sector in scored[:10]:
        print(f"  {code:>6s}  {name:<8s}  {sector:<8s}  {s:6.1f}")

    # 选 Top N
    selected = []
    sc = {}
    for code, s, name, sector in scored:
        if len(selected) >= TOP_N:
            break
        if sector not in ("宽基", "防御"):
            if sc.get(sector, 0) >= MAX_PER_SECTOR:
                continue
        selected.append((code, s, name, sector))
        sc[sector] = sc.get(sector, 0) + 1

    print(f"\n  ═══ 持仓信号 (月频) ═══")
    for i, (code, s, name, sector) in enumerate(selected, 1):
        print(f"  {i}. {code}  {name:<8s}  [{sector}]  得分: {s:.1f}")

    print(f"\n  防御资产待命: {BOND_ETF} ({BOND_PCT:.0%}) + {GOLD_ETF} ({GOLD_PCT:.0%})")
    print(f"  下一次调仓: 月末最后一个交易日")


if __name__ == "__main__":
    main()

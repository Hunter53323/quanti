"""
Full trade trace: Every stop/liquidation/rotate with entry/exit/HWM/PnL detail.
S1 with 3 stocks + 1 ETF MR, full daily walkthrough.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

D = Path("C:/study/AIWorkspace/quanti/data/clean")
STOP_PCT = -0.10
MA_P = 120

def sma(s, p):
    return s.rolling(p).mean()

def sc(st, c, d):
    if c not in st:
        return None
    s = st[c][st[c].index <= d]
    return None if len(s) == 0 else float(s["close"].iloc[-1])

def ec(et, c, d):
    if c not in et:
        return None
    s = et[c][et[c].index <= d]
    return None if len(s) == 0 else float(s["close"].iloc[-1])


# Load data
stocks = {}
for c in ["600157", "601288", "600010", "000725", "300059"]:
    fp = D / f"{c}.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        stocks[c] = df.sort_values("date").set_index("date")

et = {}
for c in ["510300"]:
    fp = D / f"{c}.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        et[c] = df.sort_values("date").set_index("date")

csi = pd.read_parquet(D / "CSI300.parquet")
csi["date"] = pd.to_datetime(csi["trade_date"], format="%Y%m%d")
csi = csi.sort_values("date").set_index("date")


def compute_rsi(cl, period=14):
    d = cl.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def trail_stop(pos, cash_ref, stt, ett, c, d, pct, trades, tag):
    pr = sc(stt, c, d) or ec(ett, c, d)
    if pr is None:
        return False
    p = pos[c]
    hwm = p.get("hwm", pr)
    old_hwm = hwm
    if pr > hwm:
        hwm = pr
        p["hwm"] = hwm
    ratio = pr / hwm - 1.0
    if ratio < pct:
        pnl_val = (pr - p["avg"]) * p["qty"]
        cash_ref[0] += pr * p["qty"]
        trades.append({
            "date": d, "symbol": c, "side": "sell",
            "qty": p["qty"], "price": pr, "pnl": pnl_val,
            "hwm": old_hwm, "entry_avg": p["avg"],
            "tag": tag
        })
        del pos[c]
        return True, old_hwm, hwm, pr, ratio
    return False, old_hwm, hwm, pr, ratio


def select_top(dt, top_n=3):
    trending = []
    for code, df in stocks.items():
        sub = df[df.index <= dt]
        if len(sub) < 200:
            continue
        cl = sub["close"]
        m120 = sma(cl, 120).iloc[-1]
        if pd.isna(m120) or cl.iloc[-1] <= m120:
            continue
        if cl.iloc[-63] > 1e-6:
            score = (cl.iloc[-1] / cl.iloc[-63] - 1) * 100
        else:
            score = 0
        if score > 0:
            trending.append((code, score))
    trending.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in trending[:top_n]]


# Generate month-ends
mes = []
for y in [2022, 2023, 2024, 2025]:
    for m in range(1, 13):
        me = pd.Timestamp(f"{y}-{m:02d}-01") + pd.offsets.MonthEnd(0)
        d = me
        while d not in csi.index and d > me - pd.Timedelta(days=8):
            d -= pd.Timedelta(days=1)
        if d in csi.index:
            mes.append(d)

# Run S1 with detailed trade tracking
tc = 50000
mc = 40000
tp = {}
mp = {}
tpr = 50000
mpr = 40000
trades = []
dr = []

for i, d in enumerate(mes):
    if i == 0:
        continue

    sd = csi[csi.index <= d]
    bull = (
        len(sd) >= MA_P
        and not pd.isna(sma(sd["close"], MA_P).iloc[-1])
        and float(sd["close"].iloc[-1]) > float(sma(sd["close"], MA_P).iloc[-1])
    )

    # Trend: trailing stops
    tcr = [tc]
    for c in list(tp):
        trail_stop(tp, tcr, stocks, et, c, d, STOP_PCT, trades, "s1_t_stop")
    tc = tcr[0]

    # Trend: rebalance
    if bull:
        sel = select_top(d)
        for c in list(tp):
            if c not in sel:
                pr = sc(stocks, c, d)
                if pr:
                    pnl = (pr - tp[c]["avg"]) * tp[c]["qty"]
                    tc += pr * tp[c]["qty"]
                    trades.append({
                        "date": d, "symbol": c, "side": "sell",
                        "qty": tp[c]["qty"], "price": pr, "pnl": pnl,
                        "hwm": tp[c].get("hwm", 0), "entry_avg": tp[c]["avg"],
                        "tag": "s1_t_rotate"
                    })
                    del tp[c]
        nw = [s for s in sel if s not in tp]
        if nw and tc > 0:
            per = tc / max(len(sel), 1) * 0.92
            for c in nw:
                pr = sc(stocks, c, d)
                if pr and pr > 0.01 and per >= pr * 100:
                    q = int(per / pr / 100) * 100
                    if q >= 100 and q * pr <= tc:
                        tc -= q * pr
                        tp[c] = {"qty": q, "avg": pr, "hwm": pr}
                        trades.append({
                            "date": d, "symbol": c, "side": "buy",
                            "qty": q, "price": pr, "pnl": 0,
                            "hwm": pr, "entry_avg": pr, "tag": "s1_t_entry"
                        })
    else:
        for c in list(tp):
            pr = sc(stocks, c, d)
            if pr:
                pnl = (pr - tp[c]["avg"]) * tp[c]["qty"]
                tc += pr * tp[c]["qty"]
                trades.append({
                    "date": d, "symbol": c, "side": "sell",
                    "qty": tp[c]["qty"], "price": pr, "pnl": pnl,
                    "hwm": tp[c].get("hwm", 0), "entry_avg": tp[c]["avg"],
                    "tag": "s1_t_liq"
                })
                del tp[c]

    te = tc + sum(
        sc(stocks, c, d) * p["qty"]
        for c, p in tp.items()
        if sc(stocks, c, d)
    )

    # MR
    if not bull:
        for code in ["510300"]:
            sub = et[code][et[code].index <= d]
            if len(sub) < 30:
                continue
            cl = sub["close"]
            rsi_v = float(compute_rsi(cl, 14).iloc[-1])
            if pd.isna(rsi_v):
                continue

            if code in mp:
                pr = float(cl.iloc[-1])
                hwm = mp[code].get("hwm", pr)
                old_hwm = hwm
                if pr > hwm:
                    hwm = pr
                    mp[code]["hwm"] = hwm

                exited = False
                if rsi_v > 70:
                    pnl = (pr - mp[code]["avg"]) * mp[code]["qty"]
                    mc += pr * mp[code]["qty"]
                    trades.append({
                        "date": d, "symbol": code, "side": "sell",
                        "qty": mp[code]["qty"], "price": pr, "pnl": pnl,
                        "hwm": old_hwm, "entry_avg": mp[code]["avg"],
                        "tag": "s1_m_rsi70"
                    })
                    del mp[code]
                    exited = True
                elif (pr / hwm - 1.0) < STOP_PCT:
                    pnl = (pr - mp[code]["avg"]) * mp[code]["qty"]
                    mc += pr * mp[code]["qty"]
                    trades.append({
                        "date": d, "symbol": code, "side": "sell",
                        "qty": mp[code]["qty"], "price": pr, "pnl": pnl,
                        "hwm": old_hwm, "entry_avg": mp[code]["avg"],
                        "tag": "s1_m_trailstop"
                    })
                    del mp[code]
                    exited = True
                if exited:
                    continue

            elif rsi_v < 30 and mc > 0:
                pr = float(cl.iloc[-1])
                inv = mc * 0.5
                if inv >= pr * 100:
                    q = int(inv / pr / 100) * 100
                    if q >= 100 and q * pr <= mc:
                        mc -= q * pr
                        mp[code] = {"qty": q, "avg": pr, "hwm": pr}
                        trades.append({
                            "date": d, "symbol": code, "side": "buy",
                            "qty": q, "price": pr, "pnl": 0,
                            "hwm": pr, "entry_avg": pr, "tag": "s1_m_entry"
                        })
    else:
        for c in list(mp):
            pr = ec(et, c, d)
            if pr:
                pnl = (pr - mp[c]["avg"]) * mp[c]["qty"]
                mc += pr * mp[c]["qty"]
                trades.append({
                    "date": d, "symbol": c, "side": "sell",
                    "qty": mp[c]["qty"], "price": pr, "pnl": pnl,
                    "hwm": mp[c].get("hwm", 0), "entry_avg": mp[c]["avg"],
                    "tag": "s1_m_liq"
                })
                del mp[c]

    meq = mc + sum(
        ec(et, c, d) * p["qty"]
        for c, p in mp.items()
        if ec(et, c, d)
    )
    tn = te + meq
    tpv = tpr + mpr
    if tpv > 0:
        dr.append((tn - tpv) / tpv)
    tpr = te
    mpr = meq


# Print ALL trades
print("=" * 130)
print("COMPLETE TRADE LOG: Every SELL with entry, HWM, exit price, P&L")
print("=" * 130)
header = f"{'Date':<12s} {'Symbol':<8s} {'Tag':<20s} {'Qty':>6s} {'Entry':>8s} {'HWM':>8s} {'Exit':>8s} {'PnL':>10s} {'Ret%':>7s} {'Win':>4s}"
print(header)
print("-" * 130)

sells = [t for t in trades if t["side"] == "sell"]
total_pnl = 0
win_count = 0
loss_count = 0
stop_count = 0

for t in sells:
    entry = t.get("entry_avg", 0)
    exit_p = t["price"]
    hwm_v = t.get("hwm", exit_p)

    if t["qty"] > 0 and entry > 0:
        ret_pct = (exit_p / entry - 1) * 100
        hwm_ret = (exit_p / max(hwm_v, 0.01) - 1) * 100
    else:
        ret_pct = 0
        hwm_ret = 0

    is_win = t["pnl"] > 0
    if is_win:
        win_count += 1
    else:
        loss_count += 1
    total_pnl += t["pnl"]

    tag = t.get("tag", "")
    if "stop" in tag.lower():
        stop_count += 1

    win_str = "WIN" if is_win else "LOSS"
    print(
        f'{t["date"].strftime("%Y-%m-%d"):<12s} {t["symbol"]:<8s} {tag:<20s} '
        f'{t["qty"]:>6d} {entry:>8.2f} {hwm_v:>8.2f} {exit_p:>8.2f} '
        f'{t["pnl"]:>10.0f} {ret_pct:>6.1f}% {win_str:>4s}'
    )

print("-" * 130)
print(f"SELLS: {len(sells)}  |  WINS: {win_count} ({win_count/max(len(sells),1)*100:.0f}%)  |  LOSSES: {loss_count}")
print(f"STOP-specific: {stop_count}  |  TOTAL PnL: {total_pnl:.0f} RMB")

# BUYS
buys = [t for t in trades if t["side"] == "buy"]
print(f"\nBUYS: {len(buys)} entries - see trade log above for position openings")

# Strategy metrics
cum_ret = np.cumprod(1 + np.array(dr))
total_ret = cum_ret[-1] - 1
years = len(dr) / 252
cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1
vol = np.std(dr, ddof=1) * np.sqrt(252)
sharpe = (cagr - 0.03) / max(vol, 1e-10)
peak = np.maximum.accumulate(cum_ret)
maxdd = np.min((cum_ret - peak) / peak)
print(f"\nSTRATEGY: CAGR={cagr*100:.2f}%  Sharpe={sharpe:.3f}  MaxDD={maxdd*100:.2f}%")

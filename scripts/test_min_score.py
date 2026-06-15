"""Test min_score threshold impact on backtest with corrected counters.

Uses omc_utils for data loading and rebalance-date generation.
Keeps its own run() loop for the invested/dded/skipped counters that
run_backtest() does not expose.
"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".omc"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from quanti.strategy.etf_rotation import ETFRotationStrategy
from omc_utils import ETFData, monthly_rebal_dates

CAP = 90000
COMM = 0.00025
POOL = ["510300", "510500", "159915", "588360", "563300", "510880", "518880", "511880"]

data = ETFData.load(POOL)
all_rebal = monthly_rebal_dates(data.dates, from_date="20150101", to_date="20251231")


def run(start, end, top_n=3, min_score=0.0,
        w_trend=0.35, w_adx=0.40, w_momentum=0.25, w_macd=0.0, w_kdj=0.0):
    """Run backtest with per-rebalance filtering and monthly counters."""
    rebal = [d for d in all_rebal if start <= d <= end]
    cash = CAP; hld = {}; eq = [CAP]; pk = CAP
    invested = 0; dded = 0; skipped = 0

    for rd in rebal:
        total = cash + sum(
            h["q"] * data.price(s, rd)
            for s, h in hld.items() if data.price(s, rd)
        )
        if total > pk:
            pk = total
        dd = (pk - total) / pk * 100 if pk > 0 else 0

        if dd > 15:
            for sym in list(hld.keys()):
                p = data.price(sym, rd) or hld[sym]["p"]
                cash += hld[sym]["q"] * p * (1 - COMM); del hld[sym]
            eq.append(cash); dded += 1; continue

        scored = []
        for sym in POOL:
            if not data.eligible(sym, rd):
                continue
            arrs = data.slice(sym, rd)
            if arrs is None:
                continue
            c2, h2, l2 = arrs
            mn = np.mean(c2[-120:]); mg = np.mean(c2[-140:-20])
            if not mn > mg:
                continue
            r = ETFRotationStrategy.compute_scores(
                c2, h2, l2,
                w_trend=w_trend, w_adx=w_adx, w_momentum=w_momentum,
                w_macd=w_macd, w_kdj=w_kdj,
                ma_period=120,
            )
            if r["composite"] > 0:
                scored.append((sym, r["composite"]))

        passing = [(s, sc) for s, sc in scored if sc >= min_score]
        if len(passing) > 0:
            invested += 1
        if not passing:
            for sym in list(hld.keys()):
                p = data.price(sym, rd) or hld[sym]["p"]
                cash += hld[sym]["q"] * p * (1 - COMM); del hld[sym]
            eq.append(cash); skipped += 1; continue

        passing.sort(key=lambda x: x[1], reverse=True)
        sel = {s[0] for s in passing[:top_n]}

        for sym in list(hld.keys()):
            if sym not in sel:
                p = data.price(sym, rd) or hld[sym]["p"]
                cash += hld[sym]["q"] * p * (1 - COMM); del hld[sym]
        total2 = cash + sum(
            h["q"] * data.price(s, rd)
            for s, h in hld.items() if data.price(s, rd)
        )
        n = max(len(sel), 1); per = total2 / n * 0.90
        for sym in sel:
            p = data.price(sym, rd)
            if p is None or p < 0.01:
                continue
            tq = int(per / p / 100) * 100
            if tq < 100:
                continue
            cst = tq * p
            if sym in hld:
                diff = tq - hld[sym]["q"]
                if abs(diff) >= 100:
                    cst2 = abs(diff) * p
                    if diff > 0 and cash >= cst2 * (1 + COMM):
                        cash -= cst2 * (1 + COMM); hld[sym]["q"] = tq
                    elif diff < 0:
                        cash += cst2 * (1 - COMM); hld[sym]["q"] = tq
            elif cash >= cst * (1 + COMM):
                cash -= cst * (1 + COMM); hld[sym] = {"q": tq, "p": p}
        eq.append(cash + sum(
            h["q"] * data.price(s, rd)
            for s, h in hld.items() if data.price(s, rd)
        ))

    eq = np.array(eq)
    ny = (int(end[:4]) - int(start[:4])) + (int(end[4:6]) - int(start[4:6])) / 12.0
    cagr = ((eq[-1] / eq[0]) ** (1 / ny) - 1) * 100 if ny > 0 and eq[0] > 0 else 0
    mr = np.diff(eq) / (eq[:-1] + 1e-10)
    sh = (np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12)) if len(mr) > 1 else 0
    pk2 = eq[0]; md_val = 0.0
    for v in eq:
        if v > pk2: pk2 = v
        dv = (pk2 - v) / pk2 * 100
        if dv > md_val: md_val = dv
    return cagr, sh, md_val, invested, dded, skipped


THRESHOLDS = [0, 0.25, 0.30, 0.35, 0.40]
PERIODS = [
    ("Train(15-21)", "20150101", "20211231"),
    ("Test(22-25)",  "20220101", "20251231"),
    ("2022",         "20220101", "20221231"),
    ("2023",         "20230101", "20231231"),
    ("2024",         "20240101", "20241231"),
    ("2025",         "20250101", "20251231"),
]

print(f"MinS | Period           | CAGR    | Sharpe  | MaxDD   | Invested     | DD | Skip")
print("-" * 88)
for ms in THRESHOLDS:
    for pn, ps, pe in PERIODS:
        cagr, sh, md, inv, dded, skip = run(ps, pe, min_score=ms)
        print(f" {ms:.2f} | {pn:<16s} | {cagr:+6.1f}% | {sh:+7.3f} | {md:+6.1f}% | {inv:>3d}月(+{dded}DD+{skip}跳)")

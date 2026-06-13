"""Attribute returns: stock selection vs market timing, on canonical test period."""
import os, sys, time
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025
TEST_START, TEST_END = "20220101", "20251231"

t0 = time.time()
storage = DataStorage()
raw = storage.load_bars("510300")
csi_d = np.array([r.trade_date for r in raw])
csi_c = np.array([r.close for r in raw], dtype=np.float64)
csi_v = np.array([r.volume for r in raw], dtype=np.float64)

# Import canonical functions
import importlib.util
spec = importlib.util.spec_from_file_location("rt", os.path.join(_PROJECT_ROOT, "run_backtest.py"))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

all_f = sorted(storage.clean_dir.glob("*.parquet"))
codes = [p.stem for p in all_f if len(p.stem) == 6 and not p.stem.startswith(("51", "58", "15", "56"))]
sd = {}; ads = set()
for code in codes:
    rs = storage.load_bars(code)
    if not rs or len(rs) < 200: continue
    d = [r.trade_date for r in rs]
    sd[code] = (np.array([r.close for r in rs], dtype=np.float64),
                np.array([r.high for r in rs], dtype=np.float64),
                np.array([r.low for r in rs], dtype=np.float64),
                np.array([r.volume for r in rs], dtype=np.float64), d)
    ads.update(d)
ad = sorted(ads)

print("Precomputing...")
def precompute(start, end):
    rd = rt.monthly_dates(ad, start, end); cache = {}
    for i, d in enumerate(rd):
        if i % 12 == 0: print(f"  {i}/{len(rd)}")
        t = []
        for code in sd:
            d2 = rt.data_at(code, d, 260, sd)
            if d2 is None: continue
            cl, hi, lo, vo = d2
            is_t, nc = rt.is_stock_uptrend(cl, hi, lo, vo)
            if is_t and nc >= 3: t.append((code, rt.trend_score(cl)))
        t.sort(key=lambda x: x[1], reverse=True)
        cache[d] = t
    return cache

tc = precompute(TEST_START, TEST_END)
sm = rt.build_state_map(csi_d, csi_c, csi_v, 5, 40, 0.85, -0.02)

# Run always-invested (stock selection only, no market timing)
def run_always(tc, sdata, adates, start, end):
    rebal = rt.monthly_dates(adates, start, end)
    cash = CAPITAL; holdings = {}; eq = [cash]; max_eq = cash; dd_active = False
    for rd in rebal:
        trending = tc.get(rd, [])
        if not trending:
            eq.append(cash + sum(h["qty"] * h["price"] for h in holdings.values()))
            continue
        selected = {t[0] for t in trending[:5]}
        for sym in list(holdings.keys()):
            p = rt.price_on(sym, rd, sdata)
            if p is None or p < 0.01:
                cash += holdings[sym]["val"] * 0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p; holdings[sym]["val"] = holdings[sym]["qty"] * p
            if "hwm" not in holdings[sym] or p > holdings[sym]["hwm"]:
                holdings[sym]["hwm"] = p
            if holdings[sym]["hwm"] > 0 and (p / holdings[sym]["hwm"] - 1) * 100 < -10:
                cash += holdings[sym]["val"] * (1 - COMM); del holdings[sym]
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())
        if total > max_eq: max_eq = total
        dd = (max_eq - total) / max_eq * 100 if max_eq > 0 else 0
        if dd > 15:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1 - COMM)
                del holdings[sym]
            dd_active = True
        elif dd_active and total / max_eq > 0.92: dd_active = False
        if dd_active: eq.append(cash); continue
        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["qty"] * holdings[sym]["price"] * (1 - COMM)
                del holdings[sym]
        n_pos = max(len(selected), 1); per = total / n_pos * 0.90
        for sym in selected:
            p = rt.price_on(sym, rd, sdata)
            if p is None or p < 0.01: continue
            tq = int(per / p / 100) * 100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff) * p
                    if diff > 0 and cash >= cost * (1 + COMM):
                        cash -= cost * (1 + COMM); holdings[sym]["qty"] = tq
                    elif diff < 0:
                        cash += cost * (1 - COMM); holdings[sym]["qty"] = tq
            else:
                cost = tq * p
                if cash >= cost * (1 + COMM):
                    cash -= cost * (1 + COMM)
                    holdings[sym] = {"qty": tq, "price": p, "val": cost, "hwm": p}
        total = cash + sum(h["qty"] * h["price"] for h in holdings.values())
        eq.append(total)
    return rt.metrics(eq)

r_always = run_always(tc, sd, ad, TEST_START, TEST_END)
r_br = rt.run_bond_rotate(sm, tc, sd, ad, {}, TEST_START, TEST_END)
a43_fn = lambda m: 1.0 if m <= 4 else (0.75 if m <= 8 else 0.50)
r_a43 = rt.run_bond_rotate_decay(sm, tc, sd, ad, {}, TEST_START, TEST_END, a43_fn)

# CSI300 B&H on same period
csi_start_idx = None; csi_end_idx = None
for i, d in enumerate(raw):
    if d.trade_date >= "20220101" and csi_start_idx is None: csi_start_idx = i
    if d.trade_date <= "20251231": csi_end_idx = i
csi_ret = (raw[csi_end_idx].close / raw[csi_start_idx].close - 1) * 100
csi_ny = (2025 - 2022) + 1
csi_cagr = ((raw[csi_end_idx].close / raw[csi_start_idx].close) ** (1 / csi_ny) - 1) * 100

print(f"\n{'='*70}")
print("ATTRIBUTION ON CANONICAL TEST PERIOD (2022-2025)")
print(f"{'='*70}")
print(f"  CSI300 B&H:                    C={csi_cagr:+.2f}%")
print(f"  Always-invested stocks:        C={r_always['cagr']:+.2f}% S={r_always['sharpe']:.3f} D={r_always['maxdd']:.1f}%")
print(f"  BOND_ROTATE entry-only:        C={r_br['cagr']:+.2f}% S={r_br['sharpe']:.3f} D={r_br['maxdd']:.1f}%")
print(f"  BOND_ROTATE + A43:              C={r_a43['cagr']:+.2f}% S={r_a43['sharpe']:.3f} D={r_a43['maxdd']:.1f}%")
print(f"")
print(f"  Stock selection alpha:         {r_always['cagr'] - csi_cagr:+.1f}%  (always-invested minus CSI300 B&H)")
print(f"  Market timing (entry only):    {r_br['cagr'] - r_always['cagr']:+.1f}%  (BOND_ROTATE minus always-invested)")
print(f"  Market timing + A43:           {r_a43['cagr'] - r_always['cagr']:+.1f}%  (A43 minus always-invested)")
print(f"  Net (A43 vs CSI300):           {r_a43['cagr'] - csi_cagr:+.1f}%")
print(f"{'='*70}")

print(f"\nDone in {time.time()-t0:.0f}s")

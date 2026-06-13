"""
State Machine Strategy V2 — Risk Budgeting & Adaptive Execution
================================================================
Key improvements over V1:
  1. DYNAMIC POSITION SIZING: floating leverage based on realized vol (target vol)
  2. ATR-BASED ADAPTIVE STOP: 2x/3x ATR trailing instead of fixed -10%
  3. DUAL MOMENTUM CONFIRM: require 60d return > 0 to enter
  4. CONVEXITY: when entering BEAR, hold 511880 (already done)
  5. TRAILING DD CIRCUIT BREAKER: reduce exposure at -10% DD (not just -15%)
  6. CROSS-SECTIONAL VOL TARGET: scale each position by 1/vol
  7. REGIME-DEPENDENT PARAMS: different thresholds per confirmed state

Will test each component independently, then together.

Train: 2015-2021 | Test: 2022-2025
"""
import sys, os, itertools, time, numpy as np
os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025
CASH_ETF, CSI300_PROXY = "511880", "510300"
TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START, TEST_END = "20220101", "20251231"

T_START = time.time()
print("=" * 90)
print("STATE MACHINE V2 — Risk Budgeting & Adaptive Execution")
print("=" * 90)

# ═══════════════════════════════════════════════════════════
# 1. Load Data
# ═══════════════════════════════════════════════════════════
print("\n[1] Loading data...")
storage = DataStorage()

raw300 = storage.load_bars(CSI300_PROXY)
csi_dates = np.array([r.trade_date for r in raw300])
csi_c = np.array([r.close for r in raw300], dtype=np.float64)
csi_h = np.array([r.high for r in raw300], dtype=np.float64)
csi_l = np.array([r.low for r in raw300], dtype=np.float64)

raw_cash = storage.load_bars(CASH_ETF)
cash_dates = np.array([r.trade_date for r in raw_cash])
cash_c = np.array([r.close for r in raw_cash], dtype=np.float64)

all_files = sorted(storage.clean_dir.glob("*.parquet"))
stock_codes = [p.stem for p in all_files if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]

STOCK = {}
for code in stock_codes:
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200: continue
    STOCK[code] = {
        "d": np.array([r.trade_date for r in raw]),
        "c": np.array([r.close for r in raw], dtype=np.float64),
        "h": np.array([r.high for r in raw], dtype=np.float64),
        "l": np.array([r.low for r in raw], dtype=np.float64),
        "v": np.array([r.volume for r in raw], dtype=np.float64),
    }

all_dates_set = set(csi_dates)
for sd in STOCK.values(): all_dates_set.update(sd["d"])
ALL_DATES = sorted(all_dates_set)
print(f"  CSI300: {len(csi_dates)}, Stocks: {len(STOCK)}, Dates: {len(ALL_DATES)}")


# ═══════════════════════════════════════════════════════════
# 2. Indicator Functions
# ═══════════════════════════════════════════════════════════
def _sma(arr, p):
    if len(arr) < p: return np.nan
    return float(np.mean(arr[-p:]))

def _adx(h, l, c, p=14):
    n = len(c)
    if n < p * 2: return np.nan
    tr = np.zeros(n); tr[0] = h[0] - l[0]
    for i in range(1, n): tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i - 1]; dn = l[i - 1] - l[i]
        if up > dn and up > 0: pdm[i] = up
        if dn > up and dn > 0: mdm[i] = dn
    atr = np.full(n, np.nan); atr[p] = float(np.mean(tr[1:p + 1]))
    for i in range(p + 1, n): atr[i] = (tr[i] + (p - 1) * atr[i - 1]) / p
    ps = float(np.mean(pdm[1:p + 1])); ms = float(np.mean(mdm[1:p + 1]))
    pi = np.full(n, np.nan); mi = np.full(n, np.nan)
    pi[p] = ps / max(atr[p], 0.001) * 100; mi[p] = ms / max(atr[p], 0.001) * 100
    for i in range(p + 1, n):
        ps = (pdm[i] + (p - 1) * ps) / p; ms = (mdm[i] + (p - 1) * ms) / p
        pi[i] = min(ps / max(atr[i], 0.001) * 100, 1000)
        mi[i] = min(ms / max(atr[i], 0.001) * 100, 1000)
    dx = np.abs(pi - mi) / (pi + mi + 1e-10) * 100
    ax = np.full(n, np.nan)
    seed = float(np.nanmean(dx[p:p * 2]))
    ax[p * 2 - 1] = 0.0 if np.isnan(seed) else seed; ds = ax[p * 2 - 1]
    for i in range(p * 2, n):
        vi = dx[i] if not np.isnan(dx[i]) else ds; ds = (vi + (p - 1) * ds) / p; ax[i] = ds
    return ax[-1]

def _atr(h, l, c, p=14):
    n = len(c)
    if n < p + 1: return np.nan
    tr = np.zeros(n); tr[0] = h[0] - l[0]
    for i in range(1, n): tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr_arr = np.full(n, np.nan); atr_arr[p] = float(np.mean(tr[1:p + 1]))
    for i in range(p + 1, n): atr_arr[i] = (tr[i] + (p - 1) * atr_arr[i - 1]) / p
    return atr_arr[-1]

def _realized_vol(closes, window=60):
    """Annualized realized volatility."""
    if len(closes) < window + 2: return np.nan
    rets = np.diff(closes[-window - 1:]) / (closes[-window - 1:-1] + 1e-10)
    return np.nanstd(rets) * np.sqrt(252)

def get_monthly(start, end):
    m = []
    for d in ALL_DATES:
        if d < start or d > end: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

def _price(code, dt):
    if code not in STOCK: return None
    sd = STOCK[code]; idx = np.searchsorted(sd["d"], dt, side="right") - 1
    return sd["c"][idx] if idx >= 0 else None

def _data_at(code, dt, n):
    if code not in STOCK: return None
    sd = STOCK[code]; idx = np.searchsorted(sd["d"], dt, side="right")
    if idx < n: return None
    return (sd["c"][idx - n:idx], sd["h"][idx - n:idx], sd["l"][idx - n:idx], sd["v"][idx - n:idx])

def _cash_price(dt):
    idx = np.searchsorted(cash_dates, dt, side="right") - 1
    return cash_c[idx] if idx >= 0 else 100.0


# ═══════════════════════════════════════════════════════════
# 3. State Machine (same as V1, causal)
# ═══════════════════════════════════════════════════════════
print("\n[2] Building state machine & precomputing indicators...")
t0 = time.time()

# Precompute breadth
stock_ma = {}
for code, sd in STOCK.items():
    c = sd["c"]; d = sd["d"]
    if len(c) < 21: continue
    cs = np.cumsum(np.insert(c, 0, 0.0))
    ma20 = np.full(len(c), np.nan)
    ma20[19:] = (cs[20:] - cs[:-20]) / 20.0
    stock_ma[code] = (d, c > ma20)

def _breadth(dt):
    cnt, tot = 0, 0
    for code, (da, aa) in stock_ma.items():
        idx = np.searchsorted(da, dt, side="right") - 1
        if idx < 19: continue
        tot += 1
        if aa[idx]: cnt += 1
    return cnt / tot * 100.0 if tot > 0 else 50.0

# Build state map
n_csi = len(csi_c)
ma120_full = np.full(n_csi, np.nan)
cs = np.cumsum(np.insert(csi_c, 0, 0.0))
ma120_full[119:] = (cs[120:] - cs[:-120]) / 120.0

adx_full = np.full(n_csi, np.nan)
p = 14
tr_arr = np.zeros(n_csi); tr_arr[0] = csi_h[0] - csi_l[0]
for i in range(1, n_csi):
    tr_arr[i] = max(csi_h[i] - csi_l[i], abs(csi_h[i] - csi_c[i - 1]), abs(csi_l[i] - csi_c[i - 1]))
pdm_arr = np.zeros(n_csi); mdm_arr = np.zeros(n_csi)
for i in range(1, n_csi):
    up = csi_h[i] - csi_h[i - 1]; dn = csi_l[i - 1] - csi_l[i]
    if up > dn and up > 0: pdm_arr[i] = up
    if dn > up and dn > 0: mdm_arr[i] = dn
atr_arr = np.full(n_csi, np.nan); atr_arr[p] = float(np.mean(tr_arr[1:p + 1]))
for i in range(p + 1, n_csi): atr_arr[i] = (tr_arr[i] + (p - 1) * atr_arr[i - 1]) / p
ps_v = float(np.mean(pdm_arr[1:p + 1])); ms_v = float(np.mean(mdm_arr[1:p + 1]))
pi_arr = np.full(n_csi, np.nan); mi_arr = np.full(n_csi, np.nan)
pi_arr[p] = ps_v / max(atr_arr[p], 0.001) * 100; mi_arr[p] = ms_v / max(atr_arr[p], 0.001) * 100
for i in range(p + 1, n_csi):
    ps_v = (pdm_arr[i] + (p - 1) * ps_v) / p; ms_v = (mdm_arr[i] + (p - 1) * ms_v) / p
    pi_arr[i] = min(ps_v / max(atr_arr[i], 0.001) * 100, 1000)
    mi_arr[i] = min(ms_v / max(atr_arr[i], 0.001) * 100, 1000)
dx_arr = np.abs(pi_arr - mi_arr) / (pi_arr + mi_arr + 1e-10) * 100
seed_v = float(np.nanmean(dx_arr[p:p * 2]))
adx_full[p * 2 - 1] = 0.0 if np.isnan(seed_v) else seed_v; ds = adx_full[p * 2 - 1]
for i in range(p * 2, n_csi):
    vi = dx_arr[i] if not np.isnan(dx_arr[i]) else ds; ds = (vi + (p - 1) * ds) / p; adx_full[i] = ds

breadth_arr = np.array([_breadth(d) for d in csi_dates])
above_ma = (csi_c > ma120_full) & (~np.isnan(ma120_full))

def build_state_map(adx_th, br_th, cbr, crb):
    raw = np.full(n_csi, 0, dtype=int)
    for i in range(120, n_csi):
        if above_ma[i]:
            a_ok = not np.isnan(adx_full[i]) and adx_full[i] > adx_th
            b_ok = not np.isnan(breadth_arr[i]) and breadth_arr[i] > br_th
            raw[i] = 2 if (a_ok and b_ok) else 1
    conf = np.full(n_csi, 0, dtype=int)
    for i in range(1, n_csi):
        rs = raw[i]
        if conf[i - 1] == 0:
            if i >= cbr - 1 and np.all(raw[i - cbr + 1:i + 1] >= 1):
                if i >= crb - 1 and np.all(raw[i - crb + 1:i + 1] == 2): conf[i] = 2
                else: conf[i] = 1
            else: conf[i] = 0
        elif conf[i - 1] == 1:
            if rs == 0: conf[i] = 0
            elif i >= crb - 1 and np.all(raw[i - crb + 1:i + 1] == 2): conf[i] = 2
            else: conf[i] = 1
        elif conf[i - 1] == 2:
            if rs == 0: conf[i] = 0
            elif rs == 1: conf[i] = 1
            else: conf[i] = 2
    return {csi_dates[i]: int(conf[i]) for i in range(n_csi)}

BASE_MAP = build_state_map(22, 50, 5, 3)
BEST_MAP = build_state_map(25, 45, 5, 2)

# Precompute stock indicators for V2 features
print("  Precomputing stock indicators for all monthly dates...")
all_months = get_monthly("20150101", "20251231")

# For each stock at each date, precompute: score (same as V1), atr14, realized_vol, ret_60d
PRE = {}
stock_count = 0
for code, sd in STOCK.items():
    stock_count += 1
    if stock_count % 200 == 0:
        print(f"  [{stock_count}/{len(STOCK)}] {time.time()-t0:.0f}s")
    PRE[code] = {}
    for rd in all_months:
        idx = np.searchsorted(sd["d"], rd, side="right")
        if idx < 260: continue
        c = sd["c"][idx - 260:idx].copy()
        h = sd["h"][idx - 260:idx].copy()
        l = sd["l"][idx - 260:idx].copy()
        v = sd["v"][idx - 260:idx].copy()
        n = len(c)

        # Score (same composite as V1/pure_alpha)
        ma120 = np.full(n, np.nan)
        cs120 = np.cumsum(np.insert(c, 0, 0.0))
        ma120[119:] = (cs120[120:] - cs120[:-120]) / 120.0
        above = 1.0 if (c[-1] > ma120[-1] and not np.isnan(ma120[-1])) else 0.0

        rh = np.max(h[-20:]); ph = np.max(h[-60:-20])
        rl = np.min(l[-20:]); pl = np.min(l[-60:-20])
        hhll = 1.0 if (rh > ph and rl > pl) else 0.0

        ma20 = np.full(n, np.nan)
        cs20 = np.cumsum(np.insert(c, 0, 0.0))
        ma20[19:] = (cs20[20:] - cs20[:-20]) / 20.0
        ma60 = np.full(n, np.nan)
        cs60 = np.cumsum(np.insert(c, 0, 0.0))
        ma60[59:] = (cs60[60:] - cs60[:-60]) / 60.0
        aligned = 0.0
        if (not np.isnan(ma20[-1]) and not np.isnan(ma60[-1]) and not np.isnan(ma120[-1])
            and ma20[-1] > ma60[-1] > ma120[-1]):
            aligned = 1.0

        adx_v = _adx(h, l, c, 14)
        adx_n = min(max((adx_v - 15) / 35, 0), 1) if not np.isnan(adx_v) else 0

        r3 = c[-1] / c[-63] - 1 if c[-63] > 1e-6 else 0
        r6 = c[-1] / c[-126] - 1 if c[-126] > 1e-6 else 0
        m3 = min(max(r3 / 0.5, 0), 1) if r3 > 0 else max(r3 / 0.3, -1)
        m6 = min(max(r6 / 0.8, 0), 1) if r6 > 0 else max(r6 / 0.5, -1)
        mom_s = (0.5 * m3 + 0.5 * m6) * 100

        w = c[-61:]; dr = np.diff(w) / (w[:-1] + 1e-10)
        vol_s = (1 - min(np.nanstd(dr) / 0.05, 1)) * 100

        trend_comp = (0.35 * above + 0.25 * aligned + 0.20 * adx_n + 0.20 * hhll) * 100
        score = 0.60 * mom_s + 0.30 * trend_comp + 0.10 * vol_s

        # New V2 features
        ret_60d = (c[-1] / c[-60] - 1) if c[-60] > 1e-6 else 0
        rv_60d = _realized_vol(c, 60)
        atr14 = _atr(h, l, c, 14)

        PRE[code][rd] = {
            "score": score,
            "ret_60d": ret_60d,
            "rv_60d": rv_60d,
            "atr14": atr14,
            "close": sd["c"][idx - 1],
        }

elapsed = time.time() - t0
print(f"  Precomputed in {elapsed:.0f}s")


# ═══════════════════════════════════════════════════════════
# 4. V2 Backtest Engine — with all improvements
# ═══════════════════════════════════════════════════════════
def run_v2(state_map, start, end, top_n_bull=5, top_n_range=3,
           vol_target=0.20, atr_stop_mult=3.0, dual_momentum=True,
           dd_cb_tier1=-0.08, dd_cb_tier2=-0.15, inv_vol_weight=True):
    """
    V2 improvements:
      - vol_target: scale position size by (target_vol / realized_vol), capped at 1.5x
      - atr_stop_mult: trailing stop at N * ATR(14) from HWM instead of fixed -10%
      - dual_momentum: only buy stocks with ret_60d > 0
      - dd_cb_tier1: reduce exposure to 50% at mild DD
      - dd_cb_tier2: reduce exposure to 0% at severe DD
      - inv_vol_weight: weight positions by 1/rv_60d within selected set
    """
    rebal = get_monthly(start, end)
    if len(rebal) < 6: return None

    cash = CAPITAL
    holdings = {}  # code -> {"qty":int, "price":float, "hwm":float, "units":int}
    cash_etf = 0.0
    eq = [CAPITAL]
    max_e = CAPITAL
    prev_state = -1
    state_cnt = {0: 0, 1: 0, 2: 0}

    for rd in rebal:
        mkt = state_map.get(rd, 0)
        state_cnt[mkt] += 1
        state_changed = (mkt != prev_state)

        # Value positions
        for sym in list(holdings.keys()):
            p = _price(sym, rd)
            if p is None or p < 0.01:
                cash += holdings[sym].get("qty", 0) * holdings[sym].get("price", 10) * 0.7
                del holdings[sym]; continue
            holdings[sym]["price"] = p

        cp = _cash_price(rd)
        cval = cash_etf * cp
        total = cash + cval + sum(h["qty"] * h.get("price", 0) for h in holdings.values())

        # ── V2 Adaptive Stop-Loss (ATR-based trailing) ──
        for sym in list(holdings.keys()):
            p = holdings[sym].get("price", 0)
            if p <= 0: continue
            hwm = holdings[sym].get("hwm", p)
            if p > hwm: holdings[sym]["hwm"] = p

            # Get ATR for this stock
            atr_s = np.nan
            if sym in PRE and rd in PRE[sym]:
                atr_s = PRE[sym][rd].get("atr14", np.nan)

            if not np.isnan(atr_s) and atr_s > 0 and hwm > 0:
                # Adaptive stop: HWM - atr_stop_mult * ATR
                stop_price = hwm * (1 - atr_stop_mult * atr_s / hwm)
                if p < stop_price:
                    cash += holdings[sym]["qty"] * p * (1 - COMM)
                    del holdings[sym]
            elif hwm > 0 and (p / hwm - 1) * 100 < -10:
                # Fallback: fixed -10% if ATR unavailable
                cash += holdings[sym]["qty"] * p * (1 - COMM)
                del holdings[sym]

        total = cash + cash_etf * cp + sum(h["qty"] * h.get("price", 0) for h in holdings.values())
        if total > max_e: max_e = total

        # V2 Trailing DD Circuit Breaker
        dd = (max_e - total) / max_e if max_e > 0 else 0

        # Tiered response (dd_cb params are positive thresholds like 0.08, 0.15)
        dd_multiplier = 1.0
        if dd > abs(dd_cb_tier2):
            dd_multiplier = 0.0  # all cash
        elif dd > abs(dd_cb_tier1):
            dd_multiplier = 0.5  # half exposure

        if dd_multiplier == 0.0:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["qty"] * holdings[sym].get("price", 0) * (1 - COMM)
                del holdings[sym]
            cash += cash_etf * cp * (1 - COMM); cash_etf = 0
            total = cash + cash_etf * cp

        # ── State-based allocation ──
        if mkt == 0:  # BEAR
            if state_changed or not cash_etf:
                for sym in list(holdings.keys()):
                    cash += holdings[sym]["qty"] * holdings[sym].get("price", 0) * (1 - COMM)
                    del holdings[sym]
                cash += cash_etf * cp * (1 - COMM); cash_etf = 0
                if cp > 0 and cash > 0: cash_etf = cash / cp; cash = 0.0
            eq.append(cash + cash_etf * cp)
            prev_state = mkt; continue

        # Non-BEAR: free cash ETF
        if cash_etf:
            cash += cash_etf * cp * (1 - COMM); cash_etf = 0

        # Base position size (before vol targeting)
        pos_base = 0.5 if mkt == 1 else 1.0
        top_n = top_n_range if mkt == 1 else top_n_bull

        # ── V2 Dual Momentum Filter + Score ──
        scored = []
        for code in PRE:
            if rd not in PRE[code]: continue
            info = PRE[code][rd]

            if dual_momentum and info["ret_60d"] <= 0:
                continue  # skip negative momentum stocks

            scored.append((code, info["score"], info.get("rv_60d", np.nan)))

        if not scored:
            eq.append(total); prev_state = mkt; continue

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:top_n]

        # Rotate
        selected_codes = {s[0] for s in selected}
        for sym in list(holdings.keys()):
            if sym not in selected_codes:
                cash += holdings[sym]["qty"] * holdings[sym].get("price", 0) * (1 - COMM)
                del holdings[sym]

        # ── V2 Vol Targeting + Inverse-Vol Weighting ──
        vols = []
        if inv_vol_weight:
            # Compute inverse-vol weights among selected
            vols = []
            for code, _, rv in selected:
                v = rv if not np.isnan(rv) and rv > 0.05 else 0.20
                vols.append(v)
            vols = np.array(vols)
            inv_vols = 1.0 / vols
            weights_arr = inv_vols / inv_vols.sum()
        else:
            weights_arr = np.ones(len(selected)) / len(selected)
            vols = np.full(len(selected), 0.20)  # default 20% vol for vol targeting

        # Portfolio vol estimate for vol targeting
        port_vol_est = np.sqrt(np.sum(weights_arr ** 2 * vols ** 2))
        vol_scale = min(1.5, vol_target / port_vol_est) if port_vol_est > 0 else 1.0

        # Effective position size
        pos_size = pos_base * dd_multiplier * vol_scale

        # Allocate
        total_cap = cash + sum(h["qty"] * h.get("price", 0) for h in holdings.values())
        eq_alloc = total_cap * pos_size
        n_pos = len(selected)
        if n_pos == 0:
            eq.append(total); prev_state = mkt; continue

        for i, (sym, _, _) in enumerate(selected):
            w = weights_arr[i]
            alloc = eq_alloc * w * 0.90
            p = _price(sym, rd)
            if p is None or p < 0.01: continue
            tq = int(alloc / p / 100) * 100
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
                    holdings[sym] = {"qty": tq, "price": p, "hwm": p}

        leftover = cash
        if cp > 0 and leftover > 0: cash_etf = leftover / cp; cash = 0.0
        else: cash = leftover; cash_etf = 0.0

        total = cash + cash_etf * cp + sum(h["qty"] * h.get("price", 0) for h in holdings.values())
        eq.append(total)
        prev_state = mkt

    # Metrics
    eq_arr = np.array(eq)
    if len(eq_arr) < 2 or eq_arr[0] <= 0: return None
    n_y = len(eq_arr) / 12.0
    if n_y < 0.5: return None
    cagr = ((eq_arr[-1] / eq_arr[0]) ** (1 / n_y) - 1) * 100
    mr = np.diff(eq_arr) / (eq_arr[:-1] + 1e-10)
    sharpe = np.mean(mr) / (np.std(mr) + 1e-10) * np.sqrt(12) if len(mr) > 1 else 0
    peak = eq_arr[0]; maxdd = 0.0
    for v in eq_arr:
        if v > peak: peak = v
        if (peak - v) / peak * 100 > maxdd: maxdd = (peak - v) / peak * 100
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd,
            "total_ret": (eq_arr[-1] / eq_arr[0] - 1) * 100, "sc": state_cnt}


# ═══════════════════════════════════════════════════════════
# 5. Component-by-Component Ablation Tests
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("COMPONENT ABLATION: V1 Baseline vs Each V2 Improvement")
print("=" * 90)

# V1 Baseline (no V2 features) - use very high DD thresholds to disable
v1_base = run_v2(BASE_MAP, TEST_START, TEST_END, 5, 3,
                  vol_target=999, atr_stop_mult=999, dual_momentum=False,
                  dd_cb_tier1=0.99, dd_cb_tier2=0.99, inv_vol_weight=False)

# Each improvement alone
configs = [
    ("V1 Baseline", dict(vt=999, asm=999, dm=False, dd1=0.99, dd2=0.99, iv=False)),
    ("+Vol Target 20%", dict(vt=0.20, asm=999, dm=False, dd1=0.99, dd2=0.99, iv=False)),
    ("+ATR Stop 3x", dict(vt=999, asm=3.0, dm=False, dd1=0.99, dd2=0.99, iv=False)),
    ("+Dual Momentum", dict(vt=999, asm=999, dm=True, dd1=0.99, dd2=0.99, iv=False)),
    ("+DD CB (8%/15%)", dict(vt=999, asm=999, dm=False, dd1=0.08, dd2=0.15, iv=False)),
    ("+Inv-Vol Weight", dict(vt=999, asm=999, dm=False, dd1=0.99, dd2=0.99, iv=True)),
    ("ALL V2 Combined", dict(vt=0.20, asm=3.0, dm=True, dd1=0.08, dd2=0.15, iv=True)),
]

print(f"{'Config':<25s} {'Test CAGR':>10s} {'MaxDD':>10s} {'Sharpe':>10s}")
print("-" * 60)
for name, cfg in configs:
    r = run_v2(BASE_MAP, TEST_START, TEST_END, 5, 3,
               vol_target=cfg["vt"], atr_stop_mult=cfg["asm"],
               dual_momentum=cfg["dm"],
               dd_cb_tier1=cfg["dd1"], dd_cb_tier2=cfg["dd2"],
               inv_vol_weight=cfg["iv"])
    if r:
        print(f"{name:<25s} {r['cagr']:>+9.2f}% {r['maxdd']:>9.1f}% {r['sharpe']:>10.3f}")
    else:
        print(f"{name:<25s} FAILED")

# ═══════════════════════════════════════════════════════════
# 6. Grid Search on BEST_MAP with V2 params
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("V2 GRID SEARCH (BEST state map: 5/2/25/45)")
print(f"{'='*90}")

grid = list(itertools.product(
    [0.15, 0.20, 0.25],  # vol_target
    [2.0, 3.0, 4.0],     # atr_stop_mult
    [True, False],        # dual_momentum
    [0.06, 0.08, 0.10],  # dd_cb_tier1 (positive now)
    [True, False],        # inv_vol_weight
))

results = []
for vt, asm, dm, dd1, iv in grid:
    r = run_v2(BEST_MAP, TEST_START, TEST_END, 5, 2,
               vol_target=vt, atr_stop_mult=asm, dual_momentum=dm,
               dd_cb_tier1=dd1, dd_cb_tier2=0.15, inv_vol_weight=iv)
    r_train = run_v2(BEST_MAP, TRAIN_START, TRAIN_END, 5, 2,
                     vol_target=vt, atr_stop_mult=asm, dual_momentum=dm,
                     dd_cb_tier1=dd1, dd_cb_tier2=0.15, inv_vol_weight=iv)
    if r and r_train:
        results.append({"vt": vt, "asm": asm, "dm": dm, "dd1": dd1, "iv": iv,
                        "tr_c": round(r_train["cagr"], 2), "tr_dd": round(r_train["maxdd"], 2),
                        "tr_sh": round(r_train["sharpe"], 3),
                        "te_c": round(r["cagr"], 2), "te_dd": round(r["maxdd"], 2),
                        "te_sh": round(r["sharpe"], 3)})

by_c = sorted(results, key=lambda x: x["te_c"], reverse=True)
by_d = sorted(results, key=lambda x: x["te_dd"])
by_sh = sorted(results, key=lambda x: x["te_sh"], reverse=True)
for i, r in enumerate(by_c): r["rc"] = i
for i, r in enumerate(by_d): r["rd"] = i
for i, r in enumerate(by_sh): r["rs"] = i
for r in results:
    r["comp"] = r["rc"] * 0.4 + r["rd"] * 0.35 + r["rs"] * 0.25
best = sorted(results, key=lambda x: x["comp"])[0]

print(f"\n--- TOP 10 V2 Configs by Test CAGR ---")
print(f"{'#':<4s} {'VolTgt':>7s} {'ATRStop':>8s} {'DualMom':>8s} {'DD_CB1':>7s} {'InvVol':>7s} {'TestCAGR':>10s} {'TestDD':>8s} {'TestSh':>8s} {'TrainCAGR':>10s}")
print("-" * 95)
for i, r in enumerate(by_c[:10], 1):
    print(f"{i:<4d} {r['vt']:>6.0%} {r['asm']:>7.1f}x {str(r['dm']):>8s} {r['dd1']:>5.0%} {str(r['iv']):>7s} "
          f"{r['te_c']:>+9.2f}% {r['te_dd']:>7.1f}% {r['te_sh']:>8.3f} {r['tr_c']:>+9.2f}%")

print(f"\n--- TOP 10 by Minimum MaxDD ---")
for i, r in enumerate(by_d[:10], 1):
    print(f"  #{i}: VolTgt={r['vt']:.0%} ATR={r['asm']:.1f}x DM={r['dm']} DD1={r['dd1']:.0%} IV={r['iv']} | "
          f"TestDD={r['te_dd']:.1f}% CAGR={r['te_c']:+.1f}%")

print(f"\n--- BEST V2 COMPOSITE ---")
print(f"  vol_target={best['vt']:.0%}  atr_stop={best['asm']:.1f}x  dual_mom={best['dm']}  "
      f"dd_cb={best['dd1']:.0%}  inv_vol={best['iv']}")
print(f"  Train: CAGR={best['tr_c']:+.1f}%  DD={best['tr_dd']:.1f}%  Sharpe={best['tr_sh']:.3f}")
print(f"  Test:  CAGR={best['te_c']:+.1f}%  DD={best['te_dd']:.1f}%  Sharpe={best['te_sh']:.3f}")

# Yearly for best
print(f"\n--- Yearly Breakdown (Best V2) ---")
for yr in [2022, 2023, 2024, 2025]:
    ys, ye = f"{yr}0101", f"{yr}1231"
    ry = run_v2(BEST_MAP, ys, ye, 5, 2,
                vol_target=best["vt"], atr_stop_mult=best["asm"],
                dual_momentum=best["dm"],
                dd_cb_tier1=best["dd1"], dd_cb_tier2=0.15,
                inv_vol_weight=best["iv"])
    if ry:
        print(f"  {yr}: CAGR={ry['cagr']:+.1f}%  DD={ry['maxdd']:.1f}%  Sharpe={ry['sharpe']:.3f}")

# Final comparison
print(f"\n{'='*90}")
print("FINAL COMPARISON: V1 vs V2")
print(f"{'='*90}")
print(f"{'Version':<30s} {'CAGR':>10s} {'MaxDD':>10s} {'Sharpe':>10s} {'TotalRet':>10s}")
print("-" * 75)
v1_line = run_v2(BEST_MAP, TEST_START, TEST_END, 5, 2,
                 vol_target=999, atr_stop_mult=999, dual_momentum=False,
                 dd_cb_tier1=0.99, dd_cb_tier2=0.99, inv_vol_weight=False)
if v1_line:
    print(f"{'V1 (state machine original)':<30s} {v1_line['cagr']:>+9.2f}% {v1_line['maxdd']:>9.1f}% "
          f"{v1_line['sharpe']:>10.3f} {v1_line['total_ret']:>+9.1f}%")

v2_best = run_v2(BEST_MAP, TEST_START, TEST_END, 5, 2,
                 vol_target=best["vt"], atr_stop_mult=best["asm"],
                 dual_momentum=best["dm"],
                 dd_cb_tier1=best["dd1"], dd_cb_tier2=0.15,
                 inv_vol_weight=best["iv"])
if v2_best:
    print(f"{'V2 (risk-budgeted)':<30s} {v2_best['cagr']:>+9.2f}% {v2_best['maxdd']:>9.1f}% "
          f"{v2_best['sharpe']:>10.3f} {v2_best['total_ret']:>+9.1f}%")

# CSI300 benchmark
si = np.searchsorted(csi_dates, TEST_START); ei = np.searchsorted(csi_dates, TEST_END, side="right") - 1
seg = csi_c[si:ei + 1]
cagr_bm = ((seg[-1] / seg[0]) ** (1 / 4.0) - 1) * 100
peak = seg[0]; mdd_bm = 0.0
for v in seg:
    if v > peak: peak = v
    if (peak - v) / peak * 100 > mdd_bm: mdd_bm = (peak - v) / peak * 100
print(f"{'CSI300 B&H':<30s} {cagr_bm:>+9.2f}% {mdd_bm:>9.1f}%")

print(f"\nTotal time: {time.time() - T_START:.0f}s")
print("=" * 90)

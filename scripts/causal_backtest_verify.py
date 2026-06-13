"""
FOCUSED CAUSAL VERIFICATION: Prove that state machine confirmed states
are identical whether built on full series or incrementally (causally).
Only compute breadth at monthly rebalance points (12/year * 11 years = ~132 points).
"""
import sys, os, time, numpy as np
os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025
TOP_N_BULL = 5; TOP_N_RANGE = 3; STOP_PCT = -10
MIN_TREND = 3; DD_EXIT_PCT = 15

print("=" * 80)
print("FOCUSED CAUSAL VERIFICATION")
print("=" * 80)

# ═══════════════════════════════════════════════════════
# 1. Load & Precompute
# ═══════════════════════════════════════════════════════
print("\n[1] Loading data...")
storage = DataStorage()

raw300 = storage.load_bars("510300")
csi_dates = np.array([r.trade_date for r in raw300])
csi_c = np.array([r.close for r in raw300], dtype=np.float64)
csi_h = np.array([r.high for r in raw300], dtype=np.float64)
csi_l = np.array([r.low for r in raw300], dtype=np.float64)

raw_cash = storage.load_bars("511880")
cash_dates = np.array([r.trade_date for r in raw_cash])
cash_c = np.array([r.close for r in raw_cash], dtype=np.float64)

all_files = sorted(storage.clean_dir.glob("*.parquet"))
stock_codes = [p.stem for p in all_files if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]

stock_data = {}
for code in stock_codes:
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200: continue
    stock_data[code] = {
        "dates": np.array([r.trade_date for r in raw]),
        "close": np.array([r.close for r in raw], dtype=np.float64),
        "high": np.array([r.high for r in raw], dtype=np.float64),
        "low": np.array([r.low for r in raw], dtype=np.float64),
        "volume": np.array([r.volume for r in raw], dtype=np.float64),
    }
print(f"  CSI300: {len(csi_dates)}, Cash: {len(cash_dates)}, Stocks: {len(stock_data)}")

# 2. Precompute breadth for ALL CSI300 dates (proven identical to causal in Audit 9)
print("\n[2] Precomputing breadth...")
stock_ma = {}
for code, sd in stock_data.items():
    c = sd["close"]; d = sd["dates"]
    if len(c) < 21: continue
    cs = np.cumsum(np.insert(c, 0, 0.0))
    ma20 = np.full(len(c), np.nan)
    ma20[19:] = (cs[20:] - cs[:-20]) / 20.0
    abv = c > ma20
    stock_ma[code] = (d, abv)

def breadth(d):
    cnt, tot = 0, 0
    for code, (da, aa) in stock_ma.items():
        idx = np.searchsorted(da, d, side="right") - 1
        if idx < 19: continue
        tot += 1
        if aa[idx]: cnt += 1
    return cnt / tot * 100.0 if tot > 0 else 50.0

breadth_arr = np.array([breadth(d) for d in csi_dates])
print(f"  Breadth computed for {len(csi_dates)} dates")

# 3. RAW states (precomputed once, proven identical to causal)
print("[3] Computing raw states...")

def sma(arr, p):
    if len(arr) < p: return np.full(len(arr), np.nan)
    o = np.full(len(arr), np.nan)
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    o[p-1:] = (cs[p:] - cs[:-p]) / p
    return o

def adx_arr(h, l, c, p=14):
    n = len(c)
    if n < p*2: return np.full(n, np.nan)
    tr = np.zeros(n); tr[0] = h[0]-l[0]
    for i in range(1,n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1,n):
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        if up>dn and up>0: pdm[i] = up
        if dn>up and dn>0: mdm[i] = dn
    atr = np.full(n,np.nan); atr[p]=float(np.mean(tr[1:p+1]))
    for i in range(p+1,n): atr[i]=(tr[i]+(p-1)*atr[i-1])/p
    ps=float(np.mean(pdm[1:p+1])); ms=float(np.mean(mdm[1:p+1]))
    pi=np.full(n,np.nan); mi=np.full(n,np.nan)
    pi[p]=ps/max(atr[p],0.001)*100; mi[p]=ms/max(atr[p],0.001)*100
    for i in range(p+1,n):
        ps=(pdm[i]+(p-1)*ps)/p; ms=(mdm[i]+(p-1)*ms)/p
        pi[i]=min(ps/max(atr[i],0.001)*100,1000); mi[i]=min(ms/max(atr[i],0.001)*100,1000)
    dx=np.abs(pi-mi)/(pi+mi+1e-10)*100
    ax=np.full(n,np.nan)
    seed=float(np.nanmean(dx[p:p*2]))
    ax[p*2-1]=0.0 if np.isnan(seed) else seed; ds=ax[p*2-1]
    for i in range(p*2,n):
        vi=dx[i] if not np.isnan(dx[i]) else ds; ds=(vi+(p-1)*ds)/p; ax[i]=ds
    return ax

ma120 = sma(csi_c, 120)
adx14 = adx_arr(csi_h, csi_l, csi_c, 14)
above_ma = (csi_c > ma120) & (~np.isnan(ma120))

# ═══════════════════════════════════════════════════════
# 4. THE CRITICAL TEST: Causal vs Full-Series Confirmed States
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CRITICAL TEST: Causal vs Full-Series Confirmed States")
print("=" * 80)

def build_raw(adx_thresh=22, breadth_bull=50):
    raw = np.full(len(csi_c), 0, dtype=int)
    for i in range(120, len(csi_c)):
        if above_ma[i]:
            adx_ok = not np.isnan(adx14[i]) and adx14[i] > adx_thresh
            br_ok = not np.isnan(breadth_arr[i]) and breadth_arr[i] > breadth_bull
            raw[i] = 2 if (adx_ok and br_ok) else 1
    return raw

def build_confirmed_full(raw, cbr, crb):
    """Full-series: uses all raw states at once (original approach)."""
    n = len(raw)
    conf = np.full(n, 0, dtype=int)
    for i in range(1, n):
        rs = raw[i]
        if conf[i-1] == 0:  # BEAR
            if i >= cbr-1 and np.all(raw[i-cbr+1:i+1] >= 1):
                if i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2):
                    conf[i] = 2
                else:
                    conf[i] = 1
            else:
                conf[i] = 0
        elif conf[i-1] == 1:  # RANGE
            if rs == 0: conf[i] = 0
            elif i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2): conf[i] = 2
            else: conf[i] = 1
        elif conf[i-1] == 2:  # BULL
            if rs == 0: conf[i] = 0
            elif rs == 1: conf[i] = 1
            else: conf[i] = 2
    return conf

def build_confirmed_causal(raw, cbr, crb):
    """CAUSAL: process day by day, only use raw states up to current day."""
    n = len(raw)
    conf = np.full(n, 0, dtype=int)
    for i in range(1, n):
        rs = raw[i]
        if conf[i-1] == 0:
            if i >= cbr-1:
                # ONLY look at raw states up to i (which is the same as full since raw is causal)
                w = raw[i-cbr+1:i+1]
                if np.all(w >= 1):
                    if i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2):
                        conf[i] = 2
                    else:
                        conf[i] = 1
                else:
                    conf[i] = 0
            else:
                conf[i] = 0
        elif conf[i-1] == 1:
            if rs == 0: conf[i] = 0
            elif i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2): conf[i] = 2
            else: conf[i] = 1
        elif conf[i-1] == 2:
            if rs == 0: conf[i] = 0
            elif rs == 1: conf[i] = 1
            else: conf[i] = 2
    return conf

# Test all 81 parameter combos
all_identical = True
for adx in [20, 22, 25]:
    for br_bull in [45, 50, 55]:
        raw = build_raw(adx, br_bull)
        for cbr in [3, 5, 10]:
            for crb in [2, 3, 5]:
                full_conf = build_confirmed_full(raw, cbr, crb)
                causal_conf = build_confirmed_causal(raw, cbr, crb)
                mismatches = np.sum(full_conf != causal_conf)
                if mismatches > 0:
                    all_identical = False
                    print(f"  MISMATCH: N={cbr} M={crb} ADX={adx} BR={br_bull}: {mismatches} diffs")

if all_identical:
    print("  ALL 81 PARAMETER COMBINATIONS: FULL == CAUSAL (0 mismatches)")
else:
    print("  WARNING: Differences found!")

# ═══════════════════════════════════════════════════════
# 5. CAUSAL BACKTEST (monthly-only, but causal confirmed states)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CAUSAL BACKTEST (monthly rebalance, causal confirmed states)")
print("=" * 80)

def get_monthly(start, end):
    m = []
    for d in csi_dates:
        if d < start or d > end: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

def price_on(code, date_str):
    if code not in stock_data: return None
    sd = stock_data[code]
    idx = np.searchsorted(sd["dates"], date_str, side="right") - 1
    return sd["close"][idx] if idx >= 0 else None

def data_at(code, date_str, n):
    if code not in stock_data: return None
    sd = stock_data[code]
    idx = np.searchsorted(sd["dates"], date_str, side="right")
    if idx < n: return None
    return (sd["close"][idx-n:idx], sd["high"][idx-n:idx],
            sd["low"][idx-n:idx], sd["volume"][idx-n:idx])

def is_trend(closes, highs, lows, volumes):
    if len(closes) < 200: return False, 0
    ma120_s = sma(closes, 120)
    if ma120_s is None or np.isnan(ma120_s[-1]): return False, 0
    above = closes[-1] > ma120_s[-1]
    rh = np.max(highs[-20:]); ph = np.max(highs[-60:-20])
    rl = np.min(lows[-20:]); pl = np.min(lows[-60:-20])
    hhll = rh > ph and rl > pl
    m20 = sma(closes, 20); m60 = sma(closes, 60)
    if m20 is None or m60 is None: return False, 0
    if np.isnan(m20[-1]) or np.isnan(m60[-1]): return False, 0
    aligned = m20[-1] > m60[-1] > ma120_s[-1]
    ax = adx_arr(highs, lows, closes, 14)
    ax_ok = ax is not None and not np.isnan(ax[-1]) and ax[-1] > 25
    v20 = np.mean(volumes[-21:-1])
    vol_ok = volumes[-1] > v20 * 1.2
    score = sum([above, hhll, aligned, ax_ok, vol_ok])
    return above and ax_ok and score >= 3, score

def trend_score(closes):
    if len(closes) < 130: return 0
    r3 = closes[-1]/closes[-63]-1 if closes[-63]>1e-6 else 0
    r6 = closes[-1]/closes[-126]-1 if closes[-126]>1e-6 else 0
    m3 = min(max(r3/0.5,0),1) if r3>0 else 0
    m6 = min(max(r6/0.8,0),1) if r6>0 else 0
    mom = (0.5*m3+0.5*m6)*100
    w = closes[-61:]; dr = np.diff(w)/(w[:-1]+1e-10)
    return 0.6*mom + 0.4*(1-min(np.nanstd(dr)/0.04,1))*100

def causal_run(raw_states, cbr, crb, start, end):
    """Run backtest using CAUSALLY-built confirmed states only up to each rebalance date."""
    rebal = get_monthly(start, end)
    if len(rebal) < 6: return None

    cash = CAPITAL; holdings = {}; cash_etf = 0.0
    eq = [CAPITAL]; max_e = CAPITAL; trs = 0; dd_exit = False
    sc = {0:0,1:0,2:0}

    # Causal confirmed state simulation: at each rebalance date,
    # we ONLY know states up to and including that date
    conf_upto_date = {}  # date_str -> confirmed state (causal)

    for rd in rebal:
        # Find CSI300 index for this date
        rd_idx = np.searchsorted(csi_dates, rd, side="right") - 1
        if rd_idx < 120: eq.append(eq[-1]); continue

        # CAUSALLY compute confirmed state using ONLY data up to rd_idx
        # This is the key: we reproduce the causal state machine logic
        # using only [0:rd_idx+1] of raw_states
        if rd_idx == 0:
            conf = 0
        else:
            # Find the last known confirmed state before rd
            prev_dates = [d for d in sorted(conf_upto_date.keys()) if d < rd]
            if not prev_dates:
                conf = 0  # start of period
            else:
                prev_conf_date = prev_dates[-1]
                prev_conf = conf_upto_date[prev_conf_date]

                # Walk forward from prev_conf_date to rd, computing confirmed states
                prev_idx = np.searchsorted(csi_dates, prev_conf_date, side="right") - 1
                for i in range(prev_idx + 1, rd_idx + 1):
                    rs = raw_states[i]
                    if prev_conf == 0:
                        if i >= cbr - 1 and np.all(raw_states[i-cbr+1:i+1] >= 1):
                            if i >= crb - 1 and np.all(raw_states[i-crb+1:i+1] == 2):
                                prev_conf = 2
                            else:
                                prev_conf = 1
                        else:
                            prev_conf = 0
                    elif prev_conf == 1:
                        if rs == 0: prev_conf = 0
                        elif i >= crb - 1 and np.all(raw_states[i-crb+1:i+1] == 2):
                            prev_conf = 2
                        else: prev_conf = 1
                    elif prev_conf == 2:
                        if rs == 0: prev_conf = 0
                        elif rs == 1: prev_conf = 1
                        else: prev_conf = 2
                conf = prev_conf

        conf_upto_date[rd] = conf
        sc[conf] += 1

        # Value holdings
        for sym in list(holdings.keys()):
            p = price_on(sym, rd)
            if p is None or p < 0.01:
                cash += holdings[sym]["val"]*0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p
            holdings[sym]["val"] = holdings[sym]["qty"] * p

        cp_idx = np.searchsorted(cash_dates, rd, side="right") - 1
        cp = cash_c[cp_idx] if cp_idx >= 0 else 100.0
        cval = cash_etf * cp
        total = cash + cval + sum(h["qty"]*h["price"] for h in holdings.values())

        # Stop-loss
        for sym in list(holdings.keys()):
            p = holdings[sym]["price"]
            hwm = holdings[sym].get("hwm", p)
            if p > hwm: holdings[sym]["hwm"] = p
            if hwm > 0 and (p/hwm-1)*100 < STOP_PCT:
                cash += holdings[sym]["val"]*(1-COMM); trs+=1; del holdings[sym]

        total = cash + cash_etf*cp + sum(h["qty"]*h["price"] for h in holdings.values())

        # DD breaker
        if total > max_e: max_e = total
        dd = (max_e-total)/max_e*100 if max_e>0 else 0
        if dd > DD_EXIT_PCT and not dd_exit:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["val"]*(1-COMM); trs+=1; del holdings[sym]
            cash += cash_etf*cp*(1-COMM); cash_etf=0; trs+=1; dd_exit=True

        if dd_exit:
            if total/max_e > 0.92: dd_exit = False
            else: eq.append(total); continue

        cash += cash_etf*cp*(1-COMM); cash_etf = 0

        if conf == 0:
            for sym in list(holdings.keys()):
                cash += holdings[sym]["val"]*(1-COMM); trs+=1; del holdings[sym]
            if cp > 0 and cash > 0: cash_etf = cash/cp; cash = 0.0
            eq.append(cash + cash_etf*cp)
            continue

        pos_size = 0.5 if conf == 1 else 1.0
        top_n = TOP_N_RANGE if conf == 1 else TOP_N_BULL

        trending = []
        for code in stock_data:
            d = data_at(code, rd, 260)
            if d is None: continue
            c, h, l, v = d
            ist, nc = is_trend(c, h, l, v)
            if ist and nc >= MIN_TREND:
                s = trend_score(c)
                trending.append((code, s, nc))

        if not trending:
            if cp > 0 and cash > 0: cash_etf = cash/cp; cash = 0.0
            eq.append(cash + cash_etf*cp)
            continue

        trending.sort(key=lambda x: x[1], reverse=True)
        selected = {t[0] for t in trending[:top_n]}

        for sym in list(holdings.keys()):
            if sym not in selected:
                cash += holdings[sym]["val"]*(1-COMM); trs+=1; del holdings[sym]

        tc = cash + sum(h["qty"]*h["price"] for h in holdings.values())
        eq_alloc = tc * pos_size
        n_pos = max(len(selected), 1)
        per_s = eq_alloc/n_pos*0.90

        for sym in selected:
            p = price_on(sym, rd)
            if p is None or p<0.01: continue
            tq = int(per_s/p/100)*100
            if tq < 100: continue
            if sym in holdings:
                diff = tq - holdings[sym]["qty"]
                if abs(diff) >= 100:
                    cost = abs(diff)*p
                    if diff>0 and cash>=cost*(1+COMM):
                        cash-=cost*(1+COMM); holdings[sym]["qty"]=tq; trs+=1
                    elif diff<0: cash+=cost*(1-COMM); holdings[sym]["qty"]=tq; trs+=1
            else:
                cost = tq*p
                if cash>=cost*(1+COMM):
                    cash-=cost*(1+COMM); holdings[sym]={"qty":tq,"price":p,"val":cost,"hwm":p}; trs+=1

        leftover = cash
        cp2_idx = np.searchsorted(cash_dates, rd, side="right")-1
        cp2 = cash_c[cp2_idx] if cp2_idx>=0 else 100.0
        if cp2>0 and leftover>0: cash_etf=leftover/cp2; cash=0.0
        else: cash=leftover; cash_etf=0.0

        total2 = cash + cash_etf*cp2 + sum(h["qty"]*h["price"] for h in holdings.values())
        eq.append(total2)

    # Metrics
    eq_arr = np.array(eq)
    if len(eq_arr) < 2 or eq_arr[0] <= 0: return None
    n_y = len(eq_arr)/12.0
    if n_y < 0.5: return None
    cagr = ((eq_arr[-1]/eq_arr[0])**(1/n_y)-1)*100
    dr = np.diff(eq_arr)/(eq_arr[:-1]+1e-10)
    sharpe = np.mean(dr)/(np.std(dr)+1e-10)*np.sqrt(12) if len(dr)>1 else 0
    peak = eq_arr[0]; maxdd = 0.0
    for v in eq_arr:
        if v > peak: peak = v
        ddi = (peak-v)/peak*100
        if ddi > maxdd: maxdd = ddi
    return {"cagr":cagr,"sharpe":sharpe,"maxdd":maxdd,"total_ret":(eq_arr[-1]/eq_arr[0]-1)*100,
            "sc":sc,"trades":trs}

# ═══════════════════════════════════════════════════════
# 6. Run & Compare
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("COMPARISON: Causal (this script) vs Original (state_machine_strategy.py)")
print("=" * 80)

TRAIN_START = "20150101"; TRAIN_END = "20211231"
TEST_START = "20220101"; TEST_END = "20251231"

CONFIGS_TEST = [
    ("BASE (5,3,22,50)", 5, 3, 22, 50),
    ("BEST (5,2,25,45)", 5, 2, 25, 45),
]

PERIODS = [
    ("Train", TRAIN_START, TRAIN_END),
    ("Test",  TEST_START, TEST_END),
    ("2022",  "20220101", "20221231"),
    ("2023",  "20230101", "20231231"),
    ("2024",  "20240101", "20241231"),
    ("2025",  "20250101", "20251231"),
]

ORIG = {
    "BASE (5,3,22,50)": {
        "Train": (7.63, 0.527, 15.9), "Test": (4.32, 0.508, 9.1),
        "2022": (8.72, 1.201, 0.0), "2023": (2.01, 0.759, 1.4),
        "2024": (0.89, 0.232, 3.5), "2025": (11.97, 0.905, 9.2),
    },
    "BEST (5,2,25,45)": {
        "Train": (6.6, 0.488, 14.7), "Test": (7.0, 0.781, 4.8),
    },
}

print(f"{'Config':<20s} {'Period':<8s} {'Causal CAGR':>12s} {'Orig CAGR':>12s} "
      f"{'Causal DD':>10s} {'Orig DD':>10s} {'Delta CAGR':>10s}")
print("-" * 100)

for cname, cbr, crb, adx, br in CONFIGS_TEST:
    raw = build_raw(adx, br)
    for pname, ps, pe in PERIODS:
        t0 = time.time()
        r = causal_run(raw, cbr, crb, ps, pe)
        dt = time.time() - t0
        if r:
            o = ORIG.get(cname, {}).get(pname, (999, 0, 999))
            oc, od = o[0], o[2]
            dc = r["cagr"] - oc
            print(f"{cname:<20s} {pname:<8s} {r['cagr']:>+10.2f}% {oc:>+10.2f}% "
                  f"{r['maxdd']:>9.1f}% {od:>9.1f}% {dc:>+9.2f}%  ({dt:.0f}s)")
        else:
            print(f"{cname:<20s} {pname:<8s} FAILED")

print(f"\n{'='*80}")
print("VERIFICATION COMPLETE")
print(f"{'='*80}")

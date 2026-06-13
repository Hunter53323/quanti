"""
Turnover analysis for state machine strategy.
Computes monthly turnover rate, separated by state and year.
"""
import sys, os, numpy as np
os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025; STOP_PCT = -10
MIN_TREND = 3; DD_EXIT_PCT = 15
TOP_N_BULL = 5; TOP_N_RANGE = 3

# ═══════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════
print("Loading data...")
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
all_dates_set = set()
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
    all_dates_set.update(r.trade_date for r in raw)
all_dates = sorted(all_dates_set)
print(f"  {len(stock_data)} stocks, {len(all_dates)} trading days")

# ═══════════════════════════════════════════════════════
# Indicator functions
# ═══════════════════════════════════════════════════════
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

def cash_price_on(dt):
    idx = np.searchsorted(cash_dates, dt, side="right") - 1
    return cash_c[idx] if idx >= 0 else 100.0

# ═══════════════════════════════════════════════════════
# Build state machine (same as original)
# ═══════════════════════════════════════════════════════
print("Building state machine...")

stock_ma = {}
for code, sd in stock_data.items():
    c = sd["close"]; d = sd["dates"]
    if len(c) < 21: continue
    cs = np.cumsum(np.insert(c, 0, 0.0))
    ma20 = np.full(len(c), np.nan)
    ma20[19:] = (cs[20:] - cs[:-20]) / 20.0
    stock_ma[code] = (d, c > ma20)

def breadth(dt):
    cnt, tot = 0, 0
    for code, (da, aa) in stock_ma.items():
        idx = np.searchsorted(da, dt, side="right") - 1
        if idx < 19: continue
        tot += 1
        if aa[idx]: cnt += 1
    return cnt / tot * 100.0 if tot > 0 else 50.0

breadth_arr = np.array([breadth(d) for d in csi_dates])

ma120 = sma(csi_c, 120)
adx14 = adx_arr(csi_h, csi_l, csi_c, 14)
above_ma = (csi_c > ma120) & (~np.isnan(ma120))

def build_state_map(cbr, crb, adx_th, br_th):
    n = len(csi_c)
    raw = np.full(n, 0, dtype=int)
    for i in range(120, n):
        if above_ma[i]:
            aok = not np.isnan(adx14[i]) and adx14[i] > adx_th
            bok = not np.isnan(breadth_arr[i]) and breadth_arr[i] > br_th
            raw[i] = 2 if (aok and bok) else 1

    conf = np.full(n, 0, dtype=int)
    for i in range(1, n):
        rs = raw[i]
        if conf[i-1] == 0:
            if i >= cbr-1 and np.all(raw[i-cbr+1:i+1] >= 1):
                if i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2):
                    conf[i] = 2
                else:
                    conf[i] = 1
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

    return {csi_dates[i]: int(conf[i]) for i in range(n)}

state_map = build_state_map(5, 3, 22, 50)

def get_monthly(start, end):
    m = []
    for d in all_dates:
        if d < start or d > end: continue
        dm = d[4:6]
        if not m or dm != m[-1][4:6]: m.append(d)
    return m

# ═══════════════════════════════════════════════════════
# Backtest with turnover tracking
# ═══════════════════════════════════════════════════════
def run_with_turnover(start, end):
    rebal = get_monthly(start, end)
    cash = CAPITAL; holdings = {}; cash_etf = 0.0
    eq = [CAPITAL]; max_e = CAPITAL; trs = 0; dd_exit = False
    sc = {0:0,1:0,2:0}
    turnover_records = []

    for rd in rebal:
        mkt = state_map.get(rd, 0)
        sc[mkt] += 1

        buy_val = 0.0; sell_val = 0.0

        # Value
        for sym in list(holdings.keys()):
            p = price_on(sym, rd)
            if p is None or p<0.01:
                cash += holdings[sym]["val"]*0.7; del holdings[sym]; continue
            holdings[sym]["price"] = p
            holdings[sym]["val"] = holdings[sym]["qty"] * p

        cp = cash_price_on(rd)
        cval = cash_etf * cp
        total = cash + cval + sum(h["qty"]*h["price"] for h in holdings.values())

        # Stop-loss
        for sym in list(holdings.keys()):
            p = holdings[sym]["price"]
            hwm = holdings[sym].get("hwm", p)
            if p > hwm: holdings[sym]["hwm"] = p
            if hwm > 0 and (p/hwm-1)*100 < STOP_PCT:
                sell_val += holdings[sym]["val"]
                cash += holdings[sym]["val"]*(1-COMM); trs+=1; del holdings[sym]

        total = cash + cash_etf*cp + sum(h["qty"]*h["price"] for h in holdings.values())

        if total > max_e: max_e = total
        dd = (max_e-total)/max_e*100 if max_e>0 else 0
        if dd > DD_EXIT_PCT and not dd_exit:
            for sym in list(holdings.keys()):
                sell_val += holdings[sym]["val"]
                cash += holdings[sym]["val"]*(1-COMM); trs+=1; del holdings[sym]
            sell_val += cash_etf * cp
            cash += cash_etf*cp*(1-COMM); cash_etf=0; trs+=1; dd_exit=True

        if dd_exit:
            total = cash + cash_etf * cash_price_on(rd)
            if total/max_e > 0.92: dd_exit = False
            else:
                turnover_records.append((rd, mkt, 0.0, 0.0, total, 0.0))
                eq.append(total); continue

        # Free cash ETF
        sell_val += cash_etf * cp
        cash += cash_etf*cp*(1-COMM); cash_etf = 0

        if mkt == 0:  # BEAR
            for sym in list(holdings.keys()):
                sell_val += holdings[sym]["val"]
                cash += holdings[sym]["val"]*(1-COMM); trs+=1; del holdings[sym]
            cp2 = cash_price_on(rd)
            if cp2 > 0 and cash > 0:
                buy_val += cash
                cash_etf = cash/cp2; cash = 0.0
            total2 = cash + cash_etf*cp2
            t_pct = ((buy_val+sell_val)/2)/total2*100 if total2>0 else 0
            turnover_records.append((rd, mkt, buy_val, sell_val, total2, t_pct))
            eq.append(total2)
            continue

        pos_size = 0.5 if mkt == 1 else 1.0
        top_n = TOP_N_RANGE if mkt == 1 else TOP_N_BULL

        trending = []
        for code in stock_data:
            d = data_at(code, rd, 260)
            if d is None: continue
            c, h, l, v = d
            ist, nc = is_trend(c, h, l, v)
            if ist and nc >= MIN_TREND:
                s = trend_score(c)
                trending.append((code, s, nc))

        if trending:
            trending.sort(key=lambda x: x[1], reverse=True)
            selected = {t[0] for t in trending[:top_n]}

            for sym in list(holdings.keys()):
                if sym not in selected:
                    sell_val += holdings[sym]["val"]
                    cash += holdings[sym]["val"]*(1-COMM); trs+=1; del holdings[sym]

            tc2 = cash + sum(h["qty"]*h["price"] for h in holdings.values())
            eq_alloc = tc2 * pos_size
            n_pos = max(len(selected), 1)
            per_s = eq_alloc / n_pos * 0.90

            for sym in selected:
                p = price_on(sym, rd)
                if p is None or p<0.01: continue
                tq = int(per_s/p/100)*100
                if tq < 100: continue
                if sym in holdings:
                    diff = tq - holdings[sym]["qty"]
                    if abs(diff) >= 100:
                        cost = abs(diff)*p
                        if diff > 0 and cash >= cost*(1+COMM):
                            buy_val += cost; cash -= cost*(1+COMM)
                            holdings[sym]["qty"] = tq; trs += 1
                        elif diff < 0:
                            sell_val += cost; cash += cost*(1-COMM)
                            holdings[sym]["qty"] = tq; trs += 1
                else:
                    cost = tq*p
                    if cash >= cost*(1+COMM):
                        buy_val += cost; cash -= cost*(1+COMM)
                        holdings[sym] = {"qty":tq,"price":p,"val":cost,"hwm":p}; trs+=1

        leftover = cash
        cp3 = cash_price_on(rd)
        if cp3 > 0 and leftover > 0:
            buy_val += leftover; cash_etf = leftover/cp3; cash = 0.0
        else:
            cash = leftover; cash_etf = 0.0

        total3 = cash + cash_etf*cp3 + sum(h["qty"]*h["price"] for h in holdings.values())
        t_pct = ((buy_val+sell_val)/2)/total3*100 if total3>0 else 0
        turnover_records.append((rd, mkt, buy_val, sell_val, total3, t_pct))
        eq.append(total3)

    return turnover_records, sc, eq

# ═══════════════════════════════════════════════════════
# Run analysis
# ═══════════════════════════════════════════════════════
print("=" * 90)
print("TURNOVER ANALYSIS")
print("=" * 90)

for period_name, ps, pe in [("TEST (2022-2025)", "20220101", "20251231"),
                              ("TRAIN (2015-2021)", "20150101", "20211231")]:
    print(f"\n{'─'*90}")
    print(f"PERIOD: {period_name}")
    print(f"{'─'*90}")

    tr, sc, eq_arr = run_with_turnover(ps, pe)

    # By state
    sn = {0: "BEAR", 1: "RANGE", 2: "BULL"}
    for si in [0, 1, 2]:
        sm = [(b, s, v, t) for _, st, b, s, v, t in tr if st == si]
        if sm:
            avg_t = np.mean([t for _, _, _, t in sm])
            med_t = np.median([t for _, _, _, t in sm])
            tot_b = sum(b for b, _, _, _ in sm)
            tot_s = sum(s for _, s, _, _ in sm)
            print(f"  {sn[si]}: {len(sm)}mo  avg_turn={avg_t:.1f}%  med_turn={med_t:.1f}%")
            if avg_t > 5:
                print(f"         TOTAL buy={tot_b:,.0f}  sell={tot_s:,.0f}")

    # Overall
    all_t = [t for _, _, _, _, _, t in tr]
    print(f"  OVERALL: {len(all_t)}mo  avg_turn={np.mean(all_t):.1f}%  med_turn={np.median(all_t):.1f}%")

    # Yearly
    for yr in ["2022", "2023", "2024", "2025"]:
        ym = [(b, s, v, t) for d, _, b, s, v, t in tr if d.startswith(yr)]
        if ym:
            avg_t = np.mean([t for _, _, _, t in ym])
            med_t = np.median([t for _, _, _, t in ym])
            # Annual estimate: avg of 12 months of single-sided turnover
            ann_est = (sum(b for b, _, _, _ in ym) + sum(s for _, s, _, _ in ym)) / 2 / np.mean([v for _, _, v, _ in ym]) * 12
            print(f"  {yr}: {len(ym)}mo  avg_mo_turn={avg_t:.1f}%  med_mo_turn={med_t:.1f}%  annual_est={ann_est:.0f}%")

    # State transition turnovers
    print(f"\n  --- State Transition Turnovers ---")
    prev_st = 0
    for d, st, b, s, v, t in tr:
        if st != prev_st:
            print(f"  {d}: {sn[prev_st]}->{sn[st]}  turn={t:.1f}%  buy={b:,.0f}  sell={s:,.0f}")
        prev_st = st

print(f"\n{'='*90}")
print("DONE")
print(f"{'='*90}")

"""
Alpha Decomposition: Market Timing vs Stock Selection
=====================================================
Isolates the state machine strategy's return into:
  1. TIMING contribution: IF we held CSI300 index (not stocks) during non-BEAR months
  2. ALPHA contribution: Extra return from selecting momentum stocks vs holding index
  3. CASH contribution: Return from 511880 during BEAR months
"""
import sys, os, numpy as np
os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.storage import DataStorage

CAPITAL = 90000
COMM = 0.00025

# ═══════════════════════════════════════════════════════
# Load all data
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

# ═══════════════════════════════════════════════════════
# Build state machine (same as original)
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

# Precompute breadth
print("Computing breadth...")
all_files = sorted(storage.clean_dir.glob("*.parquet"))
stock_codes = [p.stem for p in all_files if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]

stock_data_raw = {}
for code in stock_codes:
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200: continue
    stock_data_raw[code] = {
        "dates": np.array([r.trade_date for r in raw]),
        "close": np.array([r.close for r in raw], dtype=np.float64),
        "high": np.array([r.high for r in raw], dtype=np.float64),
        "low": np.array([r.low for r in raw], dtype=np.float64),
        "volume": np.array([r.volume for r in raw], dtype=np.float64),
    }

stock_ma = {}
for code, sd in stock_data_raw.items():
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
                if i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2): conf[i] = 2
                else: conf[i] = 1
            else: conf[i] = 0
        elif conf[i-1] == 1:
            if rs == 0: conf[i] = 0
            elif i >= crb-1 and np.all(raw[i-crb+1:i+1] == 2): conf[i] = 2
            else: conf[i] = 1
        elif conf[i-1] == 2:
            if rs == 0: conf[i] = 0
            elif rs == 1: conf[i] = 1
            else: conf[i] = 2
    return {csi_dates[i]: int(conf[i]) for i in range(n)}

# ═══════════════════════════════════════════════════════
# Stock selection helpers
# ═══════════════════════════════════════════════════════
def data_at(code, date_str, n):
    if code not in stock_data_raw: return None
    sd = stock_data_raw[code]
    idx = np.searchsorted(sd["dates"], date_str, side="right")
    if idx < n: return None
    return (sd["close"][idx-n:idx], sd["high"][idx-n:idx],
            sd["low"][idx-n:idx], sd["volume"][idx-n:idx])

def price_on(code, date_str):
    if code not in stock_data_raw: return None
    sd = stock_data_raw[code]
    idx = np.searchsorted(sd["dates"], date_str, side="right") - 1
    return sd["close"][idx] if idx >= 0 else None

def is_trend(closes, highs, lows, volumes):
    if len(closes) < 200: return False, 0
    m120 = sma(closes, 120)
    if m120 is None or np.isnan(m120[-1]): return False, 0
    above = closes[-1] > m120[-1]
    rh = np.max(highs[-20:]); ph = np.max(highs[-60:-20])
    rl = np.min(lows[-20:]); pl = np.min(lows[-60:-20])
    hhll = rh > ph and rl > pl
    m20 = sma(closes, 20); m60 = sma(closes, 60)
    if m20 is None or m60 is None: return False, 0
    if np.isnan(m20[-1]) or np.isnan(m60[-1]): return False, 0
    aligned = m20[-1] > m60[-1] > m120[-1]
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

def csi300_price_on(date_str):
    idx = np.searchsorted(csi_dates, date_str, side="right") - 1
    return csi_c[idx] if idx >= 0 else None

def cash_price_on(date_str):
    idx = np.searchsorted(cash_dates, date_str, side="right") - 1
    return cash_c[idx] if idx >= 0 else 100.0

# ═══════════════════════════════════════════════════════
# Decomposition engine: simulate both strategies monthly
# ═══════════════════════════════════════════════════════
def decompose(state_map, start, end, config_label, cbr, crb, adx, br):
    """Simulate 3 strategies and decompose returns."""
    # Find all CSI300 monthly dates in period
    monthly = []
    for d in csi_dates:
        if d < start or d > end: continue
        dm = d[4:6]
        if not monthly or dm != monthly[-1][4:6]: monthly.append(d)

    # Strategy 1: "CSI300 timing only" — holds CSI300 at 0%/50%/100% during BEAR/RANGE/BULL
    # Strategy 2: "Real strategy" — momentum stocks during non-BEAR
    # Both start with CAPITAL

    # Track monthly returns directly, then compound
    csi300_monthly_rets = []
    real_monthly_rets = []

    # CSI300 price tracking
    csi300_prices = {}
    for rd in monthly:
        csi300_prices[rd] = csi300_price_on(rd)

    # Cash ETF prices
    cash_prices = {}
    for rd in monthly:
        cash_prices[rd] = cash_price_on(rd)

    # Real strategy state
    real_cash = CAPITAL
    real_holdings = {}
    real_etf_units = 0
    real_eq = [CAPITAL]
    real_trades = 0
    prev_state = 0

    monthly_detail = []

    for m_idx in range(len(monthly)):
        rd = monthly[m_idx]
        mkt_state = state_map.get(rd, 0)

        # ─── CSI300 Timing: compute monthly return using weight * ret ───
        # Weight: 0 (BEAR), 0.5 (RANGE), 1.0 (BULL)
        if mkt_state == 0:
            w_csi = 0.0
        elif mkt_state == 1:
            w_csi = 0.5
        else:
            w_csi = 1.0

        if m_idx == 0:
            csi300_ret = 0.0  # first month, no prior to compare
        else:
            prev_csi_p = csi300_prices[monthly[m_idx-1]]
            curr_csi_p = csi300_prices[rd]
            if prev_csi_p and prev_csi_p > 0:
                csi300_mom_ret = (curr_csi_p / prev_csi_p - 1) if curr_csi_p else 0
            else:
                csi300_mom_ret = 0

            prev_cp = cash_prices[monthly[m_idx-1]]
            curr_cp = cash_prices[rd]
            cash_mom_ret = (curr_cp / prev_cp - 1) if prev_cp and prev_cp > 0 else 0

            # Apply commission only on state changes
            if m_idx > 0:
                prev_state_csi = state_map.get(monthly[m_idx-1], 0)
                prev_w = 0.0 if prev_state_csi == 0 else (0.5 if prev_state_csi == 1 else 1.0)
                # Cost to rebalance from prev_w to w_csi
                turnover = abs(w_csi - prev_w)
                cost = turnover * COMM  # one-way commission on the change
            else:
                cost = 0

            csi300_ret = w_csi * csi300_mom_ret + (1-w_csi) * cash_mom_ret - cost

        csi300_monthly_rets.append(csi300_ret)

        # ─── Real Strategy (momentum stocks) ───
        # Value holdings
        for sym in list(real_holdings.keys()):
            p = price_on(sym, rd)
            if p is None or p < 0.01: continue
            real_holdings[sym]["price"] = p
            real_holdings[sym]["val"] = real_holdings[sym]["qty"] * p

        cp_real = cash_price_on(rd)
        real_val = real_cash + real_etf_units * cp_real + sum(h["qty"]*h.get("price",0) for h in real_holdings.values())

        # Stop-loss
        for sym in list(real_holdings.keys()):
            p = real_holdings[sym].get("price", 0)
            hwm = real_holdings[sym].get("hwm", p)
            if p > hwm: real_holdings[sym]["hwm"] = p
            if hwm > 0 and (p/hwm-1)*100 < -10:
                real_cash += real_holdings[sym]["val"] * (1-COMM); real_trades += 1
                del real_holdings[sym]

        # State actions
        if mkt_state == 0:  # BEAR
            if prev_state != 0:  # Just entered BEAR
                for sym in list(real_holdings.keys()):
                    real_cash += real_holdings[sym]["val"] * (1-COMM); real_trades += 1
                    del real_holdings[sym]
                # Buy cash ETF
                real_cash += real_etf_units * cp_real * (1-COMM)
                if cp_real > 0 and real_cash > 0:
                    real_etf_units = real_cash / cp_real; real_cash = 0; real_trades += 1
            # else: already in cash ETF, do nothing
        elif mkt_state in (1, 2):  # RANGE or BULL
            if prev_state == 0 or not real_etf_units == 0:  # Just entered non-BEAR or first time
                # Free cash ETF
                if real_etf_units:
                    real_cash += real_etf_units * cp_real * (1-COMM)
                    real_etf_units = 0
                    real_trades += 1

                pos_size = 0.5 if mkt_state == 1 else 1.0
                top_n = 3 if mkt_state == 1 else 5

                # Select stocks
                trending = []
                for code in stock_data_raw:
                    d = data_at(code, rd, 260)
                    if d is None: continue
                    c, h, l, v = d
                    ist, nc = is_trend(c, h, l, v)
                    if ist and nc >= 3:
                        s = trend_score(c)
                        trending.append((code, s, nc))

                if trending:
                    trending.sort(key=lambda x: x[1], reverse=True)
                    selected = {t[0] for t in trending[:top_n]}

                    for sym in list(real_holdings.keys()):
                        if sym not in selected:
                            real_cash += real_holdings[sym]["val"]*(1-COMM); real_trades+=1
                            del real_holdings[sym]

                    tc = real_cash + sum(h["qty"]*h.get("price",0) for h in real_holdings.values())
                    eq_alloc = tc * pos_size
                    n_pos = max(len(selected), 1)
                    per_s = eq_alloc / n_pos * 0.90

                    for sym in selected:
                        p = price_on(sym, rd)
                        if p is None or p < 0.01: continue
                        tq = int(per_s/p/100)*100
                        if tq < 100: continue
                        if sym in real_holdings:
                            diff = tq - real_holdings[sym]["qty"]
                            if abs(diff) >= 100:
                                cost = abs(diff)*p
                                if diff > 0 and real_cash >= cost*(1+COMM):
                                    real_cash -= cost*(1+COMM); real_holdings[sym]["qty"]=tq; real_trades+=1
                                elif diff < 0:
                                    real_cash += cost*(1-COMM); real_holdings[sym]["qty"]=tq; real_trades+=1
                        else:
                            cost = tq*p
                            if real_cash >= cost*(1+COMM):
                                real_cash -= cost*(1+COMM)
                                real_holdings[sym] = {"qty":tq,"price":p,"val":cost,"hwm":p}; real_trades+=1

                    leftover = real_cash
                else:
                    leftover = real_cash

                if cp_real > 0 and leftover > 0:
                    real_etf_units = leftover / cp_real; real_cash = 0
                else:
                    real_cash = leftover; real_etf_units = 0

        real_final = real_cash + real_etf_units * cash_price_on(rd) + sum(h["qty"]*h.get("price",0) for h in real_holdings.values())
        real_eq.append(real_final)

        # Record monthly detail
        csi300_ret_val = csi300_ret
        real_ret_val = (real_eq[-1]/real_eq[-2] - 1)*100 if len(real_eq) >= 2 else 0
        alpha_val = real_ret_val - csi300_ret_val

        monthly_detail.append({
            "date": rd,
            "state": mkt_state,
            "csi300_ret": csi300_ret_val,
            "real_ret": real_ret_val,
            "alpha": alpha_val,
            "w_csi": w_csi,
        })

        prev_state = mkt_state

    # Compound monthly returns into equity curves
    csi300_eq_arr = [CAPITAL]
    for ret in csi300_monthly_rets:
        csi300_eq_arr.append(csi300_eq_arr[-1] * (1 + ret/100))
    csi300_eq_arr = np.array(csi300_eq_arr)

    return monthly_detail, csi300_eq_arr, np.array(real_eq)

# ═══════════════════════════════════════════════════════
# Run decomposition for BASE config
# ═══════════════════════════════════════════════════════
print("=" * 90)
print("ALPHA DECOMPOSITION: Timing vs Stock Selection")
print("=" * 90)

state_map = build_state_map(5, 3, 22, 50)

for period_name, ps, pe in [("TEST (2022-2025)", "20220101", "20251231"),
                              ("TRAIN (2015-2021)", "20150101", "20211231")]:
    print(f"\n{'─'*90}")
    print(f"PERIOD: {period_name}")
    print(f"{'─'*90}")

    detail, csi300_eq, real_eq = decompose(state_map, ps, pe, "BASE", 5, 3, 22, 50)

    # Compute CAGR
    def calc_cagr(eq):
        n_y = (len(eq)-1) / 12.0
        if n_y <= 0 or eq[0] <= 0: return 0
        return ((eq[-1]/eq[0])**(1/n_y)-1)*100

    csi300_cagr = calc_cagr(csi300_eq)
    real_cagr = calc_cagr(real_eq)
    alpha_cagr = real_cagr - csi300_cagr

    print(f"\n  CSI300-Timing CAGR: {csi300_cagr:+.2f}%")
    print(f"  Real Strategy CAGR: {real_cagr:+.2f}%")
    print(f"  Stock Selection Alpha: {alpha_cagr:+.2f}%")
    print(f"  Alpha / Total: {alpha_cagr/real_cagr*100:.0f}%" if real_cagr > 0 else "")

    # State breakdown
    st_names = {0: "BEAR", 1: "RANGE", 2: "BULL"}
    for si in [0, 1, 2]:
        sm = [d for d in detail if d["state"] == si]
        if sm:
            avg_csi = np.mean([d["csi300_ret"] for d in sm])
            avg_real = np.mean([d["real_ret"] for d in sm])
            avg_alpha = np.mean([d["alpha"] for d in sm])
            total_alpha = sum(d["alpha"] for d in sm)
            print(f"\n  {st_names[si]} ({len(sm)} months):")
            print(f"    Avg CSI300-Only Ret: {avg_csi:+.2f}%/mo")
            print(f"    Avg Real Strat Ret: {avg_real:+.2f}%/mo")
            print(f"    Avg Alpha:          {avg_alpha:+.2f}%/mo")
            print(f"    Total Alpha in {st_names[si]}: {total_alpha:+.1f}%")

    # Yearly decomposition
    print(f"\n  --- Yearly Decomposition ---")
    for yr in ["2022", "2023", "2024", "2025"]:
        ym = [d for d in detail if d["date"].startswith(yr)]
        if ym:
            total_csi = sum(d["csi300_ret"] for d in ym)
            total_real = sum(d["real_ret"] for d in ym)
            total_alpha = sum(d["alpha"] for d in ym)
            n_bear = sum(1 for d in ym if d["state"]==0)
            n_range = sum(1 for d in ym if d["state"]==1)
            n_bull = sum(1 for d in ym if d["state"]==2)
            print(f"  {yr}: CSI300-timing={total_csi:+.1f}%  Real={total_real:+.1f}%  Alpha={total_alpha:+.1f}%  "
                  f"States: BEAR={n_bear} RANGE={n_range} BULL={n_bull}")

    # Top alpha months
    print(f"\n  --- Top 5 Alpha Months ---")
    by_alpha = sorted(detail, key=lambda x: x["alpha"], reverse=True)
    for i, d in enumerate(by_alpha[:5], 1):
        sn = st_names[d["state"]]
        print(f"  #{i}: {d['date']} ({sn}) CSI300={d['csi300_ret']:+.1f}% Real={d['real_ret']:+.1f}% Alpha={d['alpha']:+.1f}%")

    # Worst alpha months
    print(f"\n  --- Bottom 5 Alpha Months ---")
    for i, d in enumerate(by_alpha[-5:], 1):
        sn = st_names[d["state"]]
        print(f"  #{i}: {d['date']} ({sn}) CSI300={d['csi300_ret']:+.1f}% Real={d['real_ret']:+.1f}% Alpha={d['alpha']:+.1f}%")

    # Total wealth decomposition
    total_csi_ret = (csi300_eq[-1]/csi300_eq[0] - 1) * 100
    total_real_ret = (real_eq[-1]/real_eq[0] - 1) * 100
    total_alpha = total_real_ret - total_csi_ret
    print(f"\n  --- Total Wealth Decomposition ---")
    print(f"  CSI300-Timing Total Return: {total_csi_ret:+.1f}%")
    print(f"  Real Strategy Total Return: {total_real_ret:+.1f}%")
    print(f"  Stock Selection Alpha:      {total_alpha:+.1f}%")
    if total_real_ret != 0:
        print(f"  Alpha Share:                {total_alpha/abs(total_real_ret)*100:.0f}%")
    print(f"  Timing Share:               {(total_real_ret-total_alpha)/abs(total_real_ret)*100:.0f}%" if total_real_ret != 0 else "")

print(f"\n{'='*90}")
print("DONE")
print(f"{'='*90}")

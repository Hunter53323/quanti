"""
Bull trap detection: test which indicators distinguish real breakouts from fake ones.
Uses 624 stocks for breadth analysis + CSI300 for price/volume signals.
"""
import sys, os; os.chdir(r"C:\study\AIWorkspace\quanti"); sys.path.insert(0,".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from quanti.data.storage import DataStorage

storage = DataStorage()

# ── Load CSI300 for market timing ──
def load_etf(code):
    raw = storage.load_bars(code)
    if not raw: return None
    dates = [r.trade_date for r in raw]
    closes = np.array([r.close for r in raw], dtype=np.float64)
    highs = np.array([r.high for r in raw], dtype=np.float64)
    lows = np.array([r.low for r in raw], dtype=np.float64)
    vols = np.array([r.volume for r in raw], dtype=np.float64)
    amounts = np.array([r.amount for r in raw], dtype=np.float64)
    return dates, closes, highs, lows, vols, amounts

etf = load_etf("510300")
if not etf: raise SystemExit("No CSI300 data")
dates, closes, highs, lows, vols, amounts = etf

def sma(arr, p):
    if len(arr)<p: return np.full(len(arr), np.nan)
    o=np.full(len(arr), np.nan); cs=np.cumsum(np.insert(arr,0,0.0))
    o[p-1:]=(cs[p:]-cs[:-p])/p; return o

# ── Load all stocks for breadth ──
all_files = sorted(storage.clean_dir.glob("*.parquet"))
stock_codes = [p.stem for p in all_files if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]
stock_closes = {}
for code in stock_codes:
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200: continue
    stock_closes[code] = (np.array([r.close for r in raw], dtype=np.float64), [r.trade_date for r in raw])

def stock_above_ma_pct(date_str, ma_period=20):
    """% of stocks above their MA on given date."""
    count, total = 0, 0
    for code, (c, d) in stock_closes.items():
        idx = None
        for i in range(len(d)-1,-1,-1):
            if d[i] <= date_str: idx = i+1; break
        if idx is None or idx < ma_period+1: continue
        total += 1
        ma = np.mean(c[idx-ma_period:idx])
        if c[idx-1] > ma: count += 1
    return count/total*100 if total > 0 else None

# ── Find all breakouts and label them ──
ma120 = sma(closes, 120)
above_ma = closes > ma120

# Find breakout dates: first day above 120MA after being below
breakouts = []
for i in range(121, len(above_ma)):
    if above_ma[i] and not above_ma[i-1]:
        breakouts.append(i)

print(f"Found {len(breakouts)} breakouts above 120MA (2005-2026)")
print(f"\nAnalyzing breakout quality indicators...\n")

# For each breakout, classify: TRUE (held >20 days above MA) vs FALSE (reversed <10 days)
results = []
for idx in breakouts:
    if idx + 60 >= len(closes): continue

    # Label: how long did it stay above MA?
    days_above = 0
    for j in range(idx, min(idx+60, len(above_ma))):
        if above_ma[j]: days_above += 1
        else: break

    if days_above >= 20:
        label = "TRUE"
    elif days_above <= 10:
        label = "FALSE"
    else:
        continue  # skip ambiguous

    # ── Compute trap indicators ──

    # 1. Volume expansion ratio (breakout day vs 20-day avg)
    vol_20avg = np.mean(vols[max(0,idx-21):idx-1]) if idx >= 22 else vols[idx]
    vol_ratio = vols[idx] / (vol_20avg + 1e-10)

    # 2. Volume trend: is volume expanding over last 5 days or contracting?
    vol_5avg = np.mean(vols[max(0,idx-5):idx])
    vol_20avg2 = np.mean(vols[max(0,idx-21):idx-6])
    vol_trend = vol_5avg / (vol_20avg2 + 1e-10)  # >1 expanding, <1 contracting

    # 3. Price momentum on breakout: single-day surge or steady rise?
    ret_1d = (closes[idx] / closes[idx-1] - 1) * 100
    ret_5d = (closes[idx] / closes[max(0,idx-5)] - 1) * 100

    # 4. Gap: did it gap up?
    gap_pct = (closes[idx] - highs[idx-1]) / highs[idx-1] * 100  # positive = gapped above yesterday's high

    # 5. Breadth: % stocks above 20MA on breakout day
    date_str = dates[idx]
    breadth = stock_above_ma_pct(date_str, 20)

    # 6. RSI
    gains = np.diff(closes[max(0,idx-14):idx+1])
    avg_gain = np.mean(gains[gains>0]) if len(gains[gains>0])>0 else 0
    avg_loss = abs(np.mean(gains[gains<0])) if len(gains[gains<0])>0 else 1e-10
    rsi = 100 - 100/(1 + avg_gain/avg_loss)

    # 7. ATR expansion
    tr = np.maximum(highs[max(0,idx-20):idx+1] - lows[max(0,idx-20):idx+1],
                    np.abs(highs[max(0,idx-20):idx+1] - np.roll(closes[max(0,idx-20):idx+1], 1)))
    tr[0] = highs[max(0,idx-20)] - lows[max(0,idx-20)]
    atr_5 = np.mean(tr[-5:]); atr_20 = np.mean(tr)
    atr_ratio = atr_5 / (atr_20 + 1e-10)

    # 8. Amount ratio (成交额 often more reliable than volume for ETFs)
    amt_ratio = amounts[idx] / (np.mean(amounts[max(0,idx-21):idx-1]) + 1e-10)

    # 9. High-low range: wider range = more uncertainty
    hl_range = (highs[idx] / lows[idx] - 1) * 100

    # 10. Close position within day's range
    close_pos = (closes[idx] - lows[idx]) / (highs[idx] - lows[idx] + 1e-10)  # 1=closed at high

    results.append({
        'date': dates[idx],
        'label': label,
        'days_above': days_above,
        'vol_ratio': vol_ratio,
        'vol_trend': vol_trend,
        'ret_1d': ret_1d,
        'ret_5d': ret_5d,
        'gap_pct': gap_pct,
        'breadth': breadth,
        'rsi': rsi,
        'atr_ratio': atr_ratio,
        'amt_ratio': amt_ratio,
        'hl_range': hl_range,
        'close_pos': close_pos,
    })

true_bos = [r for r in results if r['label']=='TRUE']
false_bos = [r for r in results if r['label']=='FALSE']

print(f"TRUE breakouts (held >20d):  {len(true_bos)}")
print(f"FALSE breakouts (lost <10d): {len(false_bos)}")
print()

# ── Compare indicators ──
indicators = ['vol_ratio','vol_trend','ret_1d','ret_5d','gap_pct','breadth','rsi','atr_ratio','amt_ratio','hl_range','close_pos']
ind_names = ['成交量比','成交量趋势','单日涨幅%','5日涨幅%','跳空%','广度%(>20MA)','RSI','ATR比','成交额比','振幅%','收盘位置']

print(f"{'指标':<16s} | {'真突破均值':>10s} | {'诱多均值':>10s} | {'差值':>8s} | {'区分度':>6s}")
print("-"*65)

best = []
for ind, name in zip(indicators, ind_names):
    tv = np.mean([r[ind] for r in true_bos if r[ind] is not None])
    fv = np.mean([r[ind] for r in false_bos if r[ind] is not None])
    diff = tv - fv
    # Discrimination: how many std devs apart?
    all_vals = [r[ind] for r in results if r[ind] is not None]
    std = np.std(all_vals) if len(all_vals)>1 else 1
    disc = abs(diff) / (std + 1e-10)
    best.append((name, disc, tv, fv, diff))
    print(f"{name:<16s} | {tv:>10.3f} | {fv:>10.3f} | {diff:>+8.3f} | {disc:>5.2f}s")

print(f"\n{'─'*65}")
print("区分度 = 均值差/标准差，越大越好。>0.5 = 有区分力，>1.0 = 强区分力")
print()

# Top discriminators
best.sort(key=lambda x: x[1], reverse=True)
print("Top 5 最具区分力的指标:")
for name, disc, tv, fv, _ in best[:5]:
    arrow = "↑" if tv > fv else "↓"
    print(f"  {name}: 区分度={disc:.2f}s  真突破{arrow}高于诱多 ({tv:.3f} vs {fv:.3f})")

# Print all breakouts by period
print(f"\n{'─'*65}")
print("逐年统计:")
for year in range(2015, 2026):
    y_true = [r for r in true_bos if r['date'].startswith(str(year))]
    y_false = [r for r in false_bos if r['date'].startswith(str(year))]
    total = len(y_true) + len(y_false)
    if total > 0:
        true_pct = len(y_true)/total*100
        print(f"  {year}: 真{len(y_true)} vs 诱多{len(y_false)} | 真比例={true_pct:.0f}% | 总共{total}次突破")

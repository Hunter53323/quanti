from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sys, os, time, json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from quanti.data.storage import DataStorage
from quanti.types import Bar, MarketData
from quanti.strategy.etf_rotation import ETFRotationStrategy
from quanti.indicators import macd, kdj

app = FastAPI(title="Quanti Signal API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ETF_POOL = ["510300","510500","159915","588360","563300","510880","518880","511880"]
ETF_NAMES = {
    "510300": "沪深300", "510500": "中证500", "159915": "创业板",
    "588360": "360互联+", "563300": "中证2000",
    "510880": "红利", "518880": "黄金", "511880": "货币",
}
storage = DataStorage()

# ── Cache ───────────────────────────────────────
# Data only changes once per day (after market close at ~16:00).
# Cache expires when latest data date changes.

_cache: dict = {}           # key -> result
_cache_date: str = ""       # latest data date used for current cache

CACHE_DIR = os.path.join(_ROOT, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_file(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def _load_cache(key: str):
    """Load cached result from disk if still fresh."""
    path = _cache_file(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Check if cache is still fresh (same trading date)
        if data.get("_date") == _get_latest_date():
            return data
    except Exception:
        pass
    return None

def _save_cache(key: str, data: dict):
    """Save result to disk cache with freshness date."""
    data["_date"] = _get_latest_date()
    with open(_cache_file(key), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def _get_latest_date() -> str:
    """Get latest trading date from 510300 data (fast - no scoring needed)."""
    raw = storage.load_bars("510300")
    return raw[-1].trade_date if raw else ""

def load_market_data():
    bars_dict = {}
    latest_date = None
    for sym in ETF_POOL:
        raw = storage.load_bars(sym)
        if raw and len(raw) >= 140:
            bars_dict[sym] = [
                Bar(symbol=sym, datetime=datetime.strptime(r.trade_date, "%Y%m%d"),
                    open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume)
                for r in raw
            ]
            last = raw[-1].trade_date
            if latest_date is None or last > latest_date:
                latest_date = last
    return MarketData(bars=bars_dict, index_bars={}, timestamp=datetime.now()), latest_date


@app.get("/api/signal")
def get_signal():
    # Try cache first
    cached = _load_cache("signal")
    if cached is not None:
        del cached["_date"]
        return cached

    md, latest_date = load_market_data()
    strategy = ETFRotationStrategy()
    signals = strategy.generate_signals(md)

    result = {
        "date": latest_date,
        "market": "BULL" if strategy._last_scores else "BEAR",
        "signals": [],
        "action": "空仓 - 没有ETF满足趋势条件",
    }

    if signals:
        total = 90000
        n = len(signals)
        per = total / n * 0.90
        items = []
        for sig in signals:
            bars = md.bars.get(sig.symbol, [])
            price = bars[-1].close if bars else 0
            qty = int(per / price / 100) * 100 if price > 0 else 0
            items.append({
                "symbol": sig.symbol,
                "name": ETF_NAMES.get(sig.symbol, ""),
                "price": round(price, 2),
                "score": round(sig.strength, 3),
                "shares": qty,
                "amount": round(qty * price, 0) if qty > 0 else 0,
                "reason": sig.reason,
            })
        result["signals"] = items
        result["action"] = f"买入 {n} 只ETF, 每只约 {per:,.0f} 元"

    _save_cache("signal", result)
    return result


@app.get("/api/decision")
def get_decision():
    cached = _load_cache("decision")
    if cached is not None:
        del cached["_date"]
        return cached

    md, latest_date = load_market_data()
    strategy = ETFRotationStrategy()
    strategy.generate_signals(md)

    rankings = []
    for sym in ETF_POOL:
        score = strategy._last_scores.get(sym, 0)
        bars = md.bars.get(sym, [])
        if bars:
            close = bars[-1].close
            ma120 = sum(b.close for b in bars[-120:]) / 120 if len(bars) >= 120 else 0
            rising = close > ma120
        else:
            close = 0; ma120 = 0; rising = False

        rankings.append({
            "symbol": sym, "name": ETF_NAMES.get(sym, ""),
            "price": round(close, 2), "score": round(score, 3),
            "ma120": round(ma120, 2), "rising_ma": rising, "above_ma": close > ma120,
        })
    rankings.sort(key=lambda x: x["score"], reverse=True)

    s2 = ETFRotationStrategy()
    sigs = s2.generate_signals(md)
    selected = [sig.symbol for sig in sigs]
    min_score = s2.min_score

    result = {"date": latest_date, "rankings": rankings, "selected": selected, "min_score": round(min_score, 2)}
    _save_cache("decision", result)
    return result


@app.get("/api/history")
def get_history(days: int = Query(default=90, le=365)):
    cache_key = f"history_{days}"
    cached = _load_cache(cache_key)
    if cached is not None:
        del cached["_date"]
        return cached

    md, latest_date = load_market_data()
    all_dates = sorted({b.datetime.strftime("%Y%m%d") for bars in md.bars.values() for b in bars})
    recent = all_dates[-days:]

    history = []
    for d in recent:
        truncated = {}
        for sym, bars in md.bars.items():
            truncated[sym] = [b for b in bars if b.datetime.strftime("%Y%m%d") <= d]
        md2 = MarketData(bars=truncated, index_bars={}, timestamp=datetime.strptime(d, "%Y%m%d"))
        strategy = ETFRotationStrategy()
        strategy.generate_signals(md2)
        if strategy._last_scores:
            top = sorted(strategy._last_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            history.append({
                "date": d,
                "top3": [{"symbol": sym, "score": round(s, 3)} for sym, s in top],
            })

    result = {"history": history[-30:]}
    _save_cache(cache_key, result)
    return result


def _ema(arr, period):
    """Exponential moving average."""
    if len(arr) < period:
        return [None] * len(arr)
    result = [None] * len(arr)
    result[period - 1] = sum(arr[:period]) / period
    multiplier = 2 / (period + 1)
    for i in range(period, len(arr)):
        result[i] = (arr[i] - result[i - 1]) * multiplier + result[i - 1]
    return result

def _sma(arr, period):
    """Simple moving average."""
    if len(arr) < period:
        return [None] * len(arr)
    result = [None] * (period - 1)
    window = sum(arr[:period])
    result.append(round(window / period, 2))
    for i in range(period, len(arr)):
        window = window - arr[i - period] + arr[i]
        result.append(round(window / period, 2))
    return result

# ---------------------------------------------------------------------------
# DEPRECATED: local _macd / _kdj kept as safety net. New code imports
# macd() and kdj() from quanti.indicators. Remove after verifying
# /api/klines/{symbol} returns identical JSON.
# ---------------------------------------------------------------------------

def _macd(closes, fast=12, slow=26, signal=9):
    """MACD: DIF, DEA, histogram."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    dif = [round(ema_fast[i] - ema_slow[i], 4) if ema_fast[i] is not None and ema_slow[i] is not None else None for i in range(len(closes))]
    dea = _ema([d if d is not None else 0 for d in dif], signal)
    macd_hist = [round((dif[i] - dea[i]) * 2, 4) if dif[i] is not None and dea[i] is not None else None for i in range(len(closes))]
    return dif, dea, macd_hist

def _kdj(highs, lows, closes, n=9):
    """KDJ indicator. Returns K, D, J arrays."""
    k, d, j = [50.0] * len(closes), [50.0] * len(closes), [50.0] * len(closes)
    for i in range(n - 1, len(closes)):
        hh = max(highs[i - n + 1:i + 1])
        ll = min(lows[i - n + 1:i + 1])
        rsv = (closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50
        k[i] = 2 / 3 * k[i - 1] + 1 / 3 * rsv
        d[i] = 2 / 3 * d[i - 1] + 1 / 3 * k[i]
        j[i] = 3 * k[i] - 2 * d[i]
    for i in range(n - 1):
        k[i] = d[i] = j[i] = None
    return k, d, j

def _resample_weekly(daily_bars):
    """Resample daily bars to weekly. Each bar: [date, open, close, low, high, volume]."""
    if len(daily_bars) < 5:
        return []
    weekly = []
    week_bars = []
    for b in daily_bars:
        date_str = b[0]  # YYYYMMDD
        if not week_bars or date_str[4:6] != week_bars[-1][0][4:6]:
            if week_bars:
                week_bars = [x for x in week_bars if len(x[0]) == 8]  # sanitize
            if not week_bars or (date_str[4:6] != week_bars[0][0][4:6] and len(date_str) >= 6):
                if week_bars:
                    weekly.append([
                        week_bars[-1][0],  # last date of week
                        week_bars[0][1],    # first open
                        week_bars[-1][2],   # last close
                        min(b[3] for b in week_bars),  # week low
                        max(b[4] for b in week_bars),  # week high
                        sum(b[5] for b in week_bars),  # week volume
                    ])
                week_bars = []
        week_bars.append(b)
    if week_bars:
        weekly.append([
            week_bars[-1][0], week_bars[0][1], week_bars[-1][2],
            min(b[3] for b in week_bars), max(b[4] for b in week_bars), sum(b[5] for b in week_bars),
        ])
    return weekly

def _resample_monthly(daily_bars):
    """Resample daily bars to monthly."""
    if len(daily_bars) < 20:
        return []
    monthly = []
    month_bars = []
    for b in daily_bars:
        if not month_bars or b[0][4:6] != month_bars[-1][0][4:6]:
            if month_bars:
                monthly.append([
                    month_bars[-1][0], month_bars[0][1], month_bars[-1][2],
                    min(x[3] for x in month_bars), max(x[4] for x in month_bars), sum(x[5] for x in month_bars),
                ])
                month_bars = []
        month_bars.append(b)
    if month_bars:
        monthly.append([
            month_bars[-1][0], month_bars[0][1], month_bars[-1][2],
            min(x[3] for x in month_bars), max(x[4] for x in month_bars), sum(x[5] for x in month_bars),
        ])
    return monthly

def _arr_to_list(arr):
    """Convert ndarray to Python list, replacing NaN with None."""
    if arr is None:
        return []
    return [None if np.isnan(x) else round(float(x), 4) for x in arr]


def _compute_indicators(raw_bars):
    """From raw ETFDailyBar list, compute raw kline arrays and all indicators."""
    closes = [r.close for r in raw_bars]
    highs = [r.high for r in raw_bars]
    lows = [r.low for r in raw_bars]
    volumes = [int(r.volume) for r in raw_bars]
    dates = [r.trade_date for r in raw_bars]
    opens = [r.open for r in raw_bars]

    # Klines raw: [date, open, close, low, high, volume]
    klines_raw = [[dates[i], opens[i], closes[i], lows[i], highs[i], volumes[i]] for i in range(len(dates))]

    # MA
    ma5 = _sma(closes, 5)
    ma10 = _sma(closes, 10)
    ma20 = _sma(closes, 20)
    ma120 = _sma(closes, 120)
    ma250 = _sma(closes, 250)

    # MACD -- imported from quanti.indicators
    macd_result = macd(np.array(closes, dtype=np.float64))
    if macd_result is not None:
        dif, dea, hist = macd_result
        macd_dif = _arr_to_list(dif)
        macd_dea = _arr_to_list(dea)
        macd_hist = _arr_to_list(hist)
    else:
        n = len(closes)
        macd_dif = [None] * n
        macd_dea = [None] * n
        macd_hist = [None] * n

    # KDJ -- imported from quanti.indicators
    kdj_result = kdj(np.array(highs, dtype=np.float64),
                     np.array(lows, dtype=np.float64),
                     np.array(closes, dtype=np.float64))
    if kdj_result is not None:
        k_arr, d_arr, j_arr = kdj_result
        kdj_k = _arr_to_list(k_arr)
        kdj_d = _arr_to_list(d_arr)
        kdj_j = _arr_to_list(j_arr)
    else:
        n = len(closes)
        kdj_k = [None] * n
        kdj_d = [None] * n
        kdj_j = [None] * n

    return {
        "dates": dates, "klines": klines_raw,
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma120": ma120, "ma250": ma250,
        "macd_dif": macd_dif, "macd_dea": macd_dea, "macd_hist": macd_hist,
        "kdj_k": kdj_k, "kdj_d": kdj_d, "kdj_j": kdj_j,
    }

@app.get("/api/klines/{symbol}")
def get_klines(symbol: str, period: str = "daily", days: int = Query(default=120, le=500)):
    """Return OHLCV + all indicators for symbol. period: daily/weekly/monthly."""
    raw = storage.load_bars(symbol)
    if not raw:
        return {"symbol": symbol, "period": period, "klines": []}

    # Compute indicators from all raw data (then slice to days)
    full = _compute_indicators(raw)

    # Resample to period
    if period == "weekly":
        full["klines"] = _resample_weekly(full["klines"])
    elif period == "monthly":
        full["klines"] = _resample_monthly(full["klines"])

    # Slice to requested days
    k = full["klines"][-days:]
    if not k:
        return {"symbol": symbol, "period": period, "klines": []}

    # Slice indicator arrays to same length
    start_idx = len(full["klines"]) - len(k)
    end_idx = len(full["klines"])

    def slice_arr(arr):
        if arr is None: return []
        s = arr[start_idx:end_idx] if len(arr) >= end_idx else arr[-len(k):]
        return [round(x, 4) if x is not None else None for x in s]

    return {
        "symbol": symbol,
        "name": ETF_NAMES.get(symbol, ""),
        "period": period,
        "dates": [x[0] for x in k],
        "klines": [[x[1], x[2], x[3], x[4], x[5]] for x in k],  # [open,close,low,high,vol]
        "ma5": slice_arr(full["ma5"]), "ma10": slice_arr(full["ma10"]),
        "ma20": slice_arr(full["ma20"]), "ma120": slice_arr(full["ma120"]),
        "ma250": slice_arr(full["ma250"]),
        "macd_dif": slice_arr(full["macd_dif"]), "macd_dea": slice_arr(full["macd_dea"]),
        "macd_hist": slice_arr(full["macd_hist"]),
        "kdj_k": slice_arr(full["kdj_k"]), "kdj_d": slice_arr(full["kdj_d"]),
        "kdj_j": slice_arr(full["kdj_j"]),
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "cache_date": _get_latest_date()}

@app.post("/api/cache/clear")
def clear_cache():
    """Clear all cached results (call after data update)."""
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(CACHE_DIR, f))
    return {"status": "cleared", "files": len(os.listdir(CACHE_DIR))}

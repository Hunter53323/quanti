"""
Live signal generation for Trend-First Momentum Strategy.
Run daily to see what the strategy would buy/sell today.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime

from quanti.data.storage import DataStorage
from quanti.strategy.stock_momentum import StockMomentumStrategy
from quanti.types import Bar, MarketData

# ── Load latest data ──
storage = DataStorage()
all_files = sorted(storage.clean_dir.glob("*.parquet"))
stock_codes = [p.stem for p in all_files
               if len(p.stem) == 6 and not p.stem.startswith(("51","58","15","56"))]

print(f"Loading {len(stock_codes)} stocks...", flush=True)

# Build MarketData with latest bars for all stocks
# Find the latest common date
latest_date = None
bars_dict = {}
for code in stock_codes:
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200:
        continue
    bar_objs = [Bar(
        symbol=code,
        datetime=datetime.strptime(r.trade_date, "%Y%m%d"),
        open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume,
    ) for r in raw]
    bars_dict[code] = bar_objs
    last = raw[-1].trade_date
    if latest_date is None or last > latest_date:
        latest_date = last

print(f"Loaded {len(bars_dict)} stocks, latest date: {latest_date}", flush=True)

# Also load CSI300 ETF for market trend check
for etf in ["510300", "159919"]:
    raw_etf = storage.load_bars(etf)
    if raw_etf:
        bars_dict[etf] = [Bar(
            symbol=etf,
            datetime=datetime.strptime(r.trade_date, "%Y%m%d"),
            open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume,
        ) for r in raw_etf]
        print(f"  Market proxy: {etf} ({len(bars_dict[etf])} bars)", flush=True)
        break

# ── Run strategy ──
md = MarketData(bars=bars_dict, index_bars={}, timestamp=datetime.now())

# Test two configurations
configs = [
    ("Conservative (DD20)", dict(top_n=5, stop_loss_pct=-10.0, min_trend_score=3,
                                  market_trend_required=True, dd_exit_pct=20.0)),
    ("Aggressive (DD25)", dict(top_n=5, stop_loss_pct=-10.0, min_trend_score=3,
                                market_trend_required=True, dd_exit_pct=25.0)),
]

for label, params in configs:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    s = StockMomentumStrategy(**params)
    signals = s.generate_signals(md)

    if not signals:
        # Check WHY - market trending or not?
        mkt_ok = s._is_market_trending(md)
        if not mkt_ok:
            print("  MARKET: BEAR (CSI300 below 120MA) - no positions allowed")
        else:
            print("  MARKET: BULL but no trending stocks found")
        continue

    print(f"  Market: BULL | Selected {len(signals)} stocks:\n")
    # Show with scores
    for sig in signals:
        bar = bars_dict[sig.symbol][-1]
        print(f"  {sig.symbol}  |  close={bar.close:>8.2f}  |  score={sig.strength:.3f}  |  {sig.reason}")

    # Show allocation — only stocks under price cap
    per_stock = 90000 / len(signals) * 0.92
    min_lot_cost = 100  # A-share min 100 shares
    print(f"\n  Suggested allocation: ~{per_stock:,.0f} RMB per stock")
    print(f"  (Min lot: 100 shares. Stocks needing >{per_stock:,.0f} for 1 lot are skipped)")
    actionable = []
    for sig in signals:
        bar = bars_dict[sig.symbol][-1]
        price = bar.close
        lot_cost = 100 * price  # A-share: 100 shares/lot
        if lot_cost <= per_stock:
            qty = int(per_stock / price / 100) * 100
            cost = qty * price
            print(f"    BUY {qty} shares ({qty//100} lots) {sig.symbol} @ {price:.2f} = {cost:,.0f} RMB")
            actionable.append((sig.symbol, price, qty))
        else:
            # Try with smaller allocation: just 1 lot
            if lot_cost <= 90000 * 0.2:
                print(f"    BUY 100 shares (1 lot) {sig.symbol} @ {price:.2f} = {lot_cost:,.0f} RMB (min lot only)")
                actionable.append((sig.symbol, price, 100))
            else:
                print(f"    SKIP {sig.symbol} @ {price:.2f} — 1 lot = {lot_cost:,.0f} > 20% capital")

    # Top 20 trend scores (for reference)
    if s._last_scores:
        print("\n  Top 20 by trend strength:")
        for i, (sym, score) in enumerate(sorted(s._last_scores.items(), key=lambda x: x[1], reverse=True)[:20]):
            mark = " <<<" if any(sig.symbol == sym for sig in signals) else ""
            close = bars_dict[sym][-1].close
            print(f"    {i+1:2d}. {sym}  score={score:.1f}  close={close:.2f}{mark}")

print(f"\n{'='*70}")
print(f"  Run at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*70}")

"""
Standalone Paper Trading Engine — DelayedConfirmStrategy + Bond/Gold
=====================================================================
Self-contained: no journal, no breaker, no alerter, no recovery.
All infrastructure inlined for transparency.

Usage:
  python quanti/main_paper_delayed.py

Flow (every 60 seconds):
  1. Load latest data from data/clean/*.parquet via DataStorage
  2. Build MarketData for DelayedConfirmStrategy
  3. Generate signals → size positions → risk check → simulate fills
  4. Persist checkpoint as JSON to data/checkpoint_delayed.json
  5. Sleep 60s, repeat. Ctrl+C to stop gracefully.

Recovery: reads data/checkpoint_delayed.json on startup.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Project root ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from quanti.config import settings
from quanti.data.storage import DataStorage

try:
    from quanti.data.ingestion import run_daily_ingest
except ImportError:
    run_daily_ingest = None
from quanti.strategy.delayed_confirm import DelayedConfirmStrategy
from quanti.types import Bar, MarketData, Order, OrderSide, Portfolio, Position, Signal

# ═══════════════════════════════════════════════════════════════
# Inline infrastructure
# ═══════════════════════════════════════════════════════════════

# ── Checkpoint persistence ──
CHECKPOINT_PATH = Path(_PROJECT_ROOT) / "data" / "checkpoint_delayed.json"


def save_checkpoint(positions: dict, cash: float):
    """Save portfolio state as JSON for crash recovery."""
    pos_dict = {}
    for sym, pos in positions.items():
        if pos.quantity > 0:
            pos_dict[sym] = {"quantity": pos.quantity, "avg_cost": pos.avg_cost}
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "cash": cash,
        "positions": pos_dict,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def load_checkpoint() -> tuple[float, dict[str, Position]]:
    """Recover cash and positions from checkpoint. Returns (cash, {symbol: Position})."""
    if not CHECKPOINT_PATH.exists():
        return settings.TRADING_CAPITAL, {}
    try:
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            snap = json.load(f)
        cash = float(snap.get("cash", settings.TRADING_CAPITAL))
        positions = {}
        for sym, data in snap.get("positions", {}).items():
            positions[sym] = Position(
                symbol=sym,
                quantity=data["quantity"],
                avg_cost=data["avg_cost"],
                current_price=0.0,
            )
        return cash, positions
    except (json.JSONDecodeError, KeyError):
        return settings.TRADING_CAPITAL, {}


# ── Bare-minimum risk checker (pass-through) ──
@dataclass
class InlineRiskChecker:
    """Minimal risk checker: approves all orders, just checks capital."""
    max_position_pct: float = 0.25  # max 25% of capital per position

    def check_all(
        self, orders: list[Order], portfolio: Portfolio
    ) -> list[Order]:
        """Filter orders that exceed capital limits."""
        approved = []
        for o in orders:
            if o.side == OrderSide.BUY:
                cost = o.quantity * (o.price or 0)
                if cost > portfolio.cash * self.max_position_pct:
                    print(f"  [RISK] SKIP {o.symbol}: cost={cost:.0f} > {portfolio.cash * self.max_position_pct:.0f}")
                    continue
            approved.append(o)
        return approved


# ── Signal handler ──
_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n[SHUTDOWN] Signal received, stopping gracefully...")
    _running = False


# ═══════════════════════════════════════════════════════════════
# Main trading loop
# ═══════════════════════════════════════════════════════════════

def main():
    global _running

    print("=" * 60)
    print("PAPER TRADING — DelayedConfirmStrategy + Bond/Gold")
    print("=" * 60)

    # ── Strategy ──
    strategy = DelayedConfirmStrategy(
        confirm_days=5, cooldown_days=40, top_n=5,
        stop_loss_pct=-10.0, min_trend_score=3, dd_exit_pct=15.0,
        decay_schedule="A43", use_sharp_exit=True, sharp_threshold=-0.03,
    )
    print(f"Strategy: {strategy.name}")
    print(f"  confirm={strategy.confirm_days}d  cooldown={strategy.cooldown_days}d")
    print(f"  decay={strategy.decay_schedule}  sharp={strategy.use_sharp_exit} "
          f"(threshold={strategy.sharp_threshold*100:+.0f}%)")
    print(f"  defensive: {strategy.defensive_bond_pct:.0%} {strategy.bond_etf} / "
          f"{strategy.defensive_gold_pct:.0%} {strategy.gold_etf}")

    risk_checker = InlineRiskChecker()

    # ── Universe ──
    storage = DataStorage()
    all_files = sorted(storage.clean_dir.glob("*.parquet"))
    stock_codes = [p.stem for p in all_files if len(p.stem) == 6
                   and not p.stem.startswith(("51", "58", "15", "56"))]
    strategy._stock_universe = set(stock_codes)
    symbols = stock_codes + [strategy.bond_etf, strategy.gold_etf, "510300"]
    print(f"Universe: {len(symbols)} symbols ({len(stock_codes)} stocks + 3 ETFs)")

    # ── Recover from checkpoint ──
    cash, positions = load_checkpoint()
    print(f"Capital: {cash:,.0f} RMB  |  Positions: {len(positions)}")
    for sym, pos in sorted(positions.items()):
        print(f"  {sym}: {pos.quantity} shares @ {pos.avg_cost:.2f}")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Initial data load ──
    print("\nLoading data...")
    if run_daily_ingest:
        ok, errors = run_daily_ingest()
        for e in errors:
            print(f"  [WARN] Data: {e}")
    else:
        print("  [SKIP] run_daily_ingest not available (using existing data files)")

    bars_dict = {}
    for sym in symbols:
        bars = storage.load_bars(sym)
        if bars:
            bars_dict[sym] = bars
    print(f"Loaded {len(bars_dict)}/{len(symbols)} symbols")

    for sym in ("510300", strategy.bond_etf, strategy.gold_etf):
        if sym in bars_dict:
            b = bars_dict[sym]
            print(f"  {sym}: {len(b)} bars, last date = {b[-1].trade_date}")

    # ── Build CSI300 index bars once ──
    csi300_bars = bars_dict.get("510300", [])
    index_bars = {}
    if csi300_bars:
        index_bars["510300"] = [
            Bar(symbol=rb.symbol,
                datetime=datetime.strptime(rb.trade_date, "%Y%m%d"),
                open=rb.open, high=rb.high, low=rb.low,
                close=rb.close, volume=rb.volume)
            for rb in csi300_bars
        ]

    # ── Save initial checkpoint ──
    save_checkpoint(positions, cash)

    print("\nTrading loop starting. Ctrl+C to stop gracefully.\n")
    loop_count = 0
    total_pnl = 0.0
    total_trades = 0
    commission_rate = getattr(settings, "COMMISSION_RATE", 0.00025)

    try:
        while _running:
            loop_count += 1

            # ── Build MarketData ──
            common = {}
            for sym, bars in bars_dict.items():
                common[sym] = [
                    Bar(symbol=rb.symbol,
                        datetime=datetime.strptime(rb.trade_date, "%Y%m%d"),
                        open=rb.open, high=rb.high, low=rb.low,
                        close=rb.close, volume=rb.volume)
                    for rb in bars
                ]

            md = MarketData(
                bars=common,
                index_bars=index_bars,
                timestamp=datetime.now(),
            )

            # ── Update current prices ──
            for sym, bars in bars_dict.items():
                if bars and sym in positions:
                    positions[sym].current_price = bars[-1].close

            # ── Portfolio snapshot ──
            total_mv = sum(p.quantity * p.current_price for p in positions.values())
            pf = Portfolio(
                positions={s: p for s, p in positions.items() if p.quantity > 0},
                cash=cash,
                total_capital=cash + total_mv,
                settled_cash=cash,
                timestamp=datetime.now(),
            )

            # ── Strategy cycle ──
            signals = strategy.generate_signals(md)
            if signals:
                sig_str = ", ".join(f"{s.symbol} {s.side.value}" for s in signals[:8])
                if len(signals) > 8:
                    sig_str += f" (+{len(signals)-8} more)"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SIGNALS ({len(signals)}): {sig_str}")

            orders = strategy.size_positions(signals, cash, pf, market_data=md)

            # ── Risk check ──
            # Strategy-internal risk check first
            orders = strategy.risk_check(orders, pf, market_data=md, risk_checker=risk_checker)
            # Then inline capital filter
            orders = risk_checker.check_all(orders, pf)

            # ── Decay / Sharp info ──
            di = strategy.get_decay_info()
            si = strategy.get_sharp_info()
            if di["months_in_cycle"] > 0:
                print(f"  [INFO] Decay: {di['schedule']} month={di['months_in_cycle']} "
                      f"multiplier={di['current_multiplier']:.0%}")
            if si.get("sharp_fired_recent"):
                print(f"  [ALERT] SHARP EXIT FIRED! threshold={si['sharp_threshold']*100:+.0f}%")

            # ── Execute orders ──
            daily_pnl = 0.0
            for o in orders:
                bar_list = bars_dict.get(o.symbol, [])
                if not bar_list:
                    print(f"  [SKIP] {o.symbol} {o.side.value}: no price data")
                    continue

                bar = bar_list[-1]
                fill_price = o.price if o.price and o.price > 0 else bar.close
                fill_notional = o.quantity * fill_price
                comm = fill_notional * commission_rate

                if o.side == OrderSide.BUY:
                    total_cost = fill_notional + comm
                    if total_cost <= cash:
                        cash -= total_cost
                        if o.symbol in positions:
                            p = positions[o.symbol]
                            new_qty = p.quantity + o.quantity
                            new_avg = (p.avg_cost * p.quantity + fill_price * o.quantity) / new_qty
                            positions[o.symbol] = Position(
                                symbol=o.symbol, quantity=new_qty,
                                avg_cost=new_avg, current_price=fill_price,
                            )
                        else:
                            positions[o.symbol] = Position(
                                symbol=o.symbol, quantity=o.quantity,
                                avg_cost=fill_price, current_price=fill_price,
                            )
                        print(f"  [BUY]  {o.symbol} x{o.quantity} @{fill_price:.2f}  "
                              f"cost={total_cost:,.0f}  cash={cash:,.0f}")
                        total_trades += 1
                    else:
                        print(f"  [SKIP] {o.symbol} BUY: need {total_cost:,.0f} > cash={cash:,.0f}")

                elif o.side == OrderSide.SELL:
                    pos = positions.get(o.symbol)
                    if pos and pos.quantity >= o.quantity:
                        proceeds = fill_notional - comm
                        cash += proceeds
                        pos.quantity -= o.quantity
                        realized_pnl = (fill_price - pos.avg_cost) * o.quantity
                        daily_pnl += realized_pnl
                        total_pnl += realized_pnl
                        if pos.quantity == 0:
                            del positions[o.symbol]
                        print(f"  [SELL] {o.symbol} x{o.quantity} @{fill_price:.2f}  "
                              f"proceeds={proceeds:,.0f}  PnL={realized_pnl:+,.0f}  cash={cash:,.0f}")
                        total_trades += 1
                    else:
                        qty = pos.quantity if pos else 0
                        print(f"  [SKIP] {o.symbol} SELL: have {qty} < need {o.quantity}")

            # ── Portfolio summary ──
            stock_mv = sum(
                p.quantity * p.current_price
                for s, p in positions.items()
                if s not in (strategy.bond_etf, strategy.gold_etf)
            )
            bond_mv = sum(
                p.quantity * p.current_price
                for s, p in positions.items()
                if s == strategy.bond_etf
            )
            gold_mv = sum(
                p.quantity * p.current_price
                for s, p in positions.items()
                if s == strategy.gold_etf
            )
            total_eq = cash + stock_mv + bond_mv + gold_mv
            print(f"  [PORT] cash={cash:,.0f}  stocks={stock_mv:,.0f}  "
                  f"bond={bond_mv:,.0f}  gold={gold_mv:,.0f}  "
                  f"total={total_eq:,.0f}  PnL={daily_pnl:+,.0f}  trades={total_trades}")

            # ── Checkpoint ──
            save_checkpoint(positions, cash)

            # ── Wait ──
            time.sleep(60)

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[FATAL] {exc}")
    finally:
        save_checkpoint(positions, cash)
        print(f"\nStopped.  Loops: {loop_count}  Trades: {total_trades}  "
              f"Total PnL: {total_pnl:+,.0f}  Cash: {cash:,.0f}")
        stock_mv = sum(p.quantity * p.current_price for s, p in positions.items()
                       if s not in (strategy.bond_etf, strategy.gold_etf))
        bond_mv = sum(p.quantity * p.current_price for s, p in positions.items()
                      if s == strategy.bond_etf)
        gold_mv = sum(p.quantity * p.current_price for s, p in positions.items()
                      if s == strategy.gold_etf)
        total = cash + stock_mv + bond_mv + gold_mv
        print(f"  Final NAV: {total:,.0f}  (cash={cash:,.0f} stocks={stock_mv:,.0f} "
              f"bond={bond_mv:,.0f} gold={gold_mv:,.0f})")
        print(f"  Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

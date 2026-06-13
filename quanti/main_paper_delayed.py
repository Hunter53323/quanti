"""
Paper Trading for DelayedConfirmStrategy (Bond + Gold).

Usage: python quanti/main_paper_delayed.py
"""
import signal, time
from datetime import datetime

from quanti.config import settings
from quanti.data.ingestion import run_daily_ingest
from quanti.data.storage import DataStorage
from quanti.execution.circuit_breaker import BreakerManager
from quanti.execution.risk import RiskChecker
from quanti.monitor.alerts import AlertLevel, get_alerter
from quanti.monitor.logger import get_logger, setup_logger
from quanti.monitor.metrics import get_metrics
from quanti.state.journal import Journal
from quanti.state.recovery import build_checkpoint_snapshot, recover_portfolio
from quanti.strategy.delayed_confirm import DelayedConfirmStrategy
from quanti.types import Bar, MarketData, Portfolio, Position

logger = get_logger("paper")
running = True

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received")
    running = False


def main():
    global running
    setup_logger("paper_trading_delayed")
    logger.info("=" * 60)
    logger.info("PAPER TRADING -- DelayedConfirmStrategy + Bond/Gold")
    logger.info("=" * 60)

    journal = Journal()
    storage = DataStorage()
    alerter = get_alerter()
    metrics = get_metrics()
    breaker_mgr = BreakerManager()

    strategy = DelayedConfirmStrategy(
        confirm_days=5, cooldown_days=40, top_n=5,
        stop_loss_pct=-10.0, min_trend_score=3, dd_exit_pct=15.0,
        decay_schedule="A43", use_sharp_exit=True, sharp_threshold=-0.03,
    )
    logger.info(f"Strategy: {strategy.name}")
    logger.info(f"  confirm={strategy.confirm_days}d cooldown={strategy.cooldown_days}d")
    logger.info(f"  decay={strategy.decay_schedule} sharp={strategy.use_sharp_exit} threshold={strategy.sharp_threshold}")
    logger.info(f"  defensive: {strategy.defensive_bond_pct:.0%} {strategy.bond_etf} / {strategy.defensive_gold_pct:.0%} {strategy.gold_etf}")
    risk_checker = RiskChecker()

    all_files = sorted(storage.clean_dir.glob("*.parquet"))
    stock_codes = [p.stem for p in all_files if len(p.stem) == 6
                   and not p.stem.startswith(("51", "58", "15", "56"))]
    symbols = stock_codes + [strategy.bond_etf, strategy.gold_etf, "510300"]
    strategy._stock_universe = set(stock_codes)
    logger.info(f"Universe: {len(symbols)} symbols ({len(stock_codes)} stocks)")

    recovered = recover_portfolio(journal)
    cash = recovered["cash"]
    if cash is None:
        cash = settings.TRADING_CAPITAL
    positions = {}
    for s, q in recovered["positions"].items():
        if s in strategy._stock_universe or s in (strategy.bond_etf, strategy.gold_etf):
            positions[s] = Position(symbol=s, quantity=q["quantity"], avg_cost=q["avg_cost"], current_price=0)
        else:
            logger.info(f"  Dropping stale checkpoint: {s}")
    logger.info(f"Capital: {cash:.0f} RMB, Positions: {len(positions)}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if recovered["checkpoint_ts"] is None:
        pos_dict = {s: {"quantity": p.quantity, "avg_cost": p.avg_cost}
                    for s, p in positions.items() if p.quantity > 0}
        snapshot = build_checkpoint_snapshot(pos_dict, cash, [])
        snapshot["breaker_state"] = breaker_mgr.save_state()
        journal.save_checkpoint(snapshot)

    logger.info("Fetching data...")
    ok, errors = run_daily_ingest()
    for e in errors: logger.warning(f"Data: {e}")

    bars_dict = {}
    for sym in symbols:
        bars = storage.load_bars(sym)
        if bars: bars_dict[sym] = bars
    logger.info(f"Loaded {len(bars_dict)}/{len(symbols)} symbols")
    for sym in ("510300", strategy.bond_etf, strategy.gold_etf):
        if sym in bars_dict:
            b = bars_dict[sym]; logger.info(f"  {sym}: {len(b)} bars, last={b[-1].trade_date}")

    csi300_bars = bars_dict.get("510300", [])
    index_bars = {}
    if csi300_bars:
        index_bars["510300"] = [Bar(symbol=b.symbol, datetime=datetime.strptime(b.trade_date, "%Y%m%d"),
                                     open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
                                for b in csi300_bars]

    logger.info("Trading loop starting. Ctrl+C to stop.")

    try:
        while running:
            if not breaker_mgr.check_all():
                logger.warning(f"BREAKER: {breaker_mgr.status()}")
                time.sleep(30); continue

            common = {}
            for sym, bars in bars_dict.items():
                common[sym] = [Bar(symbol=b.symbol, datetime=datetime.strptime(b.trade_date, "%Y%m%d"),
                                   open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
                               for b in bars]
            ib = {"510300": index_bars["510300"]} if index_bars else {}
            md = MarketData(bars=common, index_bars=ib, timestamp=datetime.now())

            for sym, bars in bars_dict.items():
                if bars and sym in positions: positions[sym].current_price = bars[-1].close

            total_mv = sum(p.quantity * p.current_price for p in positions.values())
            pf = Portfolio(positions={s: p for s, p in positions.items() if p.quantity > 0},
                           cash=cash, total_capital=cash + total_mv,
                           settled_cash=cash, timestamp=datetime.now())

            strategy._csi300_bars_loaded = False  # rebuild state machine each tick to catch new data
            signals = strategy.generate_signals(md)
            if signals:
                logger.info(f"SIGNALS: {[(s.symbol, s.side.value) for s in signals]}")

            orders = strategy.size_positions(signals, cash, pf, market_data=md)
            orders = strategy.risk_check(orders, pf, market_data=md, risk_checker=risk_checker)

            di = strategy.get_decay_info()
            si = strategy.get_sharp_info()
            if di["months_in_cycle"] > 0:
                logger.info(f"  Decay: {di['schedule']} mo={di['months_in_cycle']} mult={di['current_multiplier']:.0%}")
            if si["sharp_fired_recent"]:
                logger.warning(f"  Sharp exit FIRED! threshold={si['sharp_threshold']*100:.0f}%")

            daily_pnl = 0.0
            for o in orders:
                bar = bars_dict.get(o.symbol, [])[-1] if bars_dict.get(o.symbol) else None
                if not bar: continue
                fill_price = o.price or bar.close
                fill_cost = o.quantity * fill_price
                comm = fill_cost * getattr(settings, "COMMISSION_RATE", 0.00025)

                if o.side.value == "buy":
                    total_cost = fill_cost + comm
                    if total_cost <= cash:
                        cash -= total_cost
                        if o.symbol in positions:
                            p = positions[o.symbol]
                            nq = p.quantity + o.quantity
                            na = (p.avg_cost * p.quantity + fill_price * o.quantity) / nq
                            positions[o.symbol] = Position(symbol=o.symbol, quantity=nq, avg_cost=na, current_price=fill_price)
                        else:
                            positions[o.symbol] = Position(symbol=o.symbol, quantity=o.quantity, avg_cost=fill_price, current_price=fill_price)
                        logger.info(f"  FILL BUY  {o.symbol} x{o.quantity} @{fill_price:.2f} cost={total_cost:.0f}")
                    else:
                        logger.warning(f"  SKIP BUY  {o.symbol}: need {total_cost:.0f}, have {cash:.0f}")
                else:
                    pos = positions.get(o.symbol)
                    if pos and pos.quantity >= o.quantity:
                        proceeds = fill_cost - comm
                        cash += proceeds
                        pos.quantity -= o.quantity
                        if pos.quantity == 0: del positions[o.symbol]
                        pnl = (fill_price - pos.avg_cost) * o.quantity
                        daily_pnl += pnl
                        logger.info(f"  FILL SELL {o.symbol} x{o.quantity} @{fill_price:.2f} PnL={pnl:+.0f}")
                    else:
                        logger.warning(f"  SKIP SELL {o.symbol}: insufficient quantity")
                metrics.inc_counter("paper.orders")
                breaker_mgr.general.record_execution_success()

            breaker_mgr.general.update_pnl(daily_pnl)

            pos_dict = {s: {"quantity": p.quantity, "avg_cost": p.avg_cost}
                        for s, p in positions.items() if p.quantity > 0}
            snapshot = build_checkpoint_snapshot(pos_dict, cash, [])
            snapshot["breaker_state"] = breaker_mgr.save_state()
            journal.save_checkpoint(snapshot)

            time.sleep(60)

    except KeyboardInterrupt: pass
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        alerter.send(AlertLevel.CRITICAL, "Paper Trading Crashed", str(e))
    finally:
        pos_dict = {s: {"quantity": p.quantity, "avg_cost": p.avg_cost}
                    for s, p in positions.items() if p.quantity > 0}
        snapshot = build_checkpoint_snapshot(pos_dict, cash, [])
        snapshot["breaker_state"] = breaker_mgr.save_state()
        journal.save_checkpoint(snapshot)
        logger.info(f"Stopped. Metrics: {metrics.snapshot()}")


if __name__ == "__main__":
    main()

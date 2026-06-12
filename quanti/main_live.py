"""Live Trading Engine (Phase 5+).
Managed as NSSM Windows Service in production.
Usage: python -m quanti.main_live
"""
import signal
import time
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
from quanti.strategy.etf_trend import ETFTrendStrategy
from quanti.types import Bar, MarketData, Portfolio, Position

logger = get_logger("live")
running = True


def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal. Saving checkpoint...")
    running = False


def main():
    global running
    setup_logger("live_trading")
    logger.info("=" * 60)
    logger.info("LIVE TRADING ENGINE")
    logger.info(f"Capital: {settings.TOTAL_CAPITAL} RMB")
    logger.info(f"Universe: {settings.ETF_UNIVERSE} | Max positions: {settings.MAX_POSITIONS}")
    logger.info("=" * 60)

    journal = Journal()
    storage = DataStorage()
    alerter = get_alerter()
    metrics = get_metrics()
    breaker_mgr = BreakerManager()
    strategy = ETFTrendStrategy()
    risk_checker = RiskChecker()

    # Crash recovery
    logger.info("Running crash recovery...")
    recovered = recover_portfolio(journal)
    cash = recovered["cash"]
    if cash is None:
        logger.error("No checkpoint found - cannot determine capital. Aborting.")
        return
    positions = {}
    for s, q in recovered["positions"].items():
        positions[s] = Position(symbol=s, quantity=q["quantity"], avg_cost=q["avg_cost"], current_price=0)
    logger.info(f"Recovered: {len(positions)} positions, {recovered['replayed_entries']} entries")

    pending = journal.get_pending_orders()
    if pending:
        logger.warning(f"{len(pending)} pending orders need review!")
        alerter.send(AlertLevel.WARNING, "Pending orders on startup",
                     f"{len(pending)} orders not in terminal state")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Restore breaker state if checkpoint exists
    last_cp = journal.get_last_checkpoint()
    if last_cp:
        _, snapshot = last_cp
        if "breaker_state" in snapshot:
            try:
                breaker_mgr = BreakerManager.restore_state(snapshot["breaker_state"])
                logger.info("Breaker state restored from checkpoint")
            except Exception as exc:
                logger.warning(f"Could not restore breaker state: {exc}")

    if recovered["checkpoint_ts"] is None:
        pos_dict = {s: {"quantity": p.quantity, "avg_cost": p.avg_cost}
                    for s, p in positions.items() if p.quantity > 0}
        snapshot = build_checkpoint_snapshot(pos_dict, cash, pending)
        snapshot["breaker_state"] = breaker_mgr.save_state()
        journal.save_checkpoint(snapshot)

    # Ingest data
    logger.info("Fetching market data...")
    ok, errors = run_daily_ingest()
    for e in errors:
        logger.warning(e)

    symbols = settings.ETF_UNIVERSE
    bars_dict = {}
    for sym in symbols:
        bars = storage.load_bars(sym)
        if bars:
            bars_dict[sym] = bars

    if not bars_dict:
        logger.error("No data. Set TUSHARE_TOKEN in .env then re-run.")
        return

    logger.info(f"Loaded data for {len(bars_dict)} symbols. Live loop starting.")
    alerter.send(AlertLevel.WARNING, "Live Engine Started",
                 f"Capital: {settings.TOTAL_CAPITAL}, Universe: {symbols}")
    metrics.inc_counter("live.runs")
    metrics.set_gauge("live.capital", settings.TOTAL_CAPITAL)

    try:
        while running:
            if not breaker_mgr.check_all():
                logger.warning(f"CIRCUIT BREAKER TRIPPED: {breaker_mgr.status()}")
                time.sleep(30)
                continue

            # Build market data
            common = {}
            for sym, bars in bars_dict.items():
                common[sym] = [Bar(symbol=b.symbol,
                                   datetime=datetime.strptime(b.trade_date, "%Y%m%d"),
                                   open=b.open, high=b.high, low=b.low,
                                   close=b.close, volume=b.volume) for b in bars]
            md = MarketData(bars=common, index_bars={}, timestamp=datetime.now())

            for sym, bars in bars_dict.items():
                if bars and sym in positions:
                    positions[sym].current_price = bars[-1].close

            total_mv = sum(p.quantity * p.current_price for p in positions.values())
            pf = Portfolio(
                positions={s: p for s, p in positions.items() if p.quantity > 0},
                cash=cash, total_capital=cash + total_mv, timestamp=datetime.now(),
            )

            # Strategy -> Orders
            signals = strategy.generate_signals(md)
            if signals:
                logger.info(f"Signals: {[(s.symbol, s.side.value) for s in signals]}")

            orders = strategy.size_positions(signals, cash, pf)

            # Unified risk path: strategy exits + RiskChecker pre-trade checks
            orders = strategy.risk_check(orders, pf, market_data=md, risk_checker=risk_checker)

            daily_pnl = 0.0
            for o in orders:
                logger.info(f"LIVE ORDER: {o.symbol} {o.side.value} {o.quantity} @ {o.price or 'MKT'}")
                # TODO Phase 5: Send to MiniQMT broker
                # broker.submit_order(o)
                metrics.inc_counter("live.orders")
                breaker_mgr.general.record_execution_success()
                # Track PnL for circuit breakers (live: use fill price)
                if o.side.value == "sell":
                    pos = positions.get(o.symbol)
                    if pos and pos.avg_cost > 0:
                        est_pnl = (pos.current_price - pos.avg_cost) * o.quantity
                        daily_pnl += est_pnl
                        breaker_mgr.monthly_drawdown.record_trade_pnl(est_pnl)
                        breaker_mgr.consecutive_loss.record_profit(o.symbol, est_pnl)

            breaker_mgr.general.update_pnl(daily_pnl)
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"FATAL: {e}")
        alerter.send(AlertLevel.CRITICAL, "Live Engine CRASHED", str(e))
    finally:
        pos_dict = {s: {"quantity": p.quantity, "avg_cost": p.avg_cost}
                    for s, p in positions.items() if p.quantity > 0}
        snapshot = build_checkpoint_snapshot(pos_dict, cash, [])
        snapshot["breaker_state"] = breaker_mgr.save_state()
        journal.save_checkpoint(snapshot)
        logger.info("Stopped. Final checkpoint saved.")
        logger.info(f"Metrics: {metrics.snapshot()}")


if __name__ == "__main__":
    main()

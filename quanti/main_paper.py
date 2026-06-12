"""Paper Trading Engine (Phase 4).
Runs the full pipeline end-to-end with live data and simulated fills.
Usage: python -m quanti.main_paper
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

logger = get_logger("paper")
running = True


def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received")
    running = False


def main():
    global running
    setup_logger("paper_trading")
    logger.info("=" * 60)
    logger.info("PAPER TRADING ENGINE")
    logger.info(f"Capital: {settings.TOTAL_CAPITAL} RMB | Allocation: {settings.TRADING_CAPITAL} RMB")
    logger.info(f"Universe: {settings.ETF_UNIVERSE}")
    logger.info("=" * 60)

    journal = Journal()
    storage = DataStorage()
    alerter = get_alerter()
    metrics = get_metrics()
    breaker_mgr = BreakerManager()
    strategy = ETFTrendStrategy()
    risk_checker = RiskChecker()

    # Crash recovery
    recovered = recover_portfolio(journal)
    cash = recovered["cash"]
    if cash is None:
        logger.warning("No checkpoint: starting with fresh capital")
        cash = settings.TRADING_CAPITAL
    positions = {}
    for s, q in recovered["positions"].items():
        positions[s] = Position(symbol=s, quantity=q["quantity"], avg_cost=q["avg_cost"], current_price=0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Restore breaker state if available from checkpoint
    last_cp = journal.get_last_checkpoint()
    if last_cp:
        _, snapshot = last_cp
        if "breaker_state" in snapshot:
            try:
                breaker_mgr = BreakerManager.restore_state(snapshot["breaker_state"])
                logger.info("Breaker state restored from checkpoint")
            except Exception as exc:
                logger.warning(f"Could not restore breaker state: {exc}")

    # Initial checkpoint (if no prior checkpoint exists)
    if recovered["checkpoint_ts"] is None:
        pos_dict = {s: {"quantity": p.quantity, "avg_cost": p.avg_cost}
                    for s, p in positions.items() if p.quantity > 0}
        snapshot = build_checkpoint_snapshot(pos_dict, cash, [])
        snapshot["breaker_state"] = breaker_mgr.save_state()
        journal.save_checkpoint(snapshot)

    # Ingest latest data
    logger.info("Fetching data...")
    ok, errors = run_daily_ingest()
    for e in errors:
        logger.warning(f"Data: {e}")

    # Load historical bars
    symbols = settings.ETF_UNIVERSE
    bars_dict = {}
    for sym in symbols:
        bars = storage.load_bars(sym)
        if bars:
            bars_dict[sym] = bars
            logger.info(f"  {sym}: {len(bars)} bars, last={bars[-1].trade_date}")
        else:
            logger.warning(f"  {sym}: no data (set TUSHARE_TOKEN in .env)")

    if not bars_dict:
        logger.error("No data. Configure TUSHARE_TOKEN in .env then re-run.")
        return

    logger.info("Trading loop starting. Ctrl+C to stop.")
    metrics.inc_counter("paper.runs")

    try:
        while running:
            if not breaker_mgr.check_all():
                logger.warning(f"BREAKER TRIPPED: {breaker_mgr.status()}")
                time.sleep(30)
                continue

            # Build market data from loaded bars
            common = {}
            for sym, bars in bars_dict.items():
                common[sym] = [Bar(symbol=b.symbol,
                                   datetime=datetime.strptime(b.trade_date, "%Y%m%d"),
                                   open=b.open, high=b.high, low=b.low,
                                   close=b.close, volume=b.volume) for b in bars]
            md = MarketData(bars=common, index_bars={}, timestamp=datetime.now())

            # Update prices
            for sym, bars in bars_dict.items():
                if bars and sym in positions:
                    positions[sym].current_price = bars[-1].close

            total_mv = sum(p.quantity * p.current_price for p in positions.values())
            pf = Portfolio(
                positions={s: p for s, p in positions.items() if p.quantity > 0},
                cash=cash, total_capital=cash + total_mv, timestamp=datetime.now(),
            )

            # Strategy cycle
            signals = strategy.generate_signals(md)
            if signals:
                logger.info(f"Signals: {[(s.symbol, s.side.value, f'{s.strength:.2f}') for s in signals]}")

            orders = strategy.size_positions(signals, cash, pf)

            # Unified risk path: strategy exits + RiskChecker pre-trade checks
            orders = strategy.risk_check(orders, pf, market_data=md, risk_checker=risk_checker)

            daily_pnl = 0.0
            for o in orders:
                logger.info(f"ORDER: {o.symbol} {o.side.value} {o.quantity} @ {o.price or 'market'}")
                # In paper mode, simulate fill at close price
                bar = bars_dict.get(o.symbol, [])[-1] if bars_dict.get(o.symbol) else None
                if bar:
                    metrics.inc_counter("paper.orders")
                    breaker_mgr.general.record_execution_success()
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

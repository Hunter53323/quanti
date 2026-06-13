"""
One-shot daily paper trading run. Called by Windows Task Scheduler once per day.
Ingests latest data, runs one strategy cycle, saves checkpoint, exits.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

logger = get_logger("daily")


def main():
    setup_logger("daily_paper_trading")
    logger.info("=" * 60)
    logger.info(f"DAILY PAPER TRADING RUN -- {datetime.now():%Y-%m-%d %H:%M}")
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
    risk_checker = RiskChecker()
    logger.info(f"Strategy: {strategy.name} | Bond: {strategy.bond_etf} {strategy.defensive_bond_pct:.0%} | Gold: {strategy.gold_etf} {strategy.defensive_gold_pct:.0%}")

    # ── Universe ──
    all_files = sorted(storage.clean_dir.glob("*.parquet"))
    stock_codes = [p.stem for p in all_files if len(p.stem) == 6
                   and not p.stem.startswith(("51", "58", "15", "56"))]
    symbols = stock_codes + [strategy.bond_etf, strategy.gold_etf, "510300"]
    strategy._stock_universe = set(stock_codes)
    logger.info(f"Universe: {len(symbols)} symbols ({len(stock_codes)} stocks)")

    # ── Recovery ──
    recovered = recover_portfolio(journal)
    cash = recovered["cash"]
    if cash is None:
        cash = settings.TRADING_CAPITAL
    positions = {}
    for s, q in recovered["positions"].items():
        if s in strategy._stock_universe or s in (strategy.bond_etf, strategy.gold_etf):
            positions[s] = Position(symbol=s, quantity=q["quantity"], avg_cost=q["avg_cost"], current_price=0)
        else:
            logger.info(f"  Dropping stale position: {s}")

    if recovered["checkpoint_ts"] is None:
        pos_dict = {s: {"quantity": p.quantity, "avg_cost": p.avg_cost}
                    for s, p in positions.items() if p.quantity > 0}
        snapshot = build_checkpoint_snapshot(pos_dict, cash, [])
        snapshot["breaker_state"] = breaker_mgr.save_state()
        journal.save_checkpoint(snapshot)

    # ── Ingest ──
    logger.info("Ingesting latest data...")
    ok, errors = run_daily_ingest()
    for e in errors:
        logger.warning(f"  Data: {e}")

    # ── Load bars ──
    bars_dict = {}
    for sym in symbols:
        bars = storage.load_bars(sym)
        if bars:
            bars_dict[sym] = bars

    csi300_bars = bars_dict.get("510300", [])
    latest_date = max(b.trade_date for v in bars_dict.values() for b in v) if bars_dict else "unknown"
    logger.info(f"  Loaded {len(bars_dict)} symbols, latest date: {latest_date}")
    for sym in ("510300", strategy.bond_etf, strategy.gold_etf):
        if sym in bars_dict:
            b = bars_dict[sym]
            logger.info(f"    {sym}: {len(b)} bars, last={b[-1].trade_date} close={b[-1].close:.2f}")

    # ── Build MarketData ──
    common = {}
    for sym, bars in bars_dict.items():
        common[sym] = [Bar(symbol=b.symbol, datetime=datetime.strptime(b.trade_date, "%Y%m%d"),
                           open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
                       for b in bars]
    index_bars = {}
    if csi300_bars:
        index_bars["510300"] = common["510300"]
    md = MarketData(bars=common, index_bars=index_bars, timestamp=datetime.now())

    # ── Update current prices ──
    for sym, bars in bars_dict.items():
        if bars and sym in positions:
            positions[sym].current_price = bars[-1].close

    total_mv = sum(p.quantity * p.current_price for p in positions.values())
    pf = Portfolio(positions={s: p for s, p in positions.items() if p.quantity > 0},
                   cash=cash, total_capital=cash + total_mv,
                   settled_cash=cash, timestamp=datetime.now())

    logger.info(f"  Portfolio: cash={cash:,.0f} equity={total_mv:,.0f} total={cash+total_mv:,.0f}")

    # ── Strategy cycle ──
    signals = strategy.generate_signals(md)
    if signals:
        logger.info(f"  SIGNALS ({len(signals)}):")
        for s in signals:
            logger.info(f"    {s.side.value:4s} {s.symbol:8s} strength={s.strength:.2f} {s.reason}")
    else:
        logger.info("  No signals generated (not a rebalance day)")

    orders = strategy.size_positions(signals, cash, pf, md)
    orders = strategy.risk_check(orders, pf, market_data=md, risk_checker=risk_checker)

    di = strategy.get_decay_info()
    si = strategy.get_sharp_info()
    if di["months_in_cycle"] > 0:
        logger.info(f"  Decay: {di['schedule']} month {di['months_in_cycle']} multiplier {di['current_multiplier']:.0%}")
    if si["sharp_fired_recent"]:
        logger.warning(f"  SHARP EXIT FIRED! threshold={si['sharp_threshold']*100:.0f}%")

    # ── Execute orders ──
    # Snapshot cash for defensive allocation (bond+gold share one capital base)
    cash_snapshot = cash
    daily_pnl = 0.0
    for o in orders:
        bar = bars_dict.get(o.symbol, [])[-1] if bars_dict.get(o.symbol) else None
        if not bar:
            continue
        fill_price = o.price or bar.close
        fill_cost = o.quantity * fill_price
        comm = fill_cost * getattr(settings, "COMMISSION_RATE", 0.00025)

        if o.side.value == "buy":
            # For defensive buys (bond/gold), check against snapshot so both fill
            effective_budget = cash_snapshot if o.symbol in (strategy.bond_etf, strategy.gold_etf) else cash
            total_cost = fill_cost + comm
            if total_cost <= effective_budget:
                cash -= total_cost
                if o.symbol in positions:
                    p = positions[o.symbol]
                    nq = p.quantity + o.quantity
                    na = (p.avg_cost * p.quantity + fill_price * o.quantity) / nq
                    positions[o.symbol] = Position(symbol=o.symbol, quantity=nq, avg_cost=na, current_price=fill_price)
                else:
                    positions[o.symbol] = Position(symbol=o.symbol, quantity=o.quantity, avg_cost=fill_price, current_price=fill_price)
                logger.info(f"    FILL BUY  {o.symbol:8s} x{o.quantity:>6d} @{fill_price:>8.2f} cost={total_cost:>10,.0f}")
            else:
                logger.warning(f"    SKIP BUY  {o.symbol:8s}: need {total_cost:,.0f}, have {cash:,.0f}")
        else:
            pos = positions.get(o.symbol)
            if pos and pos.quantity >= o.quantity:
                proceeds = fill_cost - comm
                cash += proceeds
                pos.quantity -= o.quantity
                if pos.quantity == 0:
                    del positions[o.symbol]
                pnl = (fill_price - pos.avg_cost) * o.quantity
                daily_pnl += pnl
                logger.info(f"    FILL SELL {o.symbol:8s} x{o.quantity:>6d} @{fill_price:>8.2f} PnL={pnl:>+10,.0f}")
            else:
                logger.warning(f"    SKIP SELL {o.symbol:8s}: insuf quantity")
        metrics.inc_counter("paper.orders")
        breaker_mgr.general.record_execution_success()

    breaker_mgr.general.update_pnl(daily_pnl)

    # ── Save checkpoint ──
    pos_dict = {s: {"quantity": p.quantity, "avg_cost": p.avg_cost}
                for s, p in positions.items() if p.quantity > 0}
    snapshot = build_checkpoint_snapshot(pos_dict, cash, [])
    snapshot["breaker_state"] = breaker_mgr.save_state()
    journal.save_checkpoint(snapshot)

    final_mv = sum(p.quantity * p.current_price for p in positions.values())
    logger.info(f"  Final: cash={cash:,.0f} equity={final_mv:,.0f} total={cash+final_mv:,.0f} PnL={daily_pnl:+.0f}")
    logger.info(f"  Checkpoint saved. Metrics: {metrics.snapshot()}")
    logger.info("=" * 60)

    if daily_pnl < -5000:
        alerter.send(AlertLevel.WARNING, "Paper Trading Large Loss", f"Daily PnL: {daily_pnl:,.0f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        get_alerter().send(AlertLevel.CRITICAL, "Paper Trading Crashed", str(e))
        sys.exit(1)

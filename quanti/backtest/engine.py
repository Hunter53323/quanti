"""Backtest engine with walk-forward analysis and full metrics."""
import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from quanti.config import settings
from quanti.execution.risk import RiskChecker
from quanti.strategy.etf_trend import ETFTrendStrategy
from quanti.types import Bar, MarketData, Portfolio, Position


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    period_label: str = ""
    params: dict = field(default_factory=dict)
    start_date: str = ""
    end_date: str = ""
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    annual_turnover_pct: float = 0.0
    equity_curve: list = field(default_factory=list)
    daily_returns: list = field(default_factory=list)
    drawdown_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    def as_dict(self):
        d = {k: v for k, v in self.__dict__.items()}
        for key in ("equity_curve", "daily_returns", "drawdown_curve"):
            if len(d.get(key, [])) > 100:
                d[key] = d[key][:100]
        return d

    def summarize(self):
        return (
            f"Period: {self.period_label} ({self.start_date} to {self.end_date})\n"
            f"CAGR: {self.cagr_pct:.1f}% Sharpe: {self.sharpe_ratio:.2f} MaxDD: {self.max_drawdown_pct:.1f}% Trades: {len(self.trades)}"
        )


class BacktestEngine:
    """Event-walk backtest engine for daily-bar strategies."""

    def __init__(self, strategy_class=None, params=None, initial_capital=None,
                 commission_rate=None, slippage_bps=None, pe_provider: Callable | None = None):
        self.strategy_class = strategy_class or ETFTrendStrategy
        self.params = params or {}
        self.initial_capital = initial_capital or settings.TRADING_CAPITAL
        self.commission_rate = commission_rate or settings.COMMISSION_RATE
        self.slippage_bps = slippage_bps or settings.SLIPPAGE_BPS
        self.risk_checker = RiskChecker()  # Unified risk path
        self.pe_provider = pe_provider     # Callable(date_str) -> dict | None for PE-band strategies

    def run(self, symbols, bars_dict, index_bars=None, period_label="backtest"):
        strat = self.strategy_class(**self.params)
        cash = self.initial_capital
        settled_cash = cash  # T+1: all cash is settled initially
        pending_settlement = 0.0
        settlement_lag = getattr(settings, 'SETTLEMENT_LAG_DAYS', 1)
        pending_dates: dict[str, tuple[float, int]] = {}  # date -> (amount, days_remaining)
        positions = {}
        eq, pnl_list, trades = [], [], []
        all_dates = sorted({b.trade_date for v in bars_dict.values() for b in v})
        if not all_dates:
            return BacktestResult(period_label=period_label, params=self.params)
        bd = {}
        for s, bs in bars_dict.items():
            for b in bs:
                bd.setdefault(b.trade_date, {})[s] = b
        for td in all_dates:
            # T+1 settlement: advance pending settlements
            newly_settled = 0.0
            expired_keys = []
            for key, (amount, days) in pending_dates.items():
                remaining = days - 1
                if remaining <= 0:
                    newly_settled += amount
                    expired_keys.append(key)
                else:
                    pending_dates[key] = (amount, remaining)
            for k in expired_keys:
                del pending_dates[k]
            settled_cash += newly_settled
            pending_settlement -= newly_settled

            md = self._mk_md(td, symbols, bars_dict, index_bars)
            mv = sum(p.quantity * p.current_price for p in positions.values())
            total_capital = cash + mv
            pf = Portfolio(
                positions={k: v for k, v in positions.items() if v.quantity > 0},
                cash=cash, total_capital=total_capital,
                settled_cash=settled_cash, pending_settlement=pending_settlement,
                settlement_lag_days=settlement_lag,
                timestamp=datetime.strptime(td, "%Y%m%d"))
            for s, b in bd.get(td, {}).items():
                if s in positions:
                    positions[s].current_price = b.close
            sigs = strat.generate_signals(md)
            odrs = strat.size_positions(sigs, settled_cash, pf, market_data=md)

            # Unified risk path: strategy exits + RiskChecker pre-trade checks
            appr = strat.risk_check(odrs, pf, market_data=md, risk_checker=self.risk_checker)

            for o in appr:
                b = bd.get(td, {}).get(o.symbol)
                if not b: continue
                ep = self._slip(b.close, o.side)
                cst = o.quantity * ep
                comm = cst * self.commission_rate
                if o.side.value == "buy" and cst + comm <= settled_cash:
                    cash -= cst + comm
                    settled_cash -= cst + comm
                    if o.symbol in positions:
                        p = positions[o.symbol]
                        nq = p.quantity + o.quantity
                        nc = (p.avg_cost * p.quantity + ep * o.quantity) / nq
                        positions[o.symbol] = Position(symbol=o.symbol, quantity=nq, avg_cost=nc, current_price=b.close)
                    else:
                        positions[o.symbol] = Position(symbol=o.symbol, quantity=o.quantity, avg_cost=ep, current_price=b.close)
                    trades.append(dict(date=td, symbol=o.symbol, side="buy", quantity=o.quantity, price=ep, commission=comm))
                elif o.side.value == "sell":
                    if o.symbol in positions and positions[o.symbol].quantity >= o.quantity:
                        p = positions[o.symbol]
                        pnl_val = (ep - p.avg_cost) * o.quantity - comm
                        cash += cst - comm
                        # Proceeds go to pending settlement (T+1)
                        sell_proceeds = cst - comm
                        pending_settlement += sell_proceeds
                        # Use unique key per order to avoid overwrite on same-day sells
                        order_key = f"{td}-{o.symbol}-sell-{len(trades)}"
                        pending_dates[order_key] = (sell_proceeds, settlement_lag)
                        p.quantity -= o.quantity
                        if p.quantity == 0: del positions[o.symbol]
                        trades.append(dict(date=td, symbol=o.symbol, side="sell", quantity=o.quantity, price=ep, commission=comm, pnl=pnl_val))
            mv = sum(p.quantity * p.current_price for p in positions.values())
            eq.append(cash + mv)
            if len(eq) > 1: pnl_list.append(eq[-1] - eq[-2])
        r = BacktestResult(period_label=period_label, params=self.params, start_date=all_dates[0], end_date=all_dates[-1], equity_curve=eq, daily_returns=pnl_list, trades=trades)
        self._metrics(r, pnl_list, eq, len(all_dates))
        return r
    def _mk_md(self, date, symbols, bars_dict, index_bars=None):
        db = {}
        for sym in symbols:
            sb = bars_dict.get(sym, [])
            db[sym] = [b for b in sb if b.trade_date <= date]
        common = {}
        for sym, bl in db.items():
            common[sym] = [Bar(symbol=b.symbol, datetime=datetime.strptime(b.trade_date, "%Y%m%d"), open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume) for b in bl]
        common_idx = {}
        if index_bars:
            for idx, bl in index_bars.items():
                common_idx[idx] = [Bar(symbol=b.symbol, datetime=datetime.strptime(b.trade_date, "%Y%m%d"), open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume) for b in bl if b.trade_date <= date]

        # Inject PE/PB fundamentals for allocation strategies
        index_fundamentals = None
        if self.pe_provider is not None:
            with contextlib.suppress(Exception):
                index_fundamentals = self.pe_provider(date)

        return MarketData(bars=common, index_bars=common_idx, index_fundamentals=index_fundamentals,
                          timestamp=datetime.strptime(date, "%Y%m%d"))

    def _slip(self, price, side):
        slip = price * self.slippage_bps / 10000
        return price + slip if side.value == "buy" else price - slip
    @staticmethod
    def _metrics(result, daily_pnl, equity_curve, n_dates):
        if not equity_curve or equity_curve[0] == 0: return
        init = equity_curve[0]
        final = equity_curve[-1]
        result.total_return_pct = (final - init) / init * 100
        years = n_dates / 252
        if years > 0 and init > 0:
            result.cagr_pct = ((final / init) ** (1 / years) - 1) * 100
        if daily_pnl and len(daily_pnl) > 1:
            pa = np.array(daily_pnl)
            excess = pa - (0.025 / 252) * init
            std_excess = np.std(excess)
            if std_excess > 1e-10:
                raw = np.mean(excess) / std_excess * math.sqrt(252)
                result.sharpe_ratio = float(max(min(raw, 100.0), -100.0))  # clamp to [-100, 100]
        if equity_curve:
            peak = equity_curve[0]
            max_dd, max_dd_days, cur_dd = 0.0, 0, 0
            dd_curve = []
            for v in equity_curve:
                if v > peak: peak = v; cur_dd = 0
                dd = (peak - v) / peak * 100
                dd_curve.append(dd)
                if dd > max_dd: max_dd = dd
                if dd > 0: cur_dd += 1
                if cur_dd > max_dd_days: max_dd_days = cur_dd
            result.max_drawdown_pct = max_dd
            result.max_drawdown_duration = max_dd_days
            result.drawdown_curve = dd_curve
            if max_dd > 0: result.calmar_ratio = result.cagr_pct / max_dd
        if result.trades:
            sells = [t for t in result.trades if t.get("side") == "sell" and "pnl" in t]
            if sells:
                wins = sum(1 for t in sells if t["pnl"] > 0)
                result.win_rate_pct = wins / len(sells) * 100
                gp = sum(t["pnl"] for t in sells if t["pnl"] > 0)
                gl = abs(sum(t["pnl"] for t in sells if t["pnl"] < 0))
                if gl > 0: result.profit_factor = gp / gl
            tv = sum(t["quantity"] * t["price"] for t in result.trades if t["side"] == "buy")
            if years > 0 and init > 0:
                result.annual_turnover_pct = (tv / init / years) * 100
    def run_walk_forward(self, symbols, bars_dict, train_years=1, test_months=3, index_bars=None):
        all_dates = sorted({b.trade_date for v in bars_dict.values() for b in v})
        if not all_dates: return []
        start = datetime.strptime(all_dates[0], "%Y%m%d")
        end = datetime.strptime(all_dates[-1], "%Y%m%d")
        results = []
        ws = start
        while ws + timedelta(days=train_years * 365 + test_months * 30) <= end:
            te = ws + timedelta(days=train_years * 365)
            tse = te + timedelta(days=test_months * 30)
            ws_s = ws.strftime("%Y%m%d")
            te_s = te.strftime("%Y%m%d")
            tse_s = tse.strftime("%Y%m%d")
            train_bars = {sym: [b for b in bars if ws_s <= b.trade_date <= te_s] for sym, bars in bars_dict.items()}
            test_bars = {sym: [b for b in bars if te_s < b.trade_date <= tse_s] for sym, bars in bars_dict.items()}
            best_r, best_s = None, -999.0
            for mf in settings.WF_MA_FAST_RANGE:
                for ms in settings.WF_MA_SLOW_RANGE:
                    if mf >= ms: continue
                    for adx in settings.WF_ADX_RANGE:
                        e = BacktestEngine(strategy_class=self.strategy_class, params=dict(ma_fast=mf, ma_slow=ms, adx_threshold=adx), initial_capital=self.initial_capital)
                        r = e.run(symbols, train_bars, index_bars, period_label=f"train_{ws_s}_{te_s}")
                        if r.sharpe_ratio > best_s: best_s, best_r = r.sharpe_ratio, r
            if best_r:
                results.append(best_r)
                if test_bars:
                    te2 = BacktestEngine(strategy_class=self.strategy_class, params=best_r.params, initial_capital=self.initial_capital)
                    results.append(te2.run(symbols, test_bars, index_bars, period_label=f"test_{te_s}_{tse_s}"))
            ws = tse
        return results
    def run_out_of_sample(self, symbols, bars_dict, index_bars=None):
        in_bars = {sym: [b for b in bars if "20190101" <= b.trade_date <= "20231231"] for sym, bars in bars_dict.items()}
        out_bars = {sym: [b for b in bars if "20240101" <= b.trade_date <= "20251231"] for sym, bars in bars_dict.items()}
        wf = self.run_walk_forward(symbols, in_bars, train_years=1, test_months=3, index_bars=index_bars)
        best_params = dict(ma_fast=20, ma_slow=60)
        if wf:
            tr = [r for r in wf if "train" in r.period_label]
            if tr: best_params = max(tr, key=lambda r: r.sharpe_ratio).params
        is_eng = BacktestEngine(strategy_class=self.strategy_class, params=best_params, initial_capital=self.initial_capital)
        oos_eng = BacktestEngine(strategy_class=self.strategy_class, params=best_params, initial_capital=self.initial_capital)
        return (is_eng.run(symbols, in_bars, index_bars, period_label="in-sample-2019-2023"), oos_eng.run(symbols, out_bars, index_bars, period_label="out-of-sample-2024-2025"))
    @staticmethod
    def summarize_results(results):
        tr = [r for r in results if "train" in r.period_label]
        tst = [r for r in results if "test" in r.period_label]
        lines = ["=" * 60, "WALK-FORWARD ANALYSIS SUMMARY", "=" * 60]
        if tr:
            ss = [r.sharpe_ratio for r in tr]
            dd = [r.max_drawdown_pct for r in tr]
            lines.append(f"Train windows: {len(tr)}")
            lines.append(f"  Avg Sharpe: {np.mean(ss):.3f} (std: {np.std(ss):.3f})")
            lines.append(f"  Avg MaxDD:  {np.mean(dd):.2f}%")
            pl = [tuple(sorted(r.params.items())) for r in tr]
            unique = len(set(pl))
            lines.append(f"  Unique param sets: {unique}/{len(tr)}")
            if unique <= 2: lines.append("  PARAMS: stable")
            else: lines.append(f"  PARAMS: unstable ({unique} unique sets - suspect overfitting)")
        if tst:
            ss = [r.sharpe_ratio for r in tst]
            dd = [r.max_drawdown_pct for r in tst]
            lines.append(f"Test windows: {len(tst)}")
            lines.append(f"  Avg Sharpe: {np.mean(ss):.3f}")
            lines.append(f"  Avg MaxDD:  {np.mean(dd):.2f}%")
        return chr(10).join(lines)

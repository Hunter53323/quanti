"""
Backtest runner for MarketStateETFStrategy via BacktestEngine.

Usage:
    python scripts/run_market_state_backtest.py
"""
import sys; sys.path.insert(0, r"C:\study\AIWorkspace\quanti")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from quanti.data.storage import DataStorage
from quanti.backtest.engine import BacktestEngine
from quanti.strategy.market_state_etf import MarketStateETFStrategy
from quanti.config.etf_universe import ETF_UNIVERSE_MULTI

TRAIN_START, TRAIN_END = "20150101", "20211231"
TEST_START,  TEST_END  = "20220101", "20251231"
CSI300_CODE = "510300"


def load_etf_bars(storage, codes, start=None, end=None):
    """Load bars for a list of ETF codes, optionally filtered by date range."""
    result = {}
    for code in codes:
        bars = storage.load_bars(code)
        if not bars or len(bars) < 200:
            continue
        if start or end:
            bars = [b for b in bars
                    if (not start or b.trade_date >= start)
                    and (not end or b.trade_date <= end)]
            if len(bars) < 200:
                continue
        result[code] = bars
    return result


def progress(msg):
    print(f"  {msg}")


def main():
    t0 = time.time()
    storage = DataStorage()

    # ── CSI300 bars (for market state) ──
    csi300_bars = storage.load_bars(CSI300_CODE)
    if not csi300_bars:
        print("ERROR: CSI300 data not found")
        return
    progress(f"CSI300: {len(csi300_bars)} bars")

    # ── ETF universe ──
    etf_codes = [e["code"] for e in ETF_UNIVERSE_MULTI]
    bars_dict = load_etf_bars(storage, etf_codes)
    progress(f"Universe: {len(bars_dict)} ETFs loaded")

    # ── Build strategy ──
    params = dict(
        csi300_bars=csi300_bars,
        top_n=3,
        n_confirm=5,
        m_cooldown=40,
        sharp_threshold=-0.03,
        bond_pct=0.80,
        gold_pct=0.20,
        hwm_stop_pct=-10.0,
        dd_exit_pct=15.0,
        max_per_sector=2,
    )

    # ── Full period ──
    progress("Running full period (2015-2025)...")
    engine = BacktestEngine(
        strategy_class=MarketStateETFStrategy,
        params=params,
    )
    full_result = engine.run(
        symbols=list(bars_dict.keys()),
        bars_dict=bars_dict,
        period_label="full-2015-2025",
    )
    print(f"  Full: CAGR={full_result.cagr_pct:+.2f}% Sharpe={full_result.sharpe_ratio:.3f} "
          f"MaxDD={full_result.max_drawdown_pct:.1f}% Trades={len(full_result.trades)}")

    # ── Train / Test split ──
    progress("Train (2015-2021)...")
    train_bars = load_etf_bars(storage, etf_codes, start=TRAIN_START, end=TRAIN_END)
    engine_train = BacktestEngine(
        strategy_class=MarketStateETFStrategy,
        params=params,
    )
    train_result = engine_train.run(
        symbols=list(train_bars.keys()),
        bars_dict=train_bars,
        period_label="train-2015-2021",
    )

    progress("Test (2022-2025)...")
    test_bars = load_etf_bars(storage, etf_codes, start=TEST_START, end=TEST_END)
    engine_test = BacktestEngine(
        strategy_class=MarketStateETFStrategy,
        params=params,
    )
    test_result = engine_test.run(
        symbols=list(test_bars.keys()),
        bars_dict=test_bars,
        period_label="test-2022-2025",
    )

    print(f"\n{'='*70}")
    print(f"  MARKET-STATE ETF STRATEGY")
    print(f"{'='*70}")
    print(f"  Train (2015-2021):  CAGR={train_result.cagr_pct:+.2f}%  "
          f"Sharpe={train_result.sharpe_ratio:.3f}  MaxDD={train_result.max_drawdown_pct:.1f}%  "
          f"Trades={len(train_result.trades)}")
    print(f"  Test  (2022-2025):  CAGR={test_result.cagr_pct:+.2f}%  "
          f"Sharpe={test_result.sharpe_ratio:.3f}  MaxDD={test_result.max_drawdown_pct:.1f}%  "
          f"Trades={len(test_result.trades)}")

    print(f"\n  Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

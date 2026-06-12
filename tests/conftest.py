"""Shared test fixtures."""
import pytest


@pytest.fixture
def sample_bars():
    """Create a list of valid ETFDailyBar objects for testing."""
    from quanti.data.schema import ETFDailyBar

    return [
        ETFDailyBar(
            symbol="510300.SH",
            trade_date="20240102",
            open=3.500, high=3.520, low=3.480, close=3.510,
            volume=1000000.0, amount=3510000.0,
        ),
        ETFDailyBar(
            symbol="510300.SH",
            trade_date="20240103",
            open=3.510, high=3.550, low=3.500, close=3.540,
            volume=1200000.0, amount=4230000.0,
        ),
        ETFDailyBar(
            symbol="510300.SH",
            trade_date="20240104",
            open=3.540, high=3.560, low=3.520, close=3.530,
            volume=1100000.0, amount=3889000.0,
        ),
    ]


@pytest.fixture
def corrupt_bars():
    """Create bars with data quality issues for validation testing."""
    from quanti.data.schema import ETFDailyBar

    return [
        ETFDailyBar(
            symbol="510300.SH",
            trade_date="20240101",
            open=0.0, high=0.0, low=0.0, close=0.0,  # Zero prices (suspended)
            volume=0.0, amount=0.0,
        ),
        ETFDailyBar(
            symbol="510300.SH",
            trade_date="20240102",
            open=3.500, high=3.400, low=3.600, close=3.500,  # high < low
            volume=1000000.0, amount=3510000.0,
        ),
        ETFDailyBar(
            symbol="510300.SH",
            trade_date="20240103",
            open=3.510, high=3.550, low=3.500, close=3.540,
            volume=1200000.0, amount=4230000.0,
        ),
        ETFDailyBar(  # Duplicate date
            symbol="510300.SH",
            trade_date="20240103",
            open=3.511, high=3.551, low=3.501, close=3.541,
            volume=1000.0, amount=3541000.0,
        ),
    ]


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    return str(tmp_path / "test_quanti.db")


@pytest.fixture
def journal(temp_db_path):
    """Create a Journal backed by a temp database."""
    from quanti.state.journal import Journal
    return Journal(temp_db_path)


@pytest.fixture(autouse=True)
def restore_settings():
    """Automatically restore settings after any test that mutates them."""
    import quanti.config.settings as st
    # Save original values of commonly-mutated settings
    originals = {}
    mutable_keys = [
        "STOP_LOSS_PCT", "TRADING_CAPITAL", "ATR_TRAILING_STOP_ENABLED",
        "TIME_STOP_ENABLED", "VOLATILITY_STOP_ENABLED", "RSI_EXIT_ENABLED",
        "COMMISSION_RATE", "SLIPPAGE_BPS",
    ]
    for key in mutable_keys:
        if hasattr(st, key):
            originals[key] = getattr(st, key)

    yield  # Test runs here

    # Restore original values
    for key, value in originals.items():
        import contextlib
        with contextlib.suppress(AttributeError, TypeError):
            setattr(st, key, value)

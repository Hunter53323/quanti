"""
Application-wide settings loaded from environment variables.
All tunable parameters live here -- never hardcoded in strategy or execution code.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Data ---
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
DATA_DIR = os.getenv("DATA_DIR", "C:/study/AIWorkspace/quanti/data")

# --- Strategy: ETF Trend ---
ETF_UNIVERSE = ["510300", "510500", "159915", "518880"]  # CSI 300, CSI 500, ChiNext 50, Gold

# Extended universe for PE-band, dividend, gold, bond strategies
ETF_UNIVERSE_EXTENDED = os.getenv(
    "ETF_UNIVERSE_EXTENDED",
    "510300,510500,159915,510880,518880,511880"
).split(",")

# --- Strategy: ETF Rotation Multi-Industry ---
ETF_ROTATION_MULTI_ENABLED = os.getenv("ETF_ROTATION_MULTI_ENABLED", "false").lower() == "true"
ETF_ROTATION_MULTI_UNIVERSE = os.getenv(
    "ETF_ROTATION_MULTI_UNIVERSE",
    "510300,510500,159915,588000,512000,512800,512480,515070,515880,512720,"
    "516160,516880,516110,159928,512010,159825,516810,516310,516320,"
    "512980,159869,512660,510880,518880,511880"
).split(",")
ETF_ROTATION_MULTI_TOP_N = int(os.getenv("ETF_ROTATION_MULTI_TOP_N", "3"))
ETF_ROTATION_MULTI_MAX_PER_CATEGORY = int(os.getenv("ETF_ROTATION_MULTI_MAX_PER_CATEGORY", "2"))

# ETF Rotation Multi -- Volatility Normalization
ETF_ROTATION_VOL_NORMALIZE = os.getenv("ETF_ROTATION_VOL_NORMALIZE", "true").lower() == "true"
ETF_ROTATION_VOL_LOOKBACK = int(os.getenv("ETF_ROTATION_VOL_LOOKBACK", "60"))

# ETF Rotation Multi -- Category-Specific Risk Overrides
# Based on volatility stress test (Phase 4.1): -10% HWM stop triggers on 50-83% of
# rolling windows for equity sector ETFs. Category-specific stops reduce noise exits.
# Format: JSON dict with category -> {"hwm_stop_pct": N}
# Example: {"Tech": {"hwm_stop_pct": 15}, "Defensive": {"hwm_stop_pct": 8}}
ETF_ROTATION_CATEGORY_RISK_OVERRIDES = os.getenv("ETF_ROTATION_CATEGORY_RISK_OVERRIDES", "")

# --- PE-Band Dynamic Allocation ---
PE_BAND_ENABLED = os.getenv("PE_BAND_ENABLED", "false").lower() == "true"
PE_BAND_SOURCE_INDEX = os.getenv("PE_BAND_SOURCE_INDEX", "000300.SH")
PE_BAND_WINDOW_YEARS = int(os.getenv("PE_BAND_WINDOW_YEARS", "10"))
PE_BAND_EQUITY_MAX = float(os.getenv("PE_BAND_EQUITY_MAX", "0.80"))
PE_BAND_EQUITY_MIN = float(os.getenv("PE_BAND_EQUITY_MIN", "0.20"))
PE_BAND_EQUITY_ETF = os.getenv("PE_BAND_EQUITY_ETF", "510300")
PE_BAND_BOND_ETF = os.getenv("PE_BAND_BOND_ETF", "511880")
PE_BAND_GOLD_ETF = os.getenv("PE_BAND_GOLD_ETF", "518880")
PE_BAND_GOLD_FIXED_PCT = float(os.getenv("PE_BAND_GOLD_FIXED_PCT", "0.10"))

# --- Strategy: ETF Trend (MA / ADX) ---
MA_FAST = int(os.getenv("MA_FAST", "20"))
MA_SLOW = int(os.getenv("MA_SLOW", "60"))
ADX_THRESHOLD = int(os.getenv("ADX_THRESHOLD", "20"))

# --- Capital Allocation ---
TOTAL_CAPITAL = float(os.getenv("TOTAL_CAPITAL", "100000"))
TRADING_CAPITAL = float(os.getenv("TRADING_CAPITAL", "90000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))

# --- Execution ---
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))
COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.00025"))
ORDER_TIMEOUT_SECONDS = int(os.getenv("ORDER_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# --- Risk ---
MAX_SINGLE_DAY_DRAWDOWN_PCT = float(os.getenv("MAX_SINGLE_DAY_DRAWDOWN_PCT", "0.02"))
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "3"))
DUPLICATE_WINDOW_SECONDS = int(os.getenv("DUPLICATE_WINDOW_SECONDS", "60"))

# --- Walk-Forward Search ---
WF_MA_FAST_RANGE = [int(x) for x in os.getenv("WF_MA_FAST_RANGE", "5,15,30").split(",")]
WF_MA_SLOW_RANGE = [int(x) for x in os.getenv("WF_MA_SLOW_RANGE", "20,60,120").split(",")]
WF_ADX_RANGE = [int(x) for x in os.getenv("WF_ADX_RANGE", "10,20,30").split(",")]

# --- Risk (extended) ---
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.0"))
MONTHLY_MAX_DRAWDOWN_PCT = float(os.getenv("MONTHLY_MAX_DRAWDOWN_PCT", "0.05"))
CONSECUTIVE_LOSS_LIMIT = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "5"))

# --- State ---
DB_PATH = os.getenv("DB_PATH", "C:/study/AIWorkspace/quanti/data/quanti.db")
JOURNAL_RETENTION_DAYS = int(os.getenv("JOURNAL_RETENTION_DAYS", "90"))

# --- Logging ---
LOG_DIR = os.getenv("LOG_DIR", "C:/study/AIWorkspace/quanti/logs")
LOG_MAX_SIZE_MB = int(os.getenv("LOG_MAX_SIZE_MB", "10"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
WECHAT_WEBHOOK_URL = os.getenv("WECHAT_WEBHOOK_URL", "")

# --- Strategy: Multi-Indicator Entry (weighted scoring) ---
MA_LONG = int(os.getenv("MA_LONG", "120"))
BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
BB_STD = float(os.getenv("BB_STD", "2.0"))
VOLUME_SURGE_MULTIPLIER = float(os.getenv("VOLUME_SURGE_MULTIPLIER", "1.5"))
ADX_ENTRY_THRESHOLD = int(os.getenv("ADX_ENTRY_THRESHOLD", "25"))
DI_DIFF_THRESHOLD = int(os.getenv("DI_DIFF_THRESHOLD", "15"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", "80"))

# Weighted scoring weights (must sum to ~1.0)
ENTRY_SCORE_THRESHOLD = int(os.getenv("ENTRY_SCORE_THRESHOLD", "55"))
ENTRY_WEIGHT_MA  = float(os.getenv("ENTRY_WEIGHT_MA", "0.25"))
ENTRY_WEIGHT_ADX = float(os.getenv("ENTRY_WEIGHT_ADX", "0.25"))
ENTRY_WEIGHT_BB  = float(os.getenv("ENTRY_WEIGHT_BB", "0.20"))
ENTRY_WEIGHT_VOL = float(os.getenv("ENTRY_WEIGHT_VOL", "0.20"))
ENTRY_WEIGHT_MKT = float(os.getenv("ENTRY_WEIGHT_MKT", "0.10"))

# --- Strategy: Exit Logic ---
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_TRAILING_MULTIPLIER = float(os.getenv("ATR_TRAILING_MULTIPLIER", "2.0"))
ATR_TIGHTEN_MULTIPLIER = float(os.getenv("ATR_TIGHTEN_MULTIPLIER", "1.5"))
ATR_TRAILING_STOP_ENABLED = os.getenv("ATR_TRAILING_STOP_ENABLED", "true").lower() == "true"
TIME_STOP_ENABLED = os.getenv("TIME_STOP_ENABLED", "false").lower() == "true"
VOLATILITY_STOP_ENABLED = os.getenv("VOLATILITY_STOP_ENABLED", "false").lower() == "true"
RSI_EXIT_ENABLED = os.getenv("RSI_EXIT_ENABLED", "false").lower() == "true"
TIME_STOP_DAYS_REDUCE = int(os.getenv("TIME_STOP_DAYS_REDUCE", "40"))
TIME_STOP_DAYS_EXIT = int(os.getenv("TIME_STOP_DAYS_EXIT", "60"))

# --- Market Environment Filter ---
MARKET_ADX_THRESHOLD = int(os.getenv("MARKET_ADX_THRESHOLD", "20"))
DEFENSE_MODE_POSITION_PCT = float(os.getenv("DEFENSE_MODE_POSITION_PCT", "0.20"))
INDEX_SMA_LONG = int(os.getenv("INDEX_SMA_LONG", "120"))

# --- Market-Structure-Aware Defenses ---

# Gap Risk: detect positions at risk of gapping through stop-loss (T+1 trap)
GAP_RISK_CHECK_ENABLED = os.getenv("GAP_RISK_CHECK_ENABLED", "true").lower() == "true"
GAP_RISK_THRESHOLD_PCT = float(os.getenv("GAP_RISK_THRESHOLD_PCT", "5.0"))

# National Team Intervention Detection
NT_INTERVENTION_DETECTION_ENABLED = os.getenv("NT_INTERVENTION_DETECTION_ENABLED", "true").lower() == "true"
NT_VOLUME_SIGMA_THRESHOLD = float(os.getenv("NT_VOLUME_SIGMA_THRESHOLD", "3.0"))
NT_VOLUME_LOOKBACK_DAYS = int(os.getenv("NT_VOLUME_LOOKBACK_DAYS", "60"))
NT_POLICY_EXIT_TIGHTEN = float(os.getenv("NT_POLICY_EXIT_TIGHTEN", "0.7"))

# T+1 Settlement
SETTLEMENT_LAG_DAYS = int(os.getenv("SETTLEMENT_LAG_DAYS", "1"))

"""
run_daily.py -- Production Operations Layer for v6 PE-Band
===========================================================
Daily automated workflow: fetch data, health check, compute signal,
log history, reconcile positions, generate report, send alerts.

Architecture:
    v6_pe_band.py    = pure strategy engine (signals, backtest, verify)
    run_daily.py     = operations (fetch, health, reconcile, alert, report)

Usage:
    python scripts/run_daily.py                  # Full daily pipeline
    python scripts/run_daily.py --no-fetch        # Skip data fetch, use cached data
    python scripts/run_daily.py --signal-only     # Print signal as JSON, nothing else
    python scripts/run_daily.py --reconcile        # Reconcile vs config/daily.json portfolio
    python scripts/run_daily.py --health-only      # Health check only
    python scripts/run_daily.py --setup            # First-time configuration wizard

Schedule:
    Windows Task Scheduler: daily at 19:00 CST (after market close)
    Cron: 0 11 * * 1-5  (11:00 UTC = 19:00 CST weekdays)
"""
import pandas as pd, numpy as np, os, sys, json, argparse, logging, time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

# ─── Paths ───
PROJECT = Path(r"C:\study\AIWorkspace\quanti")
SCRIPTS = PROJECT / "scripts"
DATA_DIR = PROJECT / "data"
MACRO_DIR = DATA_DIR / "macro"
SIGNAL_LOG_PATH = DATA_DIR / "signal_log.jsonl"
TRADE_LOG_PATH = DATA_DIR / "trade_log.jsonl"
ALERT_LOG_PATH = DATA_DIR / "alerts.jsonl"
REPORT_DIR = DATA_DIR / "reports"
LOG_DIR = PROJECT / "logs"
CONFIG_PATH = PROJECT / "config" / "daily.json"

for d in [LOG_DIR, REPORT_DIR, CONFIG_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run_daily.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("run_daily")

# ─── Load v6 Strategy ───
import importlib.util
_v6_spec = importlib.util.spec_from_file_location("v6_pe_band", str(SCRIPTS / "v6_pe_band.py"))
v6 = importlib.util.module_from_spec(_v6_spec)
_v6_spec.loader.exec_module(v6)

# ─── Default Config ───
DEFAULT_CONFIG = {
    "portfolio": {"510300": 0, "518880": 0, "511010": 0, "511880": 90000},
    "notifications": {
        "enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "email_smtp_host": "",
        "email_smtp_port": 587,
        "email_user": "",
        "email_password": "",
        "email_to": "",
        "alert_on_signal_change": True,
        "alert_on_data_stale": True,
        "alert_on_trade_required": True,
    },
    "rebalance": {
        "frequency": "monthly",
        "day_of_month": 2,  # T+2
        "min_trade_threshold_pct": 0.01,  # 1% of portfolio
    },
    "risk": {
        "max_single_position_pct": 0.90,
        "max_trade_size_pct": 0.50,
        "require_confirmation_for_large_trades": True,
        "large_trade_threshold_pct": 0.30,
    },
    "health": {
        "max_staleness": {"ETF": 3, "PE": 7, "PMI": 45, "CGB": 3},
        "pe_valid_range": [5, 60],
        "gold_ma_boundary_threshold": 0.005,
    }
}

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

CFG = load_config()

# ═══════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════

def _bypass_proxy():
    for k in list(os.environ.keys()):
        if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE"):
            os.environ.pop(k, None)

def fetch_etf_data() -> Dict[str, int]:
    """Fetch all 7 ETFs from AkShare. Returns {ticker: new_bars_added}."""
    _bypass_proxy()
    from quanti.data.ingestion.akshare_fetcher import AkShareETFetcher
    from quanti.data.storage import DataStorage
    fetcher = AkShareETFetcher(); storage = DataStorage()
    results = {}
    for code in ["510300","510500","159915","510880","518880","511010","511880"]:
        prefix = "sh" if code.startswith(("51", "58", "56", "60")) else "sz"
        try:
            bars = fetcher.fetch_daily(prefix + code)
            if not bars: results[code] = 0; continue
            fp = storage.clean_dir / f"{code}.parquet"
            old_cnt = len(pd.read_parquet(fp)) if fp.exists() else 0
            storage.save_bars_clean(code, bars)
            new_cnt = len(pd.read_parquet(fp)); results[code] = new_cnt - old_cnt
            log.info(f"  {code}: +{results[code]} bars ({old_cnt}->{new_cnt})")
        except Exception as e:
            log.warning(f"  {code}: FETCH FAILED -- {e}"); results[code] = -1
    return results

def fetch_pe_data() -> int:
    """Fetch CSI300 PE from AkShare. Returns row count."""
    _bypass_proxy()
    try:
        import akshare as ak
        df = ak.stock_index_pe_lg(symbol="沪深300")
        df["date"] = pd.to_datetime(df.iloc[:, 0].astype(str).str.replace("-", ""), format="%Y%m%d")
        df["pe"] = df.iloc[:, 6].astype(float)
        df = df[["date", "pe"]].dropna(); df = df[df["pe"] > 0].sort_values("date")
        MACRO_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(MACRO_DIR / "csi300_pe.parquet", index=False)
        log.info(f"  PE: {len(df)} rows, last {df['date'].iloc[-1].date()}")
        return len(df)
    except Exception as e:
        log.error(f"  PE: FETCH FAILED -- {e}"); return 0

def fetch_macro_data() -> bool:
    """Fetch PMI + CGB yield."""
    _bypass_proxy()
    try:
        import importlib.util
        fm_spec = importlib.util.spec_from_file_location("fetch_macro", str(SCRIPTS / "fetch_macro.py"))
        fm = importlib.util.module_from_spec(fm_spec)
        fm_spec.loader.exec_module(fm)
        fm.fetch_all()
        log.info("  PMI + CGB yield fetched")
        return True
    except Exception as e:
        log.error(f"  PMI/CGB: FETCH FAILED -- {e}"); return False

def fetch_all() -> Tuple[bool, list]:
    """Fetch all data and reload strategy. Returns (ok, failures)."""
    failures = []
    for label, fn in [("ETFs", fetch_etf_data), ("PE", fetch_pe_data), ("Macro", fetch_macro_data)]:
        log.info(f"Fetching {label}...")
        r = fn()
        if isinstance(r, dict):
            fails = [k for k, v in r.items() if v < 0]
            if fails: failures.extend(f"{label}/{k}" for k in fails)
        elif r == 0:
            failures.append(label)
    v6.reload_data()
    return len(failures) == 0, failures

# ═══════════════════════════════════════════════════════════════════
# HEALTH CHECKS
# ═══════════════════════════════════════════════════════════════════

def health_check() -> Tuple[bool, list, list]:
    """Returns (ok, issues, warnings)."""
    v6.reload_data()
    issues, warnings = [], []
    hcfg = CFG["health"]
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    # ETF staleness
    for e in v6.ETFS:
        if e not in v6.T: issues.append(f"MISSING ETF {e}"); continue
        age = (today - v6.T[e].index[-1]).days
        if age > hcfg["max_staleness"]["ETF"]:
            issues.append(f"STALE {e}: {age}d old (max {hcfg['max_staleness']['ETF']}d)")
    # PE staleness + value validation
    if v6.pe_raw is not None and len(v6.pe_raw) > 0:
        age = (today - v6.pe_raw.index[-1]).days
        if age > hcfg["max_staleness"]["PE"]:
            issues.append(f"STALE PE: {age}d old")
        last_pe = float(v6.pe_raw["pe"].iloc[-1])
        lo, hi = hcfg["pe_valid_range"]
        if last_pe < lo or last_pe > hi:
            issues.append(f"PE OUT OF RANGE: {last_pe:.1f} (expected {lo}-{hi})")
    else:
        issues.append("MISSING PE data")
    # PMI staleness
    pmip = MACRO_DIR / "caixin_pmi.parquet"
    if pmip.exists():
        pmi = pd.read_parquet(pmip)
        if "date" in pmi.columns: pmi = pmi.set_index("date")
        if len(pmi) > 0:
            age = (today - pd.to_datetime(pmi.index[-1])).days
            if age > hcfg["max_staleness"]["PMI"]:
                warnings.append(f"STALE PMI: {age}d old (not used by v6)")
    else:
        warnings.append("MISSING PMI (not used by v6)")
    # CGB staleness
    cgbp = MACRO_DIR / "cgb_10y_yield.parquet"
    if cgbp.exists():
        cgb = pd.read_parquet(cgbp)
        if "date" in cgb.columns: cgb = cgb.set_index("date")
        if len(cgb) > 0:
            age = (today - pd.to_datetime(cgb.index[-1])).days
            if age > hcfg["max_staleness"]["CGB"]:
                warnings.append(f"STALE CGB: {age}d old (not used by v6)")
    # Gold MA50 boundary check
    if "518880" in v6.T:
        c = v6.T["518880"]["close"]
        if len(c) >= 51:
            dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in v6.T.values()])))
            latest = dr[dr <= today][-1] if len(dr[dr <= today]) > 0 else today
            if latest in c.index:
                ma50 = c.rolling(50).mean()
                dist = abs(float(c.loc[latest]) - float(ma50.loc[latest])) / float(ma50.loc[latest]) if ma50.loc[latest] != 0 else 0
                if dist < hcfg["gold_ma_boundary_threshold"]:
                    warnings.append(f"Gold within {dist*100:.1f}% of MA50 -- signal may be borderline at execution")
    return len(issues) == 0, issues, warnings

# ═══════════════════════════════════════════════════════════════════
# SIGNAL
# ═══════════════════════════════════════════════════════════════════

def compute_signal() -> dict:
    """Returns signal dict with targets, PE percentile, regime, gold state."""
    v6.reload_data()
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in v6.T.values()])))
    latest = dr[dr <= today][-1] if len(dr[dr <= today]) > 0 else today
    pp = v6.pe_pct_at(latest)
    eq_pct = 0.60 - pp * (0.60 - 0.10)
    eq_pct = max(0.10, min(0.60, eq_pct))
    g_pct = 0.30 if v6.trend(v6.GOLD, latest, 50) else 0.0
    bd_pct = max(0.0, 1.0 - eq_pct - g_pct)
    return {
        "date": str(latest.date()),
        "signal_ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "targets": {"510300": round(eq_pct, 4), "518880": round(g_pct, 4), "511010": round(bd_pct, 4)},
        "pe_percentile": round(float(pp), 4),
        "pe_value": round(float(v6.pe_pct_at(latest)), 1) if v6.pe_raw is not None else None,
        "gold_trending": bool(v6.trend(v6.GOLD, latest, 50)),
        "market_regime": "expensive" if pp > 0.75 else "fair" if pp > 0.25 else "cheap",
    }

def log_signal(signal: dict):
    with open(SIGNAL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(signal, ensure_ascii=False) + "\n")

def signal_history(n: int = 60) -> list:
    if not SIGNAL_LOG_PATH.exists(): return []
    signals = []
    with open(SIGNAL_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                signals.append(json.loads(line))
    return signals[-n:]

def signal_changed(new_signal: dict, threshold: float = 0.05) -> bool:
    history = signal_history(2)
    if len(history) < 1: return True
    last = history[-1]["targets"]
    return any(abs(new_signal["targets"].get(k, 0) - last.get(k, 0)) > threshold for k in last)

# ═══════════════════════════════════════════════════════════════════
# REBALANCE SCHEDULE
# ═══════════════════════════════════════════════════════════════════

def next_rebalance_date() -> pd.Timestamp:
    """Next T+2 rebalance date (second trading day of next month)."""
    dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in v6.T.values()])))
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    nm = today.month % 12 + 1; ny = today.year + (1 if today.month == 12 else 0)
    try:
        md = dr[(dr.year == ny) & (dr.month == nm)]
        return md[1] if len(md) >= 2 else (md[0] if len(md) >= 1 else pd.Timestamp(f"{ny}-{nm:02d}-03"))
    except:
        return pd.Timestamp(f"{ny}-{nm:02d}-03")

def days_to_rebalance() -> int:
    return (next_rebalance_date() - pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))).days

# ═══════════════════════════════════════════════════════════════════
# RECONCILIATION
# ═══════════════════════════════════════════════════════════════════

def reconcile(signal: dict) -> Tuple[list, float]:
    """Compare signal targets to config portfolio. Returns (trades, turnover_pct)."""
    portfolio = CFG.get("portfolio", {"511880": 90000})
    # Compute total portfolio value
    total = 0.0
    for e, val in portfolio.items():
        if e == "511880":
            total += val
        elif e in v6.T:
            dr = v6.T[e].index
            latest = dr[-1] if len(dr) > 0 else None
            if latest:
                total += val * float(v6.T[e]["close"].loc[latest]) if isinstance(val, (int, float)) and val > 0 and val < 1e6 else val
    if total <= 0: return [], 0.0

    trades = []
    for ticker, target_pct in signal["targets"].items():
        current_val = 0.0
        if ticker in portfolio:
            cv = portfolio[ticker]
            if isinstance(cv, (int, float)):
                current_val = cv if ticker == "511880" else cv
        target_val = total * target_pct
        diff = target_val - current_val
        if abs(diff) > total * CFG["rebalance"]["min_trade_threshold_pct"]:
            side = "BUY" if diff > 0 else "SELL"
            trades.append({
                "ticker": ticker, "side": side, "amount": round(abs(diff), 2),
                "pct_of_portfolio": round(abs(diff) / total, 4),
                "target_pct": round(target_pct, 4),
            })
    turnover = sum(t["pct_of_portfolio"] for t in trades) / 2
    return trades, turnover

def log_trade(trade: dict):
    trade["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(trade, ensure_ascii=False) + "\n")

# ═══════════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════════

def log_alert(level: str, message: str, metadata: dict = None):
    alert = {"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "level": level, "msg": message, "meta": metadata or {}}
    with open(ALERT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert, ensure_ascii=False) + "\n")
    log.info(f"ALERT [{level}]: {message}")

def send_telegram(message: str) -> bool:
    cfg = CFG["notifications"]
    token = cfg.get("telegram_bot_token"); cid = cfg.get("telegram_chat_id")
    if not token or not cid: return False
    try:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                         json={"chat_id": cid, "text": message, "parse_mode": "Markdown"}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.error(f"Telegram failed: {e}"); return False

def send_email(subject: str, body: str) -> bool:
    cfg = CFG["notifications"]
    host = cfg.get("email_smtp_host"); user = cfg.get("email_user")
    pwd = cfg.get("email_password"); to = cfg.get("email_to")
    if not all([host, user, pwd, to]): return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject; msg["From"] = user; msg["To"] = to
        with smtplib.SMTP(host, cfg.get("email_smtp_port", 587)) as s:
            s.starttls(); s.login(user, pwd); s.send_message(msg)
        return True
    except Exception as e:
        log.error(f"Email failed: {e}"); return False

def notify(signal: dict, issues: list, warnings: list, trades: list):
    cfg = CFG["notifications"]
    if not cfg.get("enabled"): return

    if signal_changed(signal) and cfg.get("alert_on_signal_change", True):
        msg = (
            f"*v6 Signal* — {signal['date']}\n"
            f"PE: {signal['pe_percentile']*100:.0f}th pctile | Gold: {'ON' if signal['gold_trending'] else 'OFF'}\n"
            f"CSI300: {signal['targets']['510300']*100:.0f}% | Gold: {signal['targets']['518880']*100:.0f}% | Bonds: {signal['targets']['511010']*100:.0f}%"
        )
        send_telegram(msg)
        log_alert("INFO", "Signal changed", signal)

    if issues and cfg.get("alert_on_data_stale", True):
        send_telegram(f"*v6 ISSUES* — {len(issues)} problem(s)\n" + "\n".join(f"- {i}" for i in issues[:5]))
        log_alert("WARNING", "Health issues", {"issues": issues})

    large = [t for t in trades if t["pct_of_portfolio"] > CFG["risk"]["large_trade_threshold_pct"]]
    if large and cfg.get("alert_on_trade_required", True):
        send_telegram(f"*v6 Large Trade*\n" + "\n".join(f"{t['side']} {t['ticker']}: {t['pct_of_portfolio']*100:.0f}%" for t in large))
        log_alert("INFO", "Large trade required", {"trades": large})

# ═══════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════

def generate_report(signal: dict, issues: list, warnings: list, trades: list) -> Path:
    """Generate daily markdown report. Returns path."""
    v6.reload_data()
    today = datetime.now().strftime("%Y-%m-%d")
    path = REPORT_DIR / f"v6_report_{today}.md"
    days = days_to_rebalance(); nxt = next_rebalance_date()
    ytd = v6.backtest(f"{datetime.now().year}-01-01", today)

    lines = [
        f"# v6 PE-Band Report — {today}",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        f"## Status: {'ALL CLEAR' if not issues and not warnings else f'{len(issues)} ISSUES, {len(warnings)} WARNINGS'}",
    ]
    if issues:
        lines.append("### Issues"); lines.extend(f"- [!] {i}" for i in issues)
    if warnings:
        lines.append("### Warnings"); lines.extend(f"- [!] {w}" for w in warnings)
    lines.extend([
        "", "## Signal",
        f"- Date: {signal['date']} | PE: {signal['pe_percentile']*100:.0f}th pctile | Regime: {signal['market_regime']}",
        f"- Gold: {'TRENDING' if signal['gold_trending'] else 'not trending'}",
        "", "| ETF | Ticker | Target |",
        "|-----|--------|--------|",
        f"| CSI300 | 510300 | {signal['targets']['510300']*100:.0f}% |",
        f"| Gold | 518880 | {signal['targets']['518880']*100:.0f}% |",
        f"| CGB Bonds | 511010 | {signal['targets']['511010']*100:.0f}% |",
        "", "## Rebalance",
        f"- Next: {nxt.date()} ({days} days)",
        f"- YTD Return: {ytd['total']:.2%} | YTD MaxDD: {ytd['maxdd']:.2%}",
    ])
    if trades:
        lines.extend(["", "## Required Trades", "| Ticker | Side | Amount | % Portfolio |",
                      "|--------|------|--------|-------------|"])
        for t in trades:
            lines.append(f"| {t['ticker']} | {t['side']} | {t['amount']:,.0f} | {t['pct_of_portfolio']*100:.0f}% |")
    lines.extend(["", "## 7-Day PE Trend"])
    if v6.pe_raw is not None and len(v6.pe_raw) >= 7:
        for idx, row in v6.pe_raw.iloc[-7:].iterrows():
            lines.append(f"- {idx.date()}: PE={row['pe']:.1f} ({row.get('pe_pct',0.5)*100:.0f}th)")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Report: {path}")
    return path

# ═══════════════════════════════════════════════════════════════════
# SETUP WIZARD
# ═══════════════════════════════════════════════════════════════════

def setup():
    print("=" * 60)
    print("  v6 PE-Band — First-Time Setup")
    print("=" * 60)
    cfg = DEFAULT_CONFIG.copy()
    try:
        amt = float(input("\nPortfolio value (RMB, default 90000): ") or "90000")
    except:
        amt = 90000
    cfg["portfolio"] = {"511880": amt, "510300": 0, "518880": 0, "511010": 0}
    print(f"Portfolio: {amt:,.0f} RMB (starting all cash)")

    en = input("\nEnable notifications? (y/N): ").lower().startswith("y")
    cfg["notifications"]["enabled"] = en
    if en:
        tg = input("Telegram bot token (or Enter to skip): ")
        if tg:
            cfg["notifications"]["telegram_bot_token"] = tg
            cfg["notifications"]["telegram_chat_id"] = input("Telegram chat ID: ")
            print("Telegram configured.")
        elif input("Configure email? (y/N): ").lower().startswith("y"):
            cfg["notifications"]["email_smtp_host"] = input("SMTP host: ")
            cfg["notifications"]["email_user"] = input("SMTP user: ")
            cfg["notifications"]["email_password"] = input("SMTP password: ")
            cfg["notifications"]["email_to"] = input("Alert recipient email: ")
            print("Email configured.")

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"\nConfig saved: {CONFIG_PATH}")
    print("Next: python scripts/run_daily.py")
    return cfg

# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main(fetch: bool = True, reconcile_portfolio: bool = True,
         do_notify: bool = True, do_report: bool = True) -> dict:
    """Execute the full daily pipeline. Returns result dict."""
    t0 = time.time()
    log.info(f"Daily pipeline started — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    issues, warnings, trades, signal = [], [], [], None

    # 1. Fetch
    if fetch:
        log.info("[1/6] Fetching data...")
        ok, fails = fetch_all()
        if not ok: issues.extend(f"Fetch failure: {x}" for x in fails)
    else:
        log.info("[1/6] Skipping fetch (--no-fetch)")

    # 2. Health
    log.info("[2/6] Health check...")
    ok, hi, hw = health_check()
    issues.extend(hi); warnings.extend(hw)

    # 3. Signal
    log.info("[3/6] Computing signal...")
    try:
        signal = compute_signal()
        log_signal(signal)
        log.info(f"  {signal['date']}: CSI300={signal['targets']['510300']*100:.0f}% Gold={signal['targets']['518880']*100:.0f}% Bonds={signal['targets']['511010']*100:.0f}%")
    except Exception as e:
        log.error(f"Signal failed: {e}"); issues.append(f"Signal computation error: {e}")

    # 4. Reconcile
    if signal and reconcile_portfolio:
        log.info("[4/6] Reconciling...")
        trades, to = reconcile(signal)
        for t in trades:
            if t["pct_of_portfolio"] > CFG["risk"]["max_trade_size_pct"]:
                warnings.append(f"Trade exceeds max size: {t['ticker']} {t['side']} {t['pct_of_portfolio']*100:.0f}%")
            if t["pct_of_portfolio"] > CFG["risk"]["large_trade_threshold_pct"] and CFG["risk"]["require_confirmation_for_large_trades"]:
                warnings.append(f"Large trade needs manual confirmation: {t['ticker']} {t['side']} {t['pct_of_portfolio']*100:.0f}%")
        if trades: log.info(f"  {len(trades)} trades, {to:.1%} turnover")
        else: log.info("  No material trades")

    # 5. Report
    if signal and do_report:
        log.info("[5/6] Generating report...")
        try: generate_report(signal, issues, warnings, trades)
        except Exception as e: log.error(f"Report failed: {e}")

    # 6. Notify
    if signal and do_notify:
        log.info("[6/6] Notifications...")
        notify(signal, issues, warnings, trades)

    elapsed = time.time() - t0
    log.info(f"Pipeline complete — {elapsed:.1f}s")
    if issues: log.warning(f"  {len(issues)} issue(s)")
    if warnings: log.warning(f"  {len(warnings)} warning(s)")
    if not issues and not warnings: log.info("  ALL CLEAR")
    return {"signal": signal, "issues": issues, "warnings": warnings, "trades": trades, "elapsed": elapsed, "ok": len(issues) == 0}

# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v6 PE-Band Daily Operations")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--signal-only", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.health_only:
        ok, issues, warnings = health_check()
        print(f"Health: {'OK' if ok else f'{len(issues)} ISSUES'}")
        for i in issues: print(f"  ISSUE: {i}")
        for w in warnings: print(f"  WARN: {w}")
        if not issues and not warnings: print("  ALL CLEAR")
    elif args.signal_only:
        s = compute_signal()
        log_signal(s)
        print(json.dumps(s, indent=2, ensure_ascii=False))
    else:
        r = main(fetch=not args.no_fetch, reconcile_portfolio=args.reconcile)
        if r["signal"]:
            s = r["signal"]
            print(f"\nSignal: CSI300={s['targets']['510300']*100:.0f}% Gold={s['targets']['518880']*100:.0f}% Bonds={s['targets']['511010']*100:.0f}%")
        print(f"\n{'Issues' if r['issues'] else 'OK'}: {r['elapsed']:.1f}s")

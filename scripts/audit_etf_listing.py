"""
Audit script: fetch earliest available data for all 25 ETFs in the multi-industry pool.
Outputs: data/etf_listing_audit.json
"""
import json
import os
import sys
import time
from pathlib import Path

# Ensure we can import quanti
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import akshare as ak

# ── ETF Pool: 25 ETFs across 7 categories ──
ETF_POOL = {
    "Broad":     ["510300", "510500", "159915", "588000"],
    "Finance":   ["512000", "512800"],
    "Tech":      ["512480", "515070", "515880", "512720"],
    "NewEnergy": ["516160", "516880", "516110"],
    "Consumer":  ["159928", "512010"],
    "Resources": ["159825", "516810", "516310", "516320"],
    "TMT":       ["512980", "159869"],
    "Defense":   ["512660"],
    "Defensive": ["510880", "518880", "511880"],
}

ALL_ETFS = [sym for syms in ETF_POOL.values() for sym in syms]


def bypass_proxy():
    """Remove proxy env vars that block akshare from reaching Chinese APIs."""
    for key in list(os.environ.keys()):
        if key.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE"):
            os.environ.pop(key, None)


def to_sina_symbol(symbol: str) -> str:
    """Convert bare ETF code to Sina symbol format (shXXXXXX or szXXXXXX)."""
    s = symbol.replace(".SH", "").replace(".SZ", "")
    prefix = "sh" if s.startswith(("51", "60")) else "sz"
    return prefix + s


def fetch_and_audit(symbol: str, rate_limit: float = 2.0) -> dict:
    """Fetch full history for one ETF and return audit record."""
    sina_sym = to_sina_symbol(symbol)
    print(f"  Fetching {symbol} ({sina_sym}) ...", end=" ", flush=True)

    try:
        df = ak.fund_etf_hist_sina(symbol=sina_sym)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return {
            "symbol": symbol,
            "sina_symbol": sina_sym,
            "status": "error",
            "error": str(exc),
        }

    if df is None or df.empty:
        print("EMPTY")
        return {
            "symbol": symbol,
            "sina_symbol": sina_sym,
            "status": "empty",
            "trading_days": 0,
        }

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    first_date = df["date"].iloc[0]
    last_date = df["date"].iloc[-1]
    trading_days = len(df)

    # Detect gaps: consecutive trading days > 5 calendar days apart
    dates = sorted(df["date"].tolist())
    gaps = []
    for i in range(1, len(dates)):
        delta = (dates[i] - dates[i - 1]).days
        if delta > 5:
            gaps.append({
                "from": str(dates[i - 1]),
                "to": str(dates[i]),
                "gap_days": delta,
            })

    print(f"OK  {first_date} -> {last_date}  ({trading_days} bars{', ' + str(len(gaps)) + ' gaps' if gaps else ''})")

    # Rate limit
    time.sleep(rate_limit)

    return {
        "symbol": symbol,
        "sina_symbol": sina_sym,
        "status": "ok",
        "first_date": str(first_date),
        "last_date": str(last_date),
        "trading_days": trading_days,
        "gaps": gaps,
    }


def main():
    bypass_proxy()

    print("=" * 60)
    print("ETF Listing Date Audit: 25-ETF Multi-Industry Pool")
    print("=" * 60)
    print(f"Total ETFs to audit: {len(ALL_ETFS)}")
    print()

    results = {}
    for i, sym in enumerate(ALL_ETFS, 1):
        print(f"[{i:2d}/{len(ALL_ETFS)}]", end=" ")
        results[sym] = fetch_and_audit(sym)
        # Longer pause every 10 calls
        if i % 10 == 0 and i < len(ALL_ETFS):
            print("  (cool-down 3s...)")
            time.sleep(3)

    # ── Summary ──
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ok_count = sum(1 for r in results.values() if r["status"] == "ok")
    err_count = sum(1 for r in results.values() if r["status"] != "ok")

    print(f"  Successful: {ok_count}/{len(ALL_ETFS)}")
    print(f"  Failed:     {err_count}/{len(ALL_ETFS)}")

    if ok_count > 0:
        earliest = min(r["first_date"] for r in results.values() if r["status"] == "ok")
        latest_first = max(r["first_date"] for r in results.values() if r["status"] == "ok")
        min_days = min(r["trading_days"] for r in results.values() if r["status"] == "ok")
        max_days = max(r["trading_days"] for r in results.values() if r["status"] == "ok")

        print(f"  Earliest data: {earliest}")
        print(f"  Latest first date: {latest_first}")
        print(f"  Trading days range: {min_days} - {max_days}")

        # Flag short-history ETFs
        print()
        print("  ETFs with <252 trading days (insufficient for enrollment):")
        short = [r for r in results.values() if r["status"] == "ok" and r["trading_days"] < 252]
        if short:
            for r in short:
                print(f"    {r['symbol']}: {r['trading_days']} days  (first: {r['first_date']})")
        else:
            print("    None -- all ETFs have >=252 trading days")

        # Flag ETFs with gaps
        print()
        print("  ETFs with data gaps >=5 days:")
        gapped = [r for r in results.values() if r["status"] == "ok" and r.get("gaps")]
        if gapped:
            for r in gapped:
                print(f"    {r['symbol']}: {len(r['gaps'])} gaps")
                for g in r["gaps"][:3]:
                    print(f"      {g['from']} -> {g['to']} ({g['gap_days']}d)")
        else:
            print("    None -- all ETFs have continuous data")

    if err_count > 0:
        print()
        print("  FAILED FETCHES:")
        for r in results.values():
            if r["status"] != "ok":
                print(f"    {r['symbol']}: {r['status']} -- {r.get('error', '')}")

    # ── Save to JSON ──
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "etf_listing_audit.json"

    audit_output = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_etfs": len(ALL_ETFS),
        "categories": ETF_POOL,
        "etfs": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_output, f, indent=2, ensure_ascii=False, default=str)

    print()
    print(f"Audit saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()

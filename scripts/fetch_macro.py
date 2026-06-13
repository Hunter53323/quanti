"""Macro data fetching for ETF Rotation v6.

Usage:
    python scripts/fetch_macro.py               # full: fetch + save
    python scripts/fetch_macro.py --pmi-only    # only PMI
    python scripts/fetch_macro.py --yield-only  # only CGB yield

Data sources:
    Caixin Manufacturing PMI — ak.macro_china_cx_pmi_yearly()
         fallback: ak.macro_china_pmi() (NBS PMI)
    10Y CGB Yield — ak.bond_zh_us_rate() (China 10Y column)

Output:
    data/macro/caixin_pmi.parquet
    data/macro/cgb_10y_yield.parquet
"""
import os, sys, argparse, warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Remove proxy env vars that may block HTTPS
for k in list(os.environ.keys()):
    if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE"):
        os.environ.pop(k, None)

MACRO_DIR = Path(PROJECT_ROOT) / "data" / "macro"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir():
    MACRO_DIR.mkdir(parents=True, exist_ok=True)


def _save_parquet(df, path, label):
    """Save DataFrame to parquet, merging with existing data if present."""
    if df is None or df.empty:
        print(f"  WARN  No data to save for {label}")
        return
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"], keep="last")
            df = df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"  WARN  Could not merge existing {label}: {e}")
    df.to_parquet(path, index=False)
    print(f"  OK    Saved {label} -> {path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# PMI
# ---------------------------------------------------------------------------

def fetch_caixin_pmi():
    """Fetch Caixin Manufacturing PMI. Falls back to NBS PMI on failure."""
    # --- Caixin ---
    try:
        import akshare as ak
        df = ak.macro_china_cx_pmi_yearly()
        if df is None or df.empty:
            raise ValueError("empty Caixin response")
        result = pd.DataFrame({
            "date":  pd.to_datetime(df.iloc[:, 1], errors="coerce"),
            "value": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
        }).dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
        print(f"  OK    Caixin PMI -> {len(result)} rows  "
              f"{result['date'].min().strftime('%Y-%m-%d')} ~ "
              f"{result['date'].max().strftime('%Y-%m-%d')}")
        return result
    except Exception as e:
        print(f"  FAIL  Caixin PMI fetch: {e}")

    # --- NBS fallback ---
    try:
        print("  … trying fallback: macro_china_pmi()")
        import akshare as ak
        df = ak.macro_china_pmi()
        if df is None or df.empty:
            raise ValueError("empty NBS response")
        result = pd.DataFrame({
            "date": pd.to_datetime(
                df.iloc[:, 0].astype(str).str.replace("年", "").str.replace("月", "-"),
                errors="coerce"),
            "value": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        }).dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
        print(f"  OK    NBS PMI (fallback) -> {len(result)} rows")
        print("  NOTE  Using NBS PMI (large-SOE coverage, less market-sensitive)")
        return result
    except Exception as e:
        print(f"  FAIL  NBS fallback also failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CGB Yield
# ---------------------------------------------------------------------------

def fetch_cgb_10y_yield():
    """Fetch 10Y China Government Bond yield from akshare."""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate()
        if df is None or df.empty:
            raise ValueError("empty bond response")
        # Columns: date=0, China10Y=3
        result = pd.DataFrame({
            "date":      pd.to_datetime(df.iloc[:, 0],  errors="coerce"),
            "yield_10y": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
        }).dropna(subset=["yield_10y"]).sort_values("date").reset_index(drop=True)
        print(f"  OK    10Y CGB Yield -> {len(result)} rows  "
              f"{result['date'].min().strftime('%Y-%m-%d')} ~ "
              f"{result['date'].max().strftime('%Y-%m-%d')}")
        return result
    except Exception as e:
        print(f"  FAIL  10Y CGB Yield fetch: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_all():
    """Fetch both PMI and CGB yield and save to data/macro/."""
    print("=" * 60)
    print("  FETCH MACRO DATA  —  ETF Rotation v6")
    print("=" * 60)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _ensure_dir()

    pmi = fetch_caixin_pmi()
    _save_parquet(pmi, MACRO_DIR / "caixin_pmi.parquet", "caixin_pmi")
    print()

    cgb = fetch_cgb_10y_yield()
    _save_parquet(cgb, MACRO_DIR / "cgb_10y_yield.parquet", "cgb_10y_yield")

    print(f"\n  Done — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch macro data for ETF rotation")
    parser.add_argument("--pmi-only", action="store_true", help="Only fetch PMI")
    parser.add_argument("--yield-only", action="store_true", help="Only fetch CGB yield")
    args = parser.parse_args()

    _ensure_dir()

    if args.pmi_only:
        pmi = fetch_caixin_pmi()
        _save_parquet(pmi, MACRO_DIR / "caixin_pmi.parquet", "caixin_pmi")
    elif args.yield_only:
        cgb = fetch_cgb_10y_yield()
        _save_parquet(cgb, MACRO_DIR / "cgb_10y_yield.parquet", "cgb_10y_yield")
    else:
        fetch_all()

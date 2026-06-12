"""Convertible bond data ingestion via AkShare."""
import time
from datetime import date
import akshare as ak
from quanti.data.schema import BondDailyBar
from quanti.data.storage import DataStorage


class CBDataFetcher:
    """Fetches convertible bond universe and daily pricing from AkShare."""

    def __init__(self):
        self._last_call = 0.0
        self._min_interval = 0.5

    def _rate_limit(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def fetch_universe(self) -> list[dict]:
        """
        Fetch the current convertible bond universe with fundamentals.
        Uses bond_zh_cov() which returns ~1000 bonds.
        Returns list of {code, name, stock_code, stock_name, stock_price,
                          convert_price, premium_rt, rating, maturity_date}
        """
        self._rate_limit()
        df = ak.bond_zh_cov()

        bonds = []
        # Skip NEEQ/三板 bonds (codes starting with 4) and bonds without essential data
        for _, row in df.iterrows():
            try:
                bond_code = str(row.iloc[0])
                if bond_code.startswith("4"):
                    continue  # Skip NEEQ bonds
                bond_name = str(row.iloc[1])
                stock_code = str(row.iloc[5])
                stock_name = str(row.iloc[6])
                stock_price = float(row.iloc[7]) if row.iloc[7] and not (isinstance(row.iloc[7], float) and str(row.iloc[7]) == 'nan') else 0.0
                convert_price = float(row.iloc[8]) if row.iloc[8] and not (isinstance(row.iloc[8], float) and str(row.iloc[8]) == 'nan') else 0.0
                premium_rt = float(row.iloc[9]) if row.iloc[9] and not (isinstance(row.iloc[9], float) and str(row.iloc[9]) == 'nan') else 0.0
                issue_date = row.iloc[2]
                listing_date = row.iloc[12] if len(df.columns) > 12 else None
                maturity_date = row.iloc[15] if len(df.columns) > 15 else None
                rating = str(row.iloc[18]) if len(df.columns) > 18 and str(row.iloc[18]) != 'nan' else ''

                bonds.append(dict(
                    code=bond_code, name=bond_name,
                    stock_code=stock_code, stock_name=stock_name,
                    stock_price=stock_price, convert_price=convert_price,
                    premium_rt=premium_rt, rating=rating,
                    listing_date=listing_date, maturity_date=maturity_date,
                    issue_date=issue_date,
                ))
            except (ValueError, IndexError):
                continue

        return bonds

    def fetch_daily(self, code: str) -> list[BondDailyBar]:
        """
        Fetch daily OHLCV bars for one convertible bond.
        Uses bond_zh_hs_cov_daily with Shanghai/Shenzhen exchange prefix.
        """
        self._rate_limit()

        symbol = self._to_exchange_symbol(code)
        try:
            df = ak.bond_zh_hs_cov_daily(symbol=symbol)
            if df is None or len(df) == 0:
                return []
            if "date" not in df.columns:
                return []
        except Exception as e:
            raise RuntimeError(f"CB daily fetch failed for {code} ({symbol}): {e}")

        if df is None or df.empty:
            return []

        bars = []
        for _, row in df.iterrows():
            d = row["date"]
            if isinstance(d, date):
                trade_date = d.strftime("%Y%m%d")
            else:
                trade_date = str(d).replace("-", "")

            bars.append(BondDailyBar(
                symbol=code,
                trade_date=trade_date,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                convert_price=None,
                convert_value=None,
                premium_rt=None,
                maturity_date=None,
            ))

        return sorted(bars, key=lambda b: b.trade_date)

    def fetch_with_fundamentals(self, code: str, fundamentals: dict) -> list[BondDailyBar]:
        """Fetch daily bars and enrich with fundamental data."""
        bars = self.fetch_daily(code)
        for b in bars:
            b.convert_price = fundamentals.get("convert_price")
            b.premium_rt = fundamentals.get("premium_rt")
            b.maturity_date = str(fundamentals.get("maturity_date", ""))
            if b.convert_price and b.convert_price > 0 and fundamentals.get("stock_price", 0) > 0:
                b.convert_value = fundamentals["stock_price"] * 100 / b.convert_price
        return bars

    @staticmethod
    def _to_exchange_symbol(code: str) -> str:
        """Convert bond code to exchange symbol. sh=Shanghai (11xxxx), sz=Shenzhen (12xxxx)."""
        if code.startswith("11"):
            return f"sh{code}"
        elif code.startswith("12"):
            return f"sz{code}"
        return f"sh{code}"


def run_cb_ingestion(max_bonds: int = 30, store: DataStorage | None = None):
    """
    Fetch convertible bond universe and daily pricing.
    Returns (bonds_fetched, errors).
    """
    if store is None:
        store = DataStorage()

    fetcher = CBDataFetcher()
    errors = []

    # Fetch universe
    print("Fetching CB universe...")
    try:
        bonds = fetcher.fetch_universe()
        print(f"  Universe: {len(bonds)} bonds")
    except Exception as e:
        return (0, [f"CB universe fetch failed: {e}"])

    # Filter: only bonds with valid premium and price data
    valid = [b for b in bonds if b["premium_rt"] != 0 and b["convert_price"] > 0]
    print(f"  Valid (with premium data): {len(valid)}")

    # Score by dual-low (cheap price + low premium)
    if len(valid) > 10:
        prices = [b["convert_price"] for b in valid]
        premiums = [b["premium_rt"] for b in valid]
        p_min, p_max = min(prices), max(prices)
        pr_min, pr_max = min(premiums), max(premiums)

        for b in valid:
            p_score = 1 - (b["convert_price"] - p_min) / (p_max - p_min + 1e-10)
            pr_score = 1 - (b["premium_rt"] - pr_min) / (pr_max - pr_min + 1e-10)
            b["dual_low_score"] = 0.6 * p_score + 0.4 * pr_score

        valid.sort(key=lambda x: x.get("dual_low_score", 0), reverse=True)

    # Fetch daily bars for top N
    top_n = valid[:max_bonds]
    fetched = 0
    for i, bond in enumerate(top_n):
        try:
            bars = fetcher.fetch_with_fundamentals(bond["code"], bond)
            if bars:
                store.save_bars_raw(f"cb_{bond['code']}", bars)
                store.save_bars_clean(f"cb_{bond['code']}", bars)
                fetched += 1
                if i < 5 or i % 10 == 0:
                    print(f"  [{i+1}/{len(top_n)}] {bond['code']} {bond['name']}: {len(bars)} bars, prem={bond['premium_rt']:.1f}%")
        except Exception as e:
            errors.append(f"CB {bond['code']}: {e}")

    print(f"  Fetched: {fetched}/{len(top_n)}")
    return (fetched, errors)


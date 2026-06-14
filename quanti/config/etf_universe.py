"""
ETF universe definitions with listing-date awareness for dynamic pool management.

ETFs are organized by sector and include their listing dates so newly listed
ETFs can be excluded from the available pool until sufficient price history
exists (120 calendar days -- a proxy for the ~84 trading days needed for MA
computation).
"""

from datetime import datetime, timedelta

# --- Full Universe -----------------------------------------------------------

ETF_UNIVERSE_MULTI: list[dict] = [
    # 宽基 (Broad-based)
    {"code": "510300", "name": "沪深300", "sector": "宽基", "list_date": "20120528"},
    {"code": "510500", "name": "中证500", "sector": "宽基", "list_date": "20130315"},
    {"code": "159915", "name": "创业板", "sector": "宽基", "list_date": "20111209"},
    {"code": "588000", "name": "科创50", "sector": "宽基", "list_date": "20201116"},
    # 金融 (Financial)
    {"code": "512880", "name": "证券", "sector": "金融", "list_date": "20160808"},
    {"code": "512800", "name": "银行", "sector": "金融", "list_date": "20170803"},
    # 科技 (Technology)
    {"code": "512480", "name": "半导体", "sector": "科技", "list_date": "20190612"},
    {"code": "515070", "name": "AI", "sector": "科技", "list_date": "20190801"},
    {"code": "515880", "name": "通信", "sector": "科技", "list_date": "20190816"},
    {"code": "512720", "name": "计算机", "sector": "科技", "list_date": "20190711"},
    # 新能源 (New Energy)
    {"code": "515790", "name": "光伏", "sector": "新能源", "list_date": "20201218"},
    {"code": "516160", "name": "新能源", "sector": "新能源", "list_date": "20210304"},
    {"code": "516110", "name": "汽车", "sector": "新能源", "list_date": "20210416"},
    # 消费 (Consumer)
    {"code": "159928", "name": "消费", "sector": "消费", "list_date": "20130916"},
    {"code": "512010", "name": "医药", "sector": "消费", "list_date": "20130916"},
    # 资源 (Resources)
    {"code": "159825", "name": "农业", "sector": "资源", "list_date": "20201218"},
    {"code": "516780", "name": "稀土", "sector": "资源", "list_date": "20210318"},
    {"code": "512400", "name": "有色", "sector": "资源", "list_date": "20170803"},
    {"code": "516020", "name": "化工", "sector": "资源", "list_date": "20210308"},
    # TMT
    {"code": "512980", "name": "传媒", "sector": "TMT", "list_date": "20190118"},
    {"code": "159869", "name": "游戏", "sector": "TMT", "list_date": "20210225"},
    # 高端制造 (High-end Manufacturing)
    {"code": "512660", "name": "军工", "sector": "高端制造", "list_date": "20160808"},
    # 防御 (Defensive)
    {"code": "510880", "name": "红利", "sector": "防御", "list_date": "20070118"},
    {"code": "518880", "name": "黄金", "sector": "防御", "list_date": "20130729"},
    {"code": "511880", "name": "货币", "sector": "防御", "list_date": "20130318"},
]


# --- Public Helpers ----------------------------------------------------------

def get_available_etfs(date_str: str) -> list[dict]:
    """Return ETFs whose listing date is at least 120 calendar days before
    *date_str*, ensuring enough price history exists for MA computation.

    Parameters
    ----------
    date_str : str
        Reference date in ``"YYYYMMDD"`` format (e.g. ``"20220104"``).

    Returns
    -------
    list[dict]
        Subset of *ETF_UNIVERSE_MULTI* that satisfies the listing maturity
        requirement.
    """
    date = datetime.strptime(date_str, "%Y%m%d")
    cutoff = date - timedelta(days=120)
    return [
        etf
        for etf in ETF_UNIVERSE_MULTI
        if datetime.strptime(etf["list_date"], "%Y%m%d") <= cutoff
    ]


def get_sector(etf_code: str) -> str:
    """Return the sector name for a given ETF code.

    Parameters
    ----------
    etf_code : str
        Six-digit ETF code (e.g. ``"515790"``).

    Returns
    -------
    str
        Sector name (e.g. ``"新能源"``), or ``"未知"`` if not found.
    """
    for etf in ETF_UNIVERSE_MULTI:
        if etf["code"] == etf_code:
            return etf["sector"]
    return "未知"


# --- Legacy Universe (Backward Compatibility) ---------------------------------

ETF_UNIVERSE_LEGACY = {"510300", "510500", "159915", "510880", "518880", "511880"}


def get_sector_map() -> dict[str, set[str]]:
    """Build a mapping of sector name -> set of ETF codes belonging to that
    sector.

    Returns
    -------
    dict[str, set[str]]
        e.g. ``{"科技": {"512480", "515070", "515880", "512720"}, ...}``
    """
    sector_map: dict[str, set[str]] = {}
    for etf in ETF_UNIVERSE_MULTI:
        sector_map.setdefault(etf["sector"], set()).add(etf["code"])
    return sector_map

from __future__ import annotations

ALLOWED_COMPANIES = [
    # ===== Global Tech =====
    "APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "META",
    "NETFLIX", "NVIDIA", "AMD", "INTEL",

    # ===== Indian IT =====
    "INFOSYS", "TCS", "WIPRO", "HCL", "TECH MAHINDRA",
    "LTIMINDTREE", "MPHASIS", "COFORGE",

    # ===== Indian Conglomerates =====
    "RELIANCE", "TATA MOTORS", "MARUTI", "ADANIENT", "ADANIPORTS",

    # ===== Banking & Finance =====
    "HDFC", "ICICI", "AXIS BANK",
    "JP MORGAN", "GOLDMAN SACHS", "MORGAN STANLEY",
    "MASTERCARD", "VISA",

    # ===== FMCG / Retail =====
    "COCA COLA", "PEPSICO", "WALMART", "NESTLE",

    # ===== Energy =====
    "ONGC", "BPCL", "IOC",

    # ===== Crypto / Digital Assets =====
    "BITCOIN", "ETHEREUM", "SOLANA", "DOGECOIN",
]

TICKER_MAP = {
    # ===== Global Tech =====
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "NETFLIX": "NFLX",
    "NVIDIA": "NVDA",
    "AMD": "AMD",
    "INTEL": "INTC",

    # ===== Indian IT (NSE) =====
    "INFOSYS": "INFY.NS",
    "TCS": "TCS.NS",
    "WIPRO": "WIPRO.NS",
    "HCL": "HCLTECH.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "LTIMINDTREE": "LTIM.NS",
    "MPHASIS": "MPHASIS.NS",
    "COFORGE": "COFORGE.NS",

    # ===== Indian Conglomerates =====
    "RELIANCE": "RELIANCE.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "MARUTI": "MARUTI.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",

    # ===== Banking & Finance =====
    "HDFC": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
    "JP MORGAN": "JPM",
    "GOLDMAN SACHS": "GS",
    "MORGAN STANLEY": "MS",

    # ===== FMCG / Retail =====
    "COCA COLA": "KO",
    "PEPSICO": "PEP",
    "WALMART": "WMT",
    "NESTLE": "NSRGY",

    # ===== Energy =====
    "ONGC": "ONGC.NS",
    "BPCL": "BPCL.NS",
    "IOC": "IOC.NS",

    # ===== Crypto =====
    "BITCOIN": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "SOLANA": "SOL-USD",
    "DOGECOIN": "DOGE-USD",
}


def validate_company(company: str):
    if not company:
        return None
    company = company.strip().upper()
    return company if company in ALLOWED_COMPANIES else None


def get_ticker(company: str) -> str | None:
    if not company:
        return None
    return TICKER_MAP.get(company.strip().upper())

import os

import httpx

FINANCIAL_DATASETS_API_BASE = "https://api.financialdatasets.ai"
headers= {}
headers["X-API-KEY"] = os.environ.get("FINANCIAL_DATASETS_API_KEY")

def validate_ticker(ticker:str):

        try:
            params = {"ticker": ticker}
            ticker_info = httpx.get(f"{FINANCIAL_DATASETS_API_BASE}/company/facts", params=params, headers=headers, follow_redirects=True)
            ticker_info.raise_for_status()
            return True
        except:
            return False
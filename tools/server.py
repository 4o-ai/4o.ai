import json
import os
from typing import Annotated, Any, Literal
import httpx
import logging
import sys
import pandas as pd
import pandas_ta as ta
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("finance_server")

mcp = FastMCP("finance-server")

FINANCIAL_DATASETS_API_BASE = "https://api.financialdatasets.ai"


async def make_request(url: str) -> dict[str, any] | None:
    """Make a request to the Financial Datasets API with proper error handling."""
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"Error": str(e)}

@mcp.tool()
async def get_income_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 2,
) -> str:
    """Get income statements for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the income statement (e.g. annual, quarterly, ttm)
        limit: Number of income statements to return (default: 4)
    """
    url = f"{FINANCIAL_DATASETS_API_BASE}/financials/income-statements/?ticker={ticker}&period={period}&limit={limit}"
    data = await make_request(url)

    if not data:
        return "Unable to fetch income statements or no income statements found."

    income_statements = data.get("income_statements", [])

    if not income_statements:
        return "Unable to fetch income statements or no income statements found."

    return json.dumps(income_statements, indent=2)


@mcp.tool()
async def get_balance_sheets(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get balance sheets for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the balance sheet (e.g. annual, quarterly, ttm)
        limit: Number of balance sheets to return (default: 4)
    """
    url = f"{FINANCIAL_DATASETS_API_BASE}/financials/balance-sheets/?ticker={ticker}&period={period}&limit={limit}"
    data = await make_request(url)

    if not data:
        return "Unable to fetch balance sheets or no balance sheets found."

    balance_sheets = data.get("balance_sheets", [])

    if not balance_sheets:
        return "Unable to fetch balance sheets or no balance sheets found."

    return json.dumps(balance_sheets, indent=2)


@mcp.tool()
async def get_cash_flow_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get cash flow statements for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the cash flow statement (e.g. annual, quarterly, ttm)
        limit: Number of cash flow statements to return (default: 4)
    """
    url = f"{FINANCIAL_DATASETS_API_BASE}/financials/cash-flow-statements/?ticker={ticker}&period={period}&limit={limit}"
    data = await make_request(url)

    if not data:
        return "Unable to fetch cash flow statements or no cash flow statements found."

    cash_flow_statements = data.get("cash_flow_statements", [])

    if not cash_flow_statements:
        return "Unable to fetch cash flow statements or no cash flow statements found."

    return json.dumps(cash_flow_statements, indent=2)


@mcp.tool()
async def get_current_stock_price(ticker: str) -> str:
    """Get the current / latest price of a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
    """
    url = f"{FINANCIAL_DATASETS_API_BASE}/prices/snapshot/?ticker={ticker}"
    data = await make_request(url)

    if not data:
        return "Unable to fetch current price or no current price found."

    snapshot = data.get("snapshot", {})

    if not snapshot:
        return "Unable to fetch current price or no current price found."

    return json.dumps(snapshot, indent=2)


@mcp.tool()
async def get_historical_stock_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "day",
    interval_multiplier: int = 1,
) -> str:
    """Gets historical stock prices for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        start_date: Start date of the price data (e.g. 2020-01-01)
        end_date: End date of the price data (e.g. 2020-12-31)
        interval: Interval of the price data (e.g. minute, hour, day, week, month)
        interval_multiplier: Multiplier of the interval (e.g. 1, 2, 3)
    """
    url = f"{FINANCIAL_DATASETS_API_BASE}/prices/?ticker={ticker}&interval={interval}&interval_multiplier={interval_multiplier}&start_date={start_date}&end_date={end_date}"
    data = await make_request(url)

    if not data:
        return "Unable to fetch prices or no prices found."

    prices = data.get("prices", [])

    if not prices:
        return "Unable to fetch prices or no prices found."

    return json.dumps(prices, indent=2)


@mcp.tool()
async def get_company_news(ticker: str) -> str:
    """Get news for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
    """
    url = f"{FINANCIAL_DATASETS_API_BASE}/news/?ticker={ticker}"
    data = await make_request(url)

    if not data:
        return "Unable to fetch news or no news found."

    news = data.get("news", [])

    if not news:
        return "Unable to fetch news or no news found."
    return json.dumps(news, indent=2)

@mcp.tool()
async def run_dynamic_strategy(
    ticker: Annotated[str,
        "The stock ticker symbol, e.g., 'MSFT' or 'NVDA'."
    ],
    indicators: Annotated[list[dict[str, Any]],
        """A list of dictionaries defining the indicators to calculate.
        Each dict must have a 'kind' key.
        Example: [{"kind": "rsi"}, {"kind": "sma", "length": 50}]"""
    ],
    start_date: Annotated[str,
        "The start date for the historical data in 'YYYY-MM-DD' format."
    ],
    end_date: Annotated[str,
        "The end date for the historical data in 'YYYY-MM-DD' format."
    ],
    interval: Annotated[Literal["minute", "hour", "day", "week", "month"],
        "The time interval for each data point. Must be one of the specified values."
    ],
    interval_multiplier: Annotated[int,
        """The number to multiply the interval by. For example, with
        interval='day' and multiplier=5, the data points will be 5 days apart."""
    ] = 1
):
    """
    Analyzes a stock by dynamically building and running a custom strategy
    based on a provided list of technical indicators.
    """
    url = f"{FINANCIAL_DATASETS_API_BASE}/prices/?ticker={ticker}&interval={interval}&interval_multiplier={interval_multiplier}&start_date={start_date}&end_date={end_date}"
    data = await make_request(url)
    
    # Extract the prices data from the API response
    prices_data = data.get('prices', [])
    if not prices_data:
        return {"Error": "No price data found in API response"}
    
    df = pd.DataFrame(data=prices_data)
    study_name = f"Dynamic Analysis for {ticker}"
    indicators = [
        {"kind": "sma", "length": 20},
        {"kind": "ema", "length": 20},
        {"kind": "ema", "length": 50},
        {"kind": "rsi", "length": 14},
        {"kind": "stoch", "k": 14, "d": 3},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "bbands", "length": 20, "std": 2},
        {"kind": "atr", "length": 14},
        {"kind": "true_range", "length": 14},
        {"kind": "ha"},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "vol"},
    ]
    agent_study = ta.Study(
            name=study_name,
            description="A custom study generated by an AI agent.",
            ta=indicators  # The agent's list of indicators is passed directly here
        )
    df.ta.study(agent_study)
    indicator_columns = [col for col in df.columns if any(ind['kind'].upper() in col.upper() for ind in indicators)]
    
    close_col = None
    for col in df.columns:
        if col.lower() in ['close']:
            close_col = col
            break
    
    if close_col is None:
        return df
    
    columns_to_return = [close_col] + list(dict.fromkeys(indicator_columns))
    return df[columns_to_return]

        

if __name__ == "__main__":
    mcp.run()

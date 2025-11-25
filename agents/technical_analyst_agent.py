# technical_analyst_agent_interpreter.py

# --- Required Installations ---
# pip install langchain langgraph pandas pandas-ta

import asyncio
from typing import List, TypedDict, Annotated
from numba.core.types import none
import pandas as pd
import pandas_ta as ta
import numpy as np
import io
import contextlib
from datetime import datetime

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import tool
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from tools.client import MCPClient


class AgentState(TypedDict):
    """
    Represents the state of our agent. This must be consistent
    across all agents in your graph.
    """
    messages: List[BaseMessage]


# --- Data Fetching ---
# IMPORTANT: Replace this mock function with your actual API call to your financial datasets.
def _fetch_data_from_financial_datasets(ticker: str, period_days: int = 365) -> pd.DataFrame:
    """
    Placeholder function to fetch historical stock data.

    Args:
        ticker (str): The stock ticker symbol.
        period_days (int): The number of recent days of data to fetch.

    Returns:
        pd.DataFrame: A DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
                      and a DatetimeIndex.
    """
    print(f"--- MOCK DATA: Fetching {period_days} days of data for {ticker} from custom API ---")
    
    # In your real implementation, you would have an API client like this:
    # from financial_datasets.client import FinancialAPIClient
    # client = FinancialAPIClient(api_key="YOUR_API_KEY")
    # return client.get_daily_history(ticker, days=period_days)

    # For demonstration, we generate realistic-looking mock data.

    date_range = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=period_days, freq='D'))
    price = 150 + (np.random.randn(period_days).cumsum() * 0.5)
    
    df = pd.DataFrame({
        'Open': price + np.random.uniform(-1, 1, period_days),
        'High': price + np.random.uniform(0, 2, period_days),
        'Low': price + np.random.uniform(-2, 0, period_days),
        'Close': price,
        'Volume': np.random.randint(1_000_000, 10_000_000, period_days)
    }, index=date_range)
    
    # Ensure 'Close' is a numeric type for pandas-ta
    df['Close'] = pd.to_numeric(df['Close'])
    
    if df.empty:
        raise ValueError(f"Could not fetch data for ticker: {ticker} from your API.")
        
    return df


# --- Agent Tools ---

# --- The Code Interpreter Tool ---




class TechnicalAnalystAgent:
    """
    This agent uses a Python Code Interpreter to perform dynamic technical analysis.
    It can write and execute code using pandas-ta to respond to user requests,
    making it highly flexible.
    """
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.tools = None
        self.agent_executor = None

    async def get_tools(self):
        if self.tools is None:
            self.tools = await MCPClient().get_mcp_tools()
        return self.tools 
    async def _build_agent_executor(self):
        prompt = self._create_prompt_template()
        tools = await self.get_tools()
        if self.agent_executor is None:
            self.agent_executor = create_react_agent(model=self.model, tools = tools, prompt=prompt)
        return self.agent_executor
    def _create_prompt_template(self) -> ChatPromptTemplate: # The enhanced prompt for your TechnicalAnalystAgent

        PROMPT = """
        You are an expert Quantitative Financial Analyst. Your job is to deconstruct a user's request and use the `run_dynamic_strategy` tool to perform technical analysis for the passed ticker.

        Carefully read the user's request to identify all the necessary arguments for the `run_dynamic_strategy` tool

        **TOOL USAGE:**
        You must call the `run_dynamic_strategy` tool. The `indicators` parameter is provided by default to it. Also based on your intelligence, analyze what will be the best possible period for the technical analysis

        **Use Sensible Defaults:** If the user does not provide all the information, you must use intelligent defaults.
        - If no start or end date is given, assume the user wants to analyze the **last three months** of data from today's date ({current_date}).
        - If no interval is specified, default to `'day'`.
        
        Once you receive the tabular data, perform the following analysis:

        *A. TREND ANALYSIS:*
        - Examine moving averages (SMA/EMA): Is price above or below key levels?
        - Identify trend direction: uptrend, downtrend, or sideways
        - Check for golden cross (short MA > long MA) or death cross (short MA < long MA)
        - Note recent trend changes or breakouts

        *B. MOMENTUM ANALYSIS:*
        - *RSI (14):* 
          • >70 = Overbought (potential sell signal)
          • <30 = Oversold (potential buy signal)
          • 40-60 = Neutral zone
          • Look for divergences with price
        - *MACD:*
          • Check MACD line vs Signal line crossovers
          • Analyze histogram for momentum strength
          • Identify bullish/bearish divergences
        - *Stochastic:* Check for overbought/oversold conditions

        *C. VOLATILITY ANALYSIS:*
        - *Bollinger Bands:*
          • Price near upper band = potential reversal down
          • Price near lower band = potential reversal up
          • Band squeeze = low volatility, potential breakout coming
          • Band expansion = high volatility period
        - *ATR:* Higher values indicate increased volatility

        *D. SIGNAL GENERATION:*
        Based on the indicators, identify:
        - *BUY Signals:* Confluence of bullish indicators
        - *SELL Signals:* Confluence of bearish indicators
        - *HOLD Signals:* Mixed or neutral signals
        - *Risk Level:* Based on volatility and position relative to support/resistance

        *E. KEY METRICS TO REPORT:*
        - Latest Close Price
        - Current values of all indicators
        - Recent crossovers or signal changes
        - Support and resistance levels (if identifiable)
        - Risk/reward assessment
        
        Based on tha above analysis, frame a summary in 4-5 lines of all the indicators analyzed.
        """
        template = PromptTemplate.from_template(PROMPT)
        current_date = datetime.now().date()
        prompt_template = template.invoke({"current_date":current_date}).to_string()
        return prompt_template

    
    

    
        
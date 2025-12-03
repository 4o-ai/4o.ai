import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from tools.client import MCPClient

import os

from langchain_google_genai import ChatGoogleGenerativeAI

class FundamentalAnalysisAgent:
    """
    This agent is responsible for analyzing the fundamental analysis of a given investment instrument.
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

    def _create_prompt_template(self):
        PROMPT = """You are an expert fundamental analysis assistant specializing in comprehensive financial analysis of stocks and investment instruments. 
            You have access to powerful financial data tools:

            {tools}

            **Your Core Capabilities:**
            
            **FINANCIAL STATEMENTS ANALYSIS:**
            - **Income Statements**: Analyze revenue trends, profitability margins, expense ratios, earnings growth
            - **Balance Sheets**: Evaluate asset quality, debt levels, equity structure, financial stability
            - **Cash Flow Statements**: Assess cash generation, capital allocation, liquidity position
            
            **VALUATION & METRICS:**
            - Calculate and interpret key financial ratios (P/E, P/B, ROE, ROA, Debt-to-Equity, etc.)
            - Analyze growth rates, margin trends, and efficiency metrics
            - Compare metrics across time periods and against industry benchmarks
            
            **INVESTMENT ANALYSIS:**
            - Evaluate business model strength and competitive position
            - Assess management effectiveness and capital allocation decisions
            - Identify financial strengths, weaknesses, and red flags
            - Provide investment recommendations based on fundamental data

            **Guidelines for Analysis:**
            1. **Always start with the big picture**: Get multiple years of financial statements to identify trends
            2. **Use appropriate time periods**: 
               - Annual data for long-term trends and stability analysis
               - Quarterly data for recent performance and seasonal patterns
               - TTM (trailing twelve months) for most current performance
            3. **Calculate key ratios and metrics**: Don't just present raw numbers, interpret them
            4. **Provide context**: Compare year-over-year growth, identify trends, and explain what the numbers mean
            5. **Be comprehensive**: Analyze all three financial statements for a complete picture
            6. **Focus on quality metrics**: Emphasize sustainable earnings, cash flow quality, and balance sheet strength

            **Analysis Framework:**
            1. **Profitability Analysis**: Revenue growth, margin trends, earnings quality
            2. **Financial Health**: Liquidity, solvency, capital structure
            3. **Efficiency Metrics**: Asset utilization, working capital management
            4. **Growth Assessment**: Historical and projected growth rates
            5. **Risk Evaluation**: Financial leverage, earnings volatility, cash flow stability
            6. **Investment Thesis**: Synthesize findings into actionable insights

            **ERROR HANDLING**
            1. **NEVER stop if one tools fails** - Proceed with the available data and rest of the tools. Do not ask for any permissions for it.

            When a user provides a ticker symbol, conduct a thorough fundamental analysis using the available tools. 
            Present your findings in a clear, structured manner with specific metrics and actionable insights.

            Don't include any other verbose explanatiouns and don't include the markdown syntax anywhere.
        """
        template = PromptTemplate.from_template(PROMPT)
        rendered_prompt_template = template.invoke({"tools":self.tools}).to_string()
        return rendered_prompt_template

"""
Chainlit Frontend for Multi-Agent Financial Analysis System
Conversational interface for Technical, Fundamental, and Sentiment Analysis
"""

import textwrap   
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv
import nest_asyncio

import chainlit as cl
from graph import Graph
from agents.aggregator import AggregatedResult
from chainlit import make_async

from utils import validate_ticker

# Load environment variables
load_dotenv()

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()


# class FinancialAnalysisChat:
#     """Manages the financial analysis chat session"""
    
#     def _init_(self):
#         self.graph: Optional[Graph] = None
_graph_instance: Optional[Graph] = None
def get_or_create_graph() -> Graph:
    """Initialize the Graph instance (lazy loaded)"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = Graph()
    return _graph_instance
    
async def analyze_stock(ticker: str) -> AggregatedResult:
    """Run stock analysis using LangGraph workflow"""
    graph = get_or_create_graph()
    state = await graph.build_graph(ticker)
    return state.get("aggregated_analysis")


def format_recommendation_message(result: AggregatedResult) -> str:
    """Format the analysis result into a balanced, professional summary."""
    
    rec = result.recommendation.value
    rec_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(rec, "⚪")

    # Helper for visual bars
    def make_bar(value: float, length: int = 10) -> str:
        value = max(0.0, min(1.0, value))
        filled = int(value * length)
        return "▓" * filled + "░" * (length - filled)

    bar_conf = make_bar(result.confidence)
    bar_score = make_bar(result.overall_score)

    # 1. Compact Header with Integrated Verdict
    msg = textwrap.dedent(f"""
        ### {rec_emoji} Verdict: **{rec}**
        
        **Target:** `{result.ticker}`  |  **Confidence:** `{result.confidence:.1%}`
    """)

    # 2. Executive Summary (Justification)
    msg += textwrap.dedent(f"""
        > {result.justification}
    """)

    # 3. Scorecard Table
    msg += textwrap.dedent(f"""
        | Metric | Score |
        | :--- | :--- |
        | **Overall** | `{result.overall_score:.2f}` |
        | **Confidence** | `{result.confidence:.1%}` |
    """)

    # 4. Agent Analysis Breakdown
    scores = result.metadata.get('individual_scores', {})
    
    def format_row(label, key):
        val = scores.get(key, 0)
        return f"| {label} | `{val:.2f}` |"

    msg += textwrap.dedent(f"""
        ### 🤖 Factor Analysis
        | Component | Score |
        | :--- | :--- |
        {format_row("📈 Technical", "technical")}
        {format_row("💰 Fundamental", "fundamental")}
        {format_row("📰 Sentiment", "sentiment")}
    """)

    # 5. Data Sources (Fixed: Using clean Markdown instead of broken HTML)
    msg += "\n### 🔍 Data Sources\n"
    
    for agent_type, evidence in result.supporting_evidence.items():
        # Clean up evidence text
        clean_ev = str(evidence).replace('\n', ' ').strip()
        display_ev = clean_ev[:500] + "..." if len(clean_ev) > 500 else clean_ev
        
        # Use H4 headers and Blockquotes for clear, fail-safe formatting
        msg += textwrap.dedent(f"""
            #### 📄 {agent_type.title()} Report
            > {display_ev}
            
        """)

    return msg

def create_download_elements(result: AggregatedResult):
    """Create downloadable files for the analysis"""
    
    # JSON output
    json_data = {
        "ticker": result.ticker,
        "recommendation": result.recommendation.value,
        "confidence": result.confidence,
        "overall_score": result.overall_score,
        "justification": result.justification,
        "supporting_evidence": result.supporting_evidence,
        "metadata": result.metadata,
        "timestamp": result.timestamp
    }
    
    # Text report
    text_report = f"""
Investment Analysis Report
{'='*50}

Ticker: {result.ticker}
Recommendation: {result.recommendation.value}
Confidence: {result.confidence:.1%}
Overall Score: {result.overall_score:.2f}

Justification:
{result.justification}

Individual Scores:
•⁠  ⁠Technical: {result.metadata.get('individual_scores', {}).get('technical', 0):.2f}
•⁠  ⁠Fundamental: {result.metadata.get('individual_scores', {}).get('fundamental', 0):.2f}
•⁠  ⁠Sentiment: {result.metadata.get('individual_scores', {}).get('sentiment', 0):.2f}

Supporting Evidence:
{'-'*50}
"""
    
    for agent_type, evidence in result.supporting_evidence.items():
        text_report += f"\n{agent_type.title()}:\n{evidence}\n\n"
    
    text_report += f"\nAnalysis Date: {result.timestamp}\n"
    
    return json_data, text_report


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    
    # Create analysis instance
    # analysis_chat = FinancialAnalysisChat()
    # cl.user_session.set("analysis_chat", analysis_chat)
    
    # Send welcome message
    welcome_msg = """
# 🤖 AI Financial Analyst

Welcome! I'm your AI-powered financial analysis assistant.

## 💡 What I Can Do:
•⁠  ⁠📊 *Technical Analysis*: Charts, indicators, and price patterns
•⁠  ⁠💰 *Fundamental Analysis*: Financial statements and company metrics  
•⁠  ⁠📰 *Sentiment Analysis*: News and market sentiment

## 🎯 How to Use:
Simply send me a stock ticker symbol (e.g., ⁠ AAPL ⁠, ⁠ MSFT ⁠, ⁠ TSLA ⁠) and I'll provide you with a comprehensive investment analysis!

*Features:*
•⁠  ⁠Multi-agent AI system powered by LangGraph
•⁠  ⁠Real-time parallel analysis
•⁠  ⁠Downloadable reports (JSON & Text)
•⁠  ⁠Confidence scoring

---

Ready to analyze? Send me a ticker symbol to get started! 📈
    """
    
    await cl.Message(content=welcome_msg).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    
    # Get the analysis chat instance
    # analysis_chat = cl.user_session.get("analysis_chat")
    
    # if not analysis_chat:
    #     await cl.Message(
    #         content="❌ Session error. Please refresh the page.",
    #         type="error"
    #     ).send()
    #     return
    
    # Extract ticker from message
    ticker = message.content.strip().upper()
    
    checking_msg = cl.Message(content=f"🔎 Verifying ticker **{ticker}**...")
    await checking_msg.send()

    # Validate ticker (basic validation)
    is_valid = await make_async(validate_ticker)(ticker)
    
    if not is_valid:
        await checking_msg.remove()
        await cl.Message(content=f"❌ Ticker **{ticker}** not found. Please check the spelling and try again.").send()
        return
    
    await checking_msg.remove()
    
    # Show thinking message
    thinking_msg = cl.Message(
        content=f"🔬 Analyzing *{ticker}*...\n\nRunning parallel analysis across multiple AI agents:\n- 📊 Technical Analyst\n- 💰 Fundamental Analyst\n- 📰 Sentiment Analyst"
    )
    await thinking_msg.send()
    
    try:
        # Show progress steps
        async with cl.Step(name=f"Analysis: {ticker}", type="tool") as step:
            step.output = "Starting multi-agent analysis..."
            
            # Run the analysis
            result = await analyze_stock(ticker)
            
            step.output = f"✅ Analysis complete! Recommendation: {result.recommendation.value}"
        
        # Update thinking message
        thinking_msg.content = f"✅ Analysis for *{ticker}* completed!"
        await thinking_msg.update()
        
        # Format and send the recommendation
        recommendation_text = format_recommendation_message(result)
        
        # Create actions (download buttons)
        json_data, text_report = create_download_elements(result)
        
        actions = [
            cl.Action(
                name="download_text",
                payload = {"value": text_report},
                label="📝 Download Report"
            ),
            cl.Action(
                name="analyze_another",
                payload = {"value": "reset"},
                label="🔄 Analyze Another Stock"
            )
        ]
        
        # Send the result message with actions
        await cl.Message(
            content=recommendation_text,
            actions=actions
        ).send()
        
    except Exception as e:
        error_msg = f"❌ *Analysis Failed*\n\nError: {str(e)}\n\nPlease try again or contact support if the issue persists."
        await cl.Message(
            content=error_msg,
            type="error"
        ).send()


@cl.action_callback("download_json")
async def on_download_json(action: cl.Action):
    """Handle JSON download action"""
    await cl.Message(
        content=f"⁠  json\n{action.value}\n  ⁠\n\n*Copy the JSON above or use the download feature.*"
    ).send()


@cl.action_callback("download_text")
async def on_download_text(action: cl.Action):
    """Handle text report download action"""
    await cl.Message(
        content=f"⁠  \n{action.value}\n  ⁠\n\n*Copy the report above.*"
    ).send()


@cl.action_callback("analyze_another")
async def on_analyze_another(action: cl.Action):
    """Handle analyze another stock action"""
    await cl.Message(
        content="📈 Ready to analyze another stock! Just send me the ticker symbol."
    ).send()


# Optional: Add settings
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """
    Optional: Add authentication
    Remove this if you don't need authentication
    """
    # For demo purposes, accept any credentials
    # In production, validate against a database or auth service
    return cl.User(identifier=username, metadata={"role": "user"})


if __name__ == "__main__":
    # Run with: chainlit run chainlit_app.py
    pass
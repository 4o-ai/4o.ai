import asyncio
from email import message
import os
from os.path import isfile
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from agents import sentimental_analyst_agent
from agents import fundamental_analysis_agent
from agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from agents.sentimental_analyst_agent import SentimentalAnalystAgent
from agents.technical_analyst_agent import TechnicalAnalystAgent
from agents.aggregator import AggregatedResult, InvestmentAggregator
from langgraph.graph import END, START, StateGraph
from state import WorkflowState
from langgraph.graph.state import CompiledStateGraph
from IPython.display import Image, display

class Graph:
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=api_key
        )
        self._technical_agent = None
        self._sentiment_agent = None
        self._fundamental_agent = None
    
    def _get_sentiment_agent(self):
        if self._sentiment_agent is None:
            self._sentiment_agent = SentimentalAnalystAgent(self.llm)
        return self._sentiment_agent

    def _get_techninal_analyst_agent(self):
        if self._technical_agent is None:
            self._technical_agent = TechnicalAnalystAgent(self.llm)
        return self._technical_agent

    def _get_fundamental_agent(self):
        if self._fundamental_agent is None:
            self._fundamental_agent = FundamentalAnalysisAgent(self.llm)
        return self._fundamental_agent
    async def _run_technical_analysis(self, state: WorkflowState) -> Dict[str, Any]:
        """Run technical analysis for the ticker"""
        try:
            # Your technical agent should accept ticker as input
            analysis_query = f"Perform technical analysis for {state.ticker}"
            agent = await self._get_techninal_analyst_agent()._build_agent_executor()
            result = await agent.ainvoke({"messages": [analysis_query]})
            return {"technical_analysis": result['messages'][-1].content, "agent_type": ["technical"]}
        except Exception as e:
            return {"error": str(e), "agent_type": ["technical"]}
    
    async def _run_fundamental_analysis(self, state: WorkflowState) -> Dict[str, Any]:
        """Run fundamental analysis for the ticker""" 
        try:
            analysis_query = f"Perform fundamental analysis for {state.ticker}"
            agent = await self._get_fundamental_agent()._build_agent_executor()
            result = await agent.ainvoke({"messages": [analysis_query]})
            return {"fundamental_analysis": result['messages'][-1].content, "agent_type": ["fundamental"]}
        except Exception as e:
            return {"error": str(e), "agent_type": ["fundamental"]}
    
    async def _run_sentiment_analysis(self, state: WorkflowState) -> Dict[str, Any]:
        """Run sentiment analysis for the ticker"""
        try:
            analysis_query = f"Analyze market sentiment for {state.ticker}"
            agent : CompiledStateGraph = await self._get_sentiment_agent()._build_agent_executor()
            result = await agent.ainvoke({"messages": [analysis_query]})
            return {"sentimental_analysis": result['messages'][-1].content, "agent_type": ["sentiment"]}
        except RuntimeError as e:
            raise e
        except Exception as e:
            return {"error": str(e), "agent_type": ["sentiment"]}
    

    async def aggregate(self, state: WorkflowState) -> AggregatedResult:

        self.aggregator = InvestmentAggregator(llm=self.llm)
        technical_analysis_result = state.technical_analysis
        sentimental_analysis_result = state.sentimental_analysis
        fundamental_analysis_result = state.fundamental_analysis
        agent_results = {
            'technical': self.aggregator._standardize_agent_result('technical', technical_analysis_result),
            'fundamental': self.aggregator._standardize_agent_result('fundamental', fundamental_analysis_result),
            'sentiment': self.aggregator._standardize_agent_result('sentiment', sentimental_analysis_result)
        }
        
        result = self.aggregator._analyze(agent_results, state.ticker)
        return {"aggregated_analysis": result}


    async def build_graph(self, ticker: str):

        # Adding nodes
        graph = StateGraph(WorkflowState)
        graph.add_node("technical_analysis", self._run_technical_analysis)
        graph.add_node("sentimental_analysis", self._run_sentiment_analysis)
        graph.add_node("fundamental_analysis", self._run_fundamental_analysis)
        graph.add_node("aggregator", self.aggregate)

        # Adding edges
        graph.add_edge(START, "technical_analysis")
        graph.add_edge(START, "sentimental_analysis")
        graph.add_edge(START, "fundamental_analysis")
        graph.add_edge("technical_analysis", "aggregator")
        graph.add_edge("sentimental_analysis", "aggregator")
        graph.add_edge("fundamental_analysis", "aggregator")
        graph.add_edge("aggregator", END)
        # graph.add_edge("sentimental_analysis", END)

        parallel_workflow: CompiledStateGraph = graph.compile()
        if not os.path.isfile("graph.png"):
            display(Image(parallel_workflow.get_graph().draw_mermaid_png(output_file_path="graph.png")))

        try:
            state = await parallel_workflow.ainvoke({"ticker": ticker})
            return state
        except Exception as e:
            raise e
        
# if __name__=="__main__":
#     graph = Graph()
#     import nest_asyncio
#     nest_asyncio.apply()
#     output = asyncio.run(graph.build_graph())
#     print(output)

        

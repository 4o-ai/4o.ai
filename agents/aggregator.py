import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Recommendation(str,Enum):
    """Investment recommendation types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AgentResult(BaseModel):
    """Standardized format for agent results"""
    agent_type: str
    data: Dict[str, Any]
    score: Optional[float] = None
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if the agent result is valid and usable"""
        return self.error is None and self.data is not None


class AggregatedResult(BaseModel):
    """Final aggregated investment recommendation"""
    ticker: str
    recommendation: Recommendation
    confidence: float
    overall_score: float
    justification: str
    supporting_evidence: Dict[str, str]
    metadata: Dict[str, Any]
    timestamp: str


class InvestmentAggregator:
    """
    Aggregates outputs from Technical, Fundamental, and Sentiment analysis agents
    to generate final investment recommendations with confidence scores.
    
    Uses LLM-based semantic analysis for intelligent, context-aware scoring.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, llm=None):
        """
        Initialize the aggregator with configurable weights and LLM for intelligent scoring.
        
        Args:
            weights: Dictionary with keys 'technical', 'fundamental', 'sentiment'
                    Default weights: technical=0.4, fundamental=0.4, sentiment=0.2
            llm: Language model for intelligent semantic scoring (required)
        """
        self.weights = weights or {
            'technical': 0.4,
            'fundamental': 0.4, 
            'sentiment': 0.2
        }
        
        self.llm : BaseChatModel = llm
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def aggregate(self, 
                 technical_result: Dict[str, Any] = None,
                 fundamental_result: Dict[str, Any] = None,
                 sentiment_result: Dict[str, Any] = None,
                 ticker: str = "UNKNOWN") -> AggregatedResult:
        """
        Main aggregation method that combines all agent outputs with ONE LLM call.
        
        Args:
            technical_result: Output from technical analysis agent
            fundamental_result: Output from fundamental analysis agent  
            sentiment_result: Output from sentiment analysis agent
            ticker: Stock ticker symbol
            
        Returns:
            AggregatedResult with final recommendation
        """
        timestamp = datetime.now().isoformat()
        
        # Convert inputs to standardized format
        agent_results = {
            'technical': self._standardize_agent_result('technical', technical_result),
            'fundamental': self._standardize_agent_result('fundamental', fundamental_result),
            'sentiment': self._standardize_agent_result('sentiment', sentiment_result)
        }
        
        if self.llm:
            # Use single LLM call for everything
            result = self._analyze(agent_results, ticker)
        
        result.timestamp = timestamp
        return result
    
    def _standardize_agent_result(self, agent_type: str, result: Dict[str, Any]) -> AgentResult:
        """Convert raw agent output to standardized AgentResult format"""
        if result is None:
            return AgentResult(
                agent_type=agent_type,
                data={},
                error=f"{agent_type} agent result is None"
            )
        
        try:
            # Handle different possible result formats
            if isinstance(result, str):
                # Try to parse as JSON
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    # Treat as text result
                    result = {"analysis": result}
            
            return AgentResult(
                agent_type=agent_type,
                data=result,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error standardizing {agent_type} result: {e}")
            return AgentResult(
                agent_type=agent_type,
                data={},
                error=str(e)
            )
    
    def _analyze(self, agent_results: Dict[str, AgentResult], ticker: str) -> AggregatedResult:
        """
        OPTIMIZED: Use ONE LLM call to do everything - scoring, weighting, recommendation, and justification.
        """
        # Prepare combined analysis for the LLM
        combined_analysis = ""
        valid_agents = []
        
        for agent_type, result in agent_results.items():
            if result.is_valid():
                combined_analysis += f"\n=== {agent_type.upper()} ANALYSIS ===\n{str(result.data)}\n"
                valid_agents.append(agent_type)
        
        
        response = self.llm.invoke(self._create_prompt(ticker, combined_analysis))
        
        # Parse JSON response
        analysis_data = json.loads(response.content.strip())
        
        # Extract scores
        scores = {
            'technical': analysis_data.get('technical_score', 0.0),
            'fundamental': analysis_data.get('fundamental_score', 0.0),
            'sentiment': analysis_data.get('sentiment_score', 0.0)
        }
        
        # Validate and clamp scores
        for key in scores:
            scores[key] = max(-1.0, min(1.0, float(scores[key])))
        
        # Get other values
        overall_score = float(analysis_data.get('overall_score', 0.0))
        recommendation = Recommendation(analysis_data.get('recommendation', 'HOLD'))
        confidence = max(0.0, min(1.0, float(analysis_data.get('confidence', 0.5))))
        justification = analysis_data.get('justification', f'{recommendation.value} recommendation based on analysis.')
        
        # Create supporting evidence
        supporting_evidence = self._extract_supporting_evidence(agent_results)
        
        # Create metadata
        metadata = self._create_metadata(agent_results, scores)
        
        return AggregatedResult(
            ticker=ticker,
            recommendation=recommendation,
            confidence=confidence,
            overall_score=overall_score,
            justification=justification,
            supporting_evidence=supporting_evidence,
            metadata=metadata,
            timestamp=""  # Will be set by caller
        )

    
    def _extract_supporting_evidence(self, agent_results: Dict[str, AgentResult]) -> Dict[str, str]:
        """Extract key evidence from each agent's analysis in clean, readable format"""
        evidence = {}
        
        for agent_type, result in agent_results.items():
            if result.is_valid():
                # Clean and format the agent output
                text = str(result.data)
                
                # If it's a dict, try to extract the most relevant info
                if isinstance(result.data, dict):
                    if 'analysis' in result.data:
                        text = str(result.data['analysis'])
                    elif 'summary' in result.data:
                        text = str(result.data['summary'])
                    elif 'conclusion' in result.data:
                        text = str(result.data['conclusion'])
                
                # Clean up and limit length for readability
                text = text.replace('\n', ' ').strip()
                if len(text) > 250:
                    text = text[:250] + "..."
                
                evidence[agent_type] = text
            else:
                evidence[agent_type] = f"Analysis failed: {result.error}"
        
        return evidence
    
    def _create_metadata(self, 
                        agent_results: Dict[str, AgentResult],
                        scores: Dict[str, float]) -> Dict[str, Any]:
        """Create metadata about the aggregation process"""
        
        metadata = {
            "weights_used": self.weights.copy(),
            "agent_status": {
                agent: "success" if result.is_valid() else "failed" 
                for agent, result in agent_results.items()
            },
            "individual_scores": scores.copy(),
            "agents_analyzed": len([r for r in agent_results.values() if r.is_valid()]),
            "total_agents": len(agent_results)
        }
        
        return metadata
    
    def _create_prompt(self, ticker: str, combined_analysis: str) -> str:
        prompt = """
        You are a senior financial advisor analyzing investment recommendations for {ticker}.
        
        ANALYSIS WEIGHTS: Technical=40%, Fundamental=40%, Sentiment=20%
        
        AGENT OUTPUTS:
        {combined_analysis}
        
        Your task: Provide a complete investment analysis in JSON format.
        
        1. Score each analysis from -1.0 to +1.0 (-1=very bearish, 0=neutral, +1=very bullish)
        2. Calculate weighted score using the weights above
        3. Make recommendation: BUY (score ≥ 0.3), SELL (score ≤ -0.3), or HOLD (between)
        4. Assess confidence (0.0-1.0) based on data quality and consensus
        5. Write a professional justification (2-3 sentences maximum)
        
        Respond with ONLY this JSON structure:
        {{
            "technical_score": 0.0,
            "fundamental_score": 0.0, 
            "sentiment_score": 0.0,
            "overall_score": 0.0,
            "recommendation": "BUY",
            "confidence": 0.85,
            "justification": "Clear, professional explanation of the recommendation based on the analysis."
        }}
        - It is very critical that you answer only as the above object and JSON stringify it as a single string.
            Don't include any other verbose explanatiouns and don't include the markdown syntax anywhere.
        """
        template = ChatPromptTemplate.from_template(prompt)
        rendered_prompt_template = template.invoke({"ticker": ticker, "combined_analysis": combined_analysis})
        return rendered_prompt_template
    
    def to_json(self, result: AggregatedResult) -> str:
        """Convert aggregated result to JSON format matching supervisor requirements"""
        output = {
            "action": result.recommendation.value,
            "confidence": result.confidence,
            "justification": result.justification,
            "supporting_evidence": result.supporting_evidence,
            "metadata": {
                **result.metadata,
                "ticker": result.ticker,
                "overall_score": result.overall_score,
                "timestamp": result.timestamp
            }
        }
        
        return json.dumps(output, indent=2)



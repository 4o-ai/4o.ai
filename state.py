from typing import Any, Optional, TypedDict, Literal, Annotated
from langchain_core.messages import AnyMessage
import operator

from pydantic import BaseModel

from agents.aggregator import AggregatedResult

class WorkflowState(BaseModel):
    """Represents the state of our graph, passed between nodes."""
    messages: Annotated[list[AnyMessage], operator.add]
    ticker: str
    technical_analysis: Optional[str] = ""
    fundamental_analysis: Optional[str] = ""
    sentimental_analysis: Optional[str] = ""
    aggregated_analysis: Optional[Any] = None
    agent_type: Annotated[list, operator.add]  



from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from tools.client import MCPClient

class SentimentalAnalystAgent:
    """
    This agent is responsible for analyzing the sentimental analysis of a given investment instrument.
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
        PROMPT = """You are a stock sentimental analysis assistant specializing in market news interpretation. 
            You have access to multiple financial tools:
            
            Tools: {tools}

            Guidelines:
            - When the user request involves **market news, headlines, breaking stories, or sentiment analysis**, 
            use only the news-related tools available from the MCP server.
            - Ignore unrelated tools (e.g., stock price fetchers, chart generators, company fundamentals).
            - Always cite which news source or provider the information is coming from.
            - Never hallucinate a news source—only use what is provided by the connected MCP tools.

            Your role is to act as a **financial news strategist**: summarize key stories, 
            highlight market impact, and deliver insights grounded in the news data 
            retrieved from the MCP tools.

        """
        
        # prompt_template = PromptTemplate(
        #     template=PROMPT,
        #     input_variables=["input", "tools", "agent_scratchpad"]
        # )
        prompt_template = PromptTemplate.from_template(PROMPT)
        prompt_template = prompt_template.invoke({"tools":self.tools}).to_string()
        return prompt_template



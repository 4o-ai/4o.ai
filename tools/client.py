import functools
import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
import logging, httpx



logging.basicConfig(level=logging.DEBUG, force=True)

# Force httpx to show headers
httpx_log = logging.getLogger("httpx")
httpx_log.setLevel(logging.DEBUG)
httpx_log.addHandler(logging.StreamHandler())



def cache(func):
    cache = None
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        nonlocal cache
        if cache is None:
            cache = func(*args,**kwargs)
        return cache
    return wrapper    
@cache
def get_mcp_client():
    return MultiServerMCPClient(
                {
                    "finance_analysis": {
                        "command": "python",
                        "args": ["/Users/tj/TJ Projs/4o.ai/tools/server.py"],
                        "transport": "stdio",
                    }
                }
            )

class MCPClient:
    def __init__(self):
        self.client = get_mcp_client()
        self.tools = None
    async def get_mcp_tools(self):
        
        if self.tools is None:
            try:
                self.tools = await self.client.get_tools(server_name="finance_analysis")
                return self.tools
            except Exception as e:
                raise e
        return self.tools    

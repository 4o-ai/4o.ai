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

load_dotenv()

# def singleton(class_):
#     instances = {}
#     def getinstance(*args, **kwargs):
#         if class_ not in instances:
#             instances[class_] = class_(*args, **kwargs)
#         return instances[class_]
#     return getinstance

# @singleton
# class MCPClient:
#     def __init__(self, api_key: str):
#         self.api_key = api_key
        
#     @staticmethod
#     async def get_mcp_client_tools():
#         try:
#             # Create a temporary client instance for static method
#             client = MultiServerMCPClient(
#                 {
#                     "finance_analysis": {
#                         "command": "python",
#                         "args": ["/Users/tj/TJ Projs/4o.ai/tools/server.py"],
#                         "transport": "stdio",
#                     }
#                 }
#             )
#             tools = await client.get_tools(server_name="finance_analysis")
#             return tools
#         except Exception as e:
#             raise e

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



# if __name__ == "__main__":
#     finance_api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
#     print("apikey......",finance_api_key)
#     client1 = MCPClient().client
#     client2 = MCPClient().client
#     cl
#     print(client1 is client2)
#     for i in range(5):
#         asyncio.run(client1.get_mcp_tools())


    # print("Discovered tools:", [t.name for t in tools])


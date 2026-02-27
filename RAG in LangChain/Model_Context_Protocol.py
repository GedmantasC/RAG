from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Connect to MCP server and fetch tools
    client = MultiServerMCPClient({
        "weather": {
            "url": "http://localhost:8000/mcp/",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()
    
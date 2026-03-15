from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import json
from mcp import ClientSession
from mcp.client.stdio import stdio_client

print('labas as krabas')

async def main():
    # Start an MCP server as a subprocess.
    # Example: a local Python server script exposing tools.
    server_params = {
        "command": "python",
        "args": ["server.py"],
    }

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1) Initialize the connection
            await session.initialize()

            # 2) Ask the server what tools it provides
            tools = await session.list_tools()

            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
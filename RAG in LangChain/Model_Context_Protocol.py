from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import json
from mcp import ClientSession
from mcp.client.stdio import stdio_client

print('labas as krabas')
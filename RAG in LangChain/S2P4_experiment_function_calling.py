from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
import os
import toml
from openai import OpenAI
import getpass
from langgraph.checkpoint.sqlite import SqliteSaver
from mcp.server.fastmcp import FastMCP

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

client = OpenAI(api_key=secrets["OPENAI_API_KEY"])

# Define tools with the calculator_add function
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator_add",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {"type": "number"},
                    "num2": {"type": "number"},
                },
                "required": ["num1", "num2"],
            },
        },
    }
]

# Example of calling the function with the extracted arguments
def calculator_add(num1, num2):
    return num1 + num2

# Simulate a conversation
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Add 8 and 12."}],
    tools=tools,
)
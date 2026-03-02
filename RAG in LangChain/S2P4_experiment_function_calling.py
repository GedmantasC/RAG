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
import json

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
    },

    {
        "type": "function",
        "function": {
            "name": "calculator_minus",
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

def calculator_minus(num1, num2):
    return num1 - num2

# Map tool name -> function
TOOL_ROUTER = {
    "calculator_add": calculator_add,
    "calculator_minus": calculator_minus,
}

def run_prompt(prompt: str):
    messages = [{"role": "user", "content": prompt}]

    # 1) Ask model; it may decide to call a tool
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # default; explicit for clarity
    )

    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []

    # If no tool call, just print the assistant's text
    if not tool_calls:
        print("Assistant:", msg.content)
        return

    # 2) Add assistant message that contains the tool_calls to the message history
    messages.append(msg)

    # 3) Execute every tool call and append tool results
    for tc in tool_calls:
        fn_name = tc.function.name
        fn_args = json.loads(tc.function.arguments or "{}")  # <-- NO eval
        fn = TOOL_ROUTER.get(fn_name)

        if fn is None:
            tool_output = f"ERROR: Tool '{fn_name}' not implemented."
        else:
            try:
                tool_output = fn(**fn_args)
            except Exception as e:
                tool_output = f"ERROR running '{fn_name}': {e}"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(tool_output),
            }
        )

# 4) Ask model again so it can produce a final user-facing response
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    print("Assistant:", final.choices[0].message.content)

# Try it
run_prompt("add 10 and 12")
run_prompt("subtract 12 from 10")
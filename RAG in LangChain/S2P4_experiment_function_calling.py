import os
import json
import httpx
import toml
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

@tool
def calculator_add(num1: float, num2: float) -> float:
    """Add two numbers."""
    return num1 + num2

@tool
def calculator_minus(num1: float, num2: float) -> float:
    """Subtract num2 from num1."""
    return num1 - num2

@tool
def exchange_rate(base: str, target: str) -> str:
    """Get the FX exchange rate from base currency to target currency."""
    url = "https://api.exchangerate.host/latest"
    params = {"base": base.upper(), "symbols": target.upper()}

    with httpx.Client(timeout=10.0) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    rate = data["rates"][target.upper()]
    return json.dumps({"base": base.upper(), "target": target.upper(), "rate": rate})

tools = [calculator_add, calculator_minus, exchange_rate]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Optional: keeps conversation state across calls
checkpointer = InMemorySaver()

agent = create_react_agent(llm, tools=tools, checkpointer=checkpointer)

def run_prompt(prompt: str, thread_id: str = "demo"):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print("Assistant:", result["messages"][-1].content)

run_prompt("add 10 and 12")
run_prompt("subtract 12 from 10")
run_prompt("convert 1 USD to EUR in current rate")
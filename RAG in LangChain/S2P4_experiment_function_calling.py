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
def chuck_norris_joke() -> str:
    """Get a random Chuck Norris joke."""
    
    url = "https://api.chucknorris.io/jokes/random"

    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()

        return json.dumps({"joke": data["value"]})

    except Exception as e:
        return json.dumps({"error": str(e)})

tools = [calculator_add, calculator_minus, chuck_norris_joke]

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

run_prompt("Tell me a Chuck Norris joke")

# run_prompt("add 10 and 12")
# run_prompt("subtract 12 from 10")
# run_prompt("convert 1 USD to EUR in current rate")
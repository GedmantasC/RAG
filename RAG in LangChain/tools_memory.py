from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# Define tools - same pattern as before!
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, this would call a real weather API
    return f"The weather in {city} is sunny, 72°F"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Note: In production, use a safer eval alternative
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
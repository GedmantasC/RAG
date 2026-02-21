from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# Define tools - same pattern as before!
#also in general LLM is used just to predict text, i does'nt have access to the current internet. To do that we describe tools, that allows to check something or do calculation 
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, this would call a real weather API
    return f"The weather in {city} is sunny, 72°F"

#@tool is called decorator, so it marks for model that this is a function, and please feel free to use it
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Note: In production, use a safer eval alternative
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
    
# Create model
model = ChatOpenAI(model="gpt-4o")

# Create checkpointer for memory
checkpointer = InMemorySaver()

# Create agent - this handles all the tool calling logic!
agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="You are a helpful assistant that can check weather and do calculations.",
    checkpointer=checkpointer,
)

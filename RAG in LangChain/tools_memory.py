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

# Define tools - same pattern as before!
#also in general LLM is used just to predict text, i does'nt have access to the current internet. To do that we describe tools, that allows to check something or do calculation 
#also this function will always return hardcoded value, but in real live here should be api calling
#also dock strings are important because LLM use it to understand whet to use this tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, this would call a real weather API
    return f"The weather in {city} is sunny, 72°F"

#@tool is called decorator, so it marks for model that this is a function, and please feel free to use it
@tool
#such a descriprtion of the function shows that string is expected to be get and the output also should be string
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Note: In production, use a safer eval alternative
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

client = OpenAI(api_key=secrets["OPENAI_API_KEY"])


# Create model
model = ChatOpenAI(model="gpt-4o")

# Create checkpointer for memory
#I reserve some space to store conversation. This disapiers after restarting conversation
checkpointer = InMemorySaver()
# With this one it will be saved in a db, so conversation is not gone. 
#checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Create agent - this handles all the tool calling logic!
agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="You are a helpful assistant that can check weather and do calculations.",
    checkpointer=checkpointer,
)

# Use the agent with thread-based memory
#Idea of this is to save conversation. I you talk with model with this it would remember all messages in the conversation
#example. User: My name is Ged. User: What is my name? Agent: Your name is Ged.
#if you change the thread_id it becomes a totally new conversations
config = {"configurable": {"thread_id": "conversation-1"}}

# First message
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
    config=config
)
print(response["messages"][-1].content)

#from here trying model context protocol
mcp = FastMCP("weather")

@mcp.tool()
def get_weather():
    """
    Gets the current weather.
    """
    return "The weather is sunny with a high of 21°C"

if __name__ == "__main__":
    mcp.run(transport="stdio")
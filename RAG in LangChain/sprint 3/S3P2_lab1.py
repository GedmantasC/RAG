from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# Step 1: Define custom state by extending AgentState
class ShoppingState(AgentState):
    cart_items: list[dict] = []  # List of items: [{"name": "Laptop", "price": 1299}, ...]
    budget: float = 100.0         # User's budget
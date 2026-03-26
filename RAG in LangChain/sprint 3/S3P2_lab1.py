from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# Step 1: Define custom state by extending AgentState
class ShoppingState(AgentState):
    cart_items: list[dict] = []  # List of items: [{"name": "Laptop", "price": 1299}, ...]
    budget: float = 100.0         # User's budget

# Step 2: Create a simple tool (we'll make it stateful in the next section)
@tool
def search_products(query: str) -> str:
    """Search for products matching the query."""
    # Mock product search
    products = {
        "laptop": "MacBook Pro - $1,299",
        "phone": "iPhone 15 - $799",
        "headphones": "AirPods Pro - $249",
        "tablet": "iPad Air - $599"
    }
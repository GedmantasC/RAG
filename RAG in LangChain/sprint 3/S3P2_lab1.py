import os
import toml
import getpass
from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Step 1: Define custom state
class ShoppingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    cart_items: list[dict]  # List of items: [{"name": "Laptop", "price": 1299}, ...]
    budget: float            # User's budget

# Step 2: Create a simple tool
@tool
def search_products(query: str) -> str:
    """Search for products matching the query."""
    products = {
        "laptop": "MacBook Pro - $1,299",
        "phone": "iPhone 15 - $799",
        "headphones": "AirPods Pro - $249",
        "tablet": "iPad Air - $599"
    }
    results = [v for k, v in products.items() if query.lower() in k]
    return "\n".join(results) if results else "No products found."

# Step 3: Create agent with custom state
model = ChatOpenAI(model="gpt-4o")
checkpointer = MemorySaver()

shopping_agent = create_react_agent(
    model=model,
    tools=[search_products],
    state_schema=ShoppingState,
    checkpointer=checkpointer
)

print("Shopping agent created with custom state!")
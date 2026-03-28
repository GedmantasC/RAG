import os
import toml
import getpass
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Step 1: Define custom state (MessagesState already includes the required `messages` key)
class ShoppingState(MessagesState):
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

shopping_agent = create_agent(
    model=model,
    tools=[search_products],
    state_schema=ShoppingState,
    checkpointer=checkpointer
)

print("Shopping agent created with custom state!")
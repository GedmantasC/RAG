import os
import toml
import getpass
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import ToolRuntime

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

#idea is that a custom state is like a message that goes through all converation and llm can remember it. 
# in this case it is shoppingState - info about cart items and bugget
#Without custom state, the agent would have to re-parse the entire conversation history every time to figure out what's in the cart.
#  With custom state, it's structured data that any node in the graph can read or write directly — much more reliable and efficient.

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

@tool
def check_budget(item_price: float, runtime: ToolRuntime) -> str:
    """Check if an item fits within the remaining budget."""
    # Read state via runtime.state
    budget = runtime.state.get("budget", 0)
    cart_items = runtime.state.get("cart_items", [])
    
    # Calculate total spent from actual item prices
    spent = sum(item["price"] for item in cart_items)
    remaining = budget - spent
    
    if item_price <= remaining:
        return f"Yes, ${item_price} fits in your ${remaining:.2f} remaining budget."
    else:
        return f"No, only ${remaining:.2f} left in budget. ${item_price} exceeds this."

# Step 3: Create agent with custom state
model = ChatOpenAI(model="gpt-4o")
checkpointer = MemorySaver()

shopping_agent = create_agent(
    model=model,
    tools=[search_products],
    state_schema=ShoppingState,
    checkpointer=checkpointer
)

print("Shopping agent created with custom state! -> The agent now has access to cart_items and budget in addition to messages")
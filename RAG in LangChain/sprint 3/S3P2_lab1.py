import os
import toml
import getpass
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import ToolRuntime
from langgraph.types import Command
from langchain_core.messages import ToolMessage

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
    
@tool
def view_cart(runtime: ToolRuntime) -> str:
    """View all items currently in the shopping cart."""
    cart_items = runtime.state.get("cart_items", [])
    budget = runtime.state.get("budget", 0)
    
    if not cart_items:
        return "Your cart is empty."
    
    cart_display = "\n".join(f"- {item['name']} (${item['price']})" for item in cart_items)
    spent = sum(item["price"] for item in cart_items)
    remaining = budget - spent
    
    return f"Your Cart ({len(cart_items)} items):\n{cart_display}\n\nBudget: ${budget:.2f} | Spent: ${spent:.2f} | Remaining: ${remaining:.2f}"
#new tool to adjust custom state
@tool
def add_to_cart(item_name: str, item_price: float, runtime: ToolRuntime) -> Command:
    """Add an item to the shopping cart with its price."""
    # Read current cart
    current_cart = runtime.state.get("cart_items", [])

    # Create new item
    new_item = {"name": item_name, "price": item_price}

     # Update cart
    updated_cart = current_cart + [new_item]

     # Return Command to update state
    return Command(
        update={
            "cart_items": updated_cart,
            "messages": [
                ToolMessage(
                    content=f"Added '{item_name}' (${item_price}) to cart. Cart now has {len(updated_cart)} items.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool
def remove_from_cart(item_name: str, runtime: ToolRuntime) -> Command:
    """Remove an item from the shopping cart."""
    current_cart = runtime.state.get("cart_items", [])

    # Check if item exists
    item_exists = any(item["name"] == item_name for item in current_cart)

    if not item_exists:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"'{item_name}' is not in your cart.",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )
    
    # Remove item (removes first occurrence)
    updated_cart = []
    removed = False
    for item in current_cart:
        if item["name"] == item_name and not removed:
            removed = True
        else:
            updated_cart.append(item)

        return Command(
        update={
            "cart_items": updated_cart,
            "messages": [
                ToolMessage(
                    content=f"Removed '{item_name}' from cart. {len(updated_cart)} items remaining.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )



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

# Create agent with state-reading tools
shopping_agent_v2 = create_agent(
    model=model,
    tools=[search_products, check_budget, view_cart],
    state_schema=ShoppingState,
    checkpointer=checkpointer
)

print("Agent created with state-reading tools! -> this means that now agent now knows how to reasd it")

# Create final agent with all tools
full_shopping_agent = create_agent(
    model=model,
    tools=[search_products, check_budget, view_cart, add_to_cart, remove_from_cart],
    state_schema=ShoppingState,
    checkpointer=checkpointer
    )

print("Full shopping agent created with state read/write!")

# Test the state-reading tools
config = {"configurable": {"thread_id": "test-shopping-1"}}

# Initialize state with some budget
initial_state = {
    "messages": [{"role": "user", "content": "What's in my cart?"}],
    "budget": 2000.0,
    "cart_items": [
        {"name": "MacBook Pro", "price": 1299},
        {"name": "Mouse", "price": 29}
    ]
}

result = shopping_agent_v2.invoke(initial_state, config=config)
print(result["messages"][-1].content)

# Test the full agent with state updates
config = {"configurable": {"thread_id": "shopping-demo-1"}}

def chat(message: str):
    """Helper function to chat with the agent."""
    result = full_shopping_agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    response = result['messages'][-1].content
    print(f"\nAgent: {response}\n")
    return result

# Initialize with budget
initial_state = {
    "messages": [{"role": "user", "content": "Hi! I have a $2000 budget for shopping."}],
    "budget": 2000.0,
    "cart_items": []
}
full_shopping_agent.invoke(initial_state, config=config)
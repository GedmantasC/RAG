from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

# Define custom state by extending AgentState
#The class provides structured memory for the agent.
#AgentState allows not just to remember text messages between 
#conversation, but also extra info like chart
class ShoppingState(AgentState):
    cart_items: list[str]
    budget: float
    preferred_category: str

class ResearchState(AgentState):
    topics_explored: list[str]
    findings: dict[str, str]
    source_urls: list[str]

#this one cleans everything before first run, so for sure
#agent could start working
class ResearchMiddleware(AgentMiddleware):
    state_schema = ResearchState

    def before_model(self, state, runtime):
        # Initialize empty state if first run
        if not state.get("topics_explored"):
            return {
                "topics_explored": [],
                "findings": {},
                "source_urls": []
            }
        return None

@tool
def check_budget_remaining(item_price: float, runtime: ToolRuntime) -> str:
    """Check if an item fits within the user's budget."""
    # Access state via runtime - this parameter is HIDDEN from the LLM
    #all these parameter are actually hidden from LLM, so not
    #to leak info if not needed
    budget = runtime.state.get("budget", 0)
    cart_items = runtime.state.get("cart_items", [])

    spent = len(cart_items) * 50  # Simplified calculation
    remaining = budget - spent

    if item_price <= remaining:
        return f"Yes, ${item_price} fits in your ${remaining} remaining budget"
    else:
        return f"No, only ${remaining} left in budget"

@tool
def add_item_to_cart(item_name: str, runtime: ToolRuntime) -> Command:
    """Add an item to the shopping cart."""
    current_cart = runtime.state.get("cart_items", [])
    updated_cart = current_cart + [item_name]
    # Return Command to update state
    return Command(update={
        "cart_items": updated_cart,
        "messages": [
            ToolMessage(
                content=f"Added {item_name} to cart. Cart now has {len(updated_cart)} items.",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def view_cart(runtime: ToolRuntime) -> str:
    """Show what's in the cart."""
    cart = runtime.state.get("cart_items", [])
    if not cart:
        return "Cart is empty"
    return f"Cart contains: {', '.join(cart)}"

#this tool allow to save tokens, if user asks to do something 
#big
@wrap_tool_call
def authorize_tools(request, handler):
    """Only allow certain tools based on user tier."""
    tool_name = request.tool_call["name"]
# Imagine we have user tier in context
    # user_tier = request.runtime.context.tier

    # For demo, we'll block "expensive" tools
    expensive_tools = ["web_scrape_full_site", "train_model"]

    if tool_name in expensive_tools:
            return ToolMessage(
                content=f"Tool '{tool_name}' requires premium subscription",
                tool_call_id=request.tool_call["id"]
            )

        # Allow the tool to run
    return handler(request)

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Catch tool errors and return friendly messages."""

# The LLM only sees: check_budget_remaining(item_price: float)
# It doesn't know about the runtime parameter

# Pass it to create_agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_products, add_to_cart],
    state_schema=ShoppingState #this tells how to store conversation
)

# Now you can initialize state when invoking
agent.invoke({
    "messages": [{"role": "user", "content": "Find laptops under $1000"}],
    "cart_items": [],
    "budget": 1000.0,
    "preferred_category": "electronics"
})

# First invocation
agent.invoke(
    {"messages": [{"role": "user", "content": "Research AI agents"}]},
    {"configurable": {"thread_id": "research-session-1"}}
)

# Second invocation - agent still has state from first call
agent.invoke(
    {"messages": [{"role": "user", "content": "What have we learned so far?"}]},
    {"configurable": {"thread_id": "research-session-1"}}  # Same thread
)
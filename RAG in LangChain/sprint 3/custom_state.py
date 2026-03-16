from langchain.agents import create_agent, AgentState

# Define custom state by extending AgentState
class ShoppingState(AgentState):
    cart_items: list[str]
    budget: float
    preferred_category: str

# Pass it to create_agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_products, add_to_cart],
    state_schema=ShoppingState
)

# Now you can initialize state when invoking
agent.invoke({
    "messages": [{"role": "user", "content": "Find laptops under $1000"}],
    "cart_items": [],
    "budget": 1000.0,
    "preferred_category": "electronics"
})
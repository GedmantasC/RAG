from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware

# Define custom state by extending AgentState
#The class provides structured memory for the agent.
class ShoppingState(AgentState):
    cart_items: list[str]
    budget: float
    preferred_category: str

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
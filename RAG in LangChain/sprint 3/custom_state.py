from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware

# Define custom state by extending AgentState
#The class provides structured memory for the agent.
class ShoppingState(AgentState):
    cart_items: list[str]
    budget: float
    preferred_category: str

class ResearchState(AgentState):
    topics_explored: list[str]
    findings: dict[str, str]
    source_urls: list[str]

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
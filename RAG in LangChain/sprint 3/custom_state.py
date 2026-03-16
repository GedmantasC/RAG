from langchain.agents import create_agent, AgentState

# Define custom state by extending AgentState
class ShoppingState(AgentState):
    cart_items: list[str]
    budget: float
    preferred_category: str


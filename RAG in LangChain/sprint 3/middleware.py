from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
class InputValidationMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        messages = state.get("messages", [])

        # Keep conversation manageable - only last 20 messages
        if len(messages) > 20:
            return {"messages": messages[-20:]}
            return None  # No changes

#part to create agent. 
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search, calculate],
    middleware=[InputValidationMiddleware()]
)

@wrap_tool_call
def authorize_tools(request, handler):
    """Only allow certain tools based on user tier."""
    tool_name = request.tool_call["name"]

    # Imagine we have user tier in context
    # user_tier = request.runtime.context.tier

    # For demo, we'll block "expensive" tools
    expensive_tools = ["web_scrape_full_site", "train_model"]
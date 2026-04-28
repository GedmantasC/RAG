from langchain.agents.middleware import AgentMiddleware
class InputValidationMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        messages = state.get("messages", [])

        # Keep conversation manageable - only last 20 messages
        if len(messages) > 20:
            return {"messages": messages[-20:]}
            return None  # No changes

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search, calculate],
    middleware=[InputValidationMiddleware()]
)
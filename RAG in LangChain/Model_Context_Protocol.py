from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Connect to MCP server and fetch tools
    client = MultiServerMCPClient({
        "weather": {
            "url": "http://localhost:8000/mcp/",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()

    # Create agent with MCP tools and memory - just like before!
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=tools,  # MCP tools work seamlessly
        system_prompt="You are a helpful weather assistant.",
        checkpointer=checkpointer,
    )

     # Use the agent with thread-based memory
    config = {"configurable": {"thread_id": "user-123"}}
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's the weather in NYC?"}]},
        config=config
    )
    print(response["messages"][-1].content)

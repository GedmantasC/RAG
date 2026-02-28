from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    client = MultiServerMCPClient({
        "weather": {
            "transport": "stdio",
            "command": "python",
            # replace this with the actual server you want to run:
            "args": ["-m", "your_weather_mcp_server_module"]
        }
    })

    tools = await client.get_tools()

    agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=tools,
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "user-123"}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's the weather in NYC?"}]},
        config=config,
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
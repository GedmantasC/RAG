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
from langchain.agents import AgentState
from typing import List, Dict, Any

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
  

    


class TaskState(AgentState):
    id: int  #task id
    title: str      #title of the task
    completed:bool  #completion status
    pass

# Test your state schema
print("TaskState defined!")
print(f"Fields: {TaskState.__annotations__}")
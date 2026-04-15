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
from langchain.tools import tool, ToolRuntime

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
  

    


class TaskState(AgentState):
    tasks: List[Dict[str, Any]] = []  # List of tasks
    next_task_id: int = 1              # Next task ID to assign

# Test your state schema
print("TaskState defined!")
print(f"Fields: {TaskState.__annotations__}")

#takes the agentState that means taking info that goes in all conversation
@tool
def list_tasks(runtime: ToolRuntime) -> str:
    """List all tasks with their completion status."""
    tasks = runtime.state.get("tasks", [])
    
    if not tasks:
        return "No tasks found."
    
    task_lines = []
    for task in tasks:
        status = "[X]" if task["completed"] else "[ ]"
        task_lines.append(f"{status} [{task['id']}] {task['title']}")
    
    task_display = "\n".join(task_lines)
    return f"Your Tasks ({len(tasks)} total):\n{task_display}"

print("list_tasks tool created!")

#this lists_all tasks from the list
@tool
def list_tasks(runtime: ToolRuntime) -> str:
    """List all tasks with their completion status."""
    tasks = runtime.state.get("tasks", [])

    if not tasks:
        return "No tasks found."
    
    task_lines = []
    for task in tasks:
        status = "[X]" if task["completed"] else "[ ]"
        task_lines.append(f"{status} [{task['id']}] {task['title']}")

        task_display = "\n".join(task_lines)
    return f"Your Tasks ({len(tasks)} total):\n{task_display}"
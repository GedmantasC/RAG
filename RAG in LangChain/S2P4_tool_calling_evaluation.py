import os
import json
import httpx
import toml
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI

'''in general what langchain does, it allows to awoid writting huge functions, instead you use @tool define tool as a function and you get a result'''

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

MODEL = "gpt-4o-mini"
client = OpenAI()

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Say 'connection OK' in exactly two words."}],
)
print("API check:", response.choices[0].message.content)
import json
import os
from typing import List, Dict, Callable
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
from sklearn.manifold import TSNE
from adjustText import adjust_text
import matplotlib.pyplot as plt
import toml
import sqlite3
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import getpass
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

client = OpenAI(api_key=secrets["OPENAI_API_KEY"])

# Define tools with the calculator_add function. we tell for LLM what to do if somebody asks to add two numbers. 
#or in more simple language if LLM sees that we are talking of adding two numbers, it can use calculator_add function and get results
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator_add",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {"type": "number"},
                    "num2": {"type": "number"},
                },
                "required": ["num1", "num2"],
            },
        },
    }
]

# Simulate a conversation
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Add 8 and 12."}],
    tools=tools,
)

# Output tool calls (the function call details)
print(completion.choices[0].message.tool_calls)

# Example of calling the function with the extracted arguments
def calculator_add(num1, num2):
    return num1 + num2

# Simulating the tool being invoked
tool_calls = completion.choices[0].message.tool_calls
tool_call = tool_calls[0]
function_name = tool_call.function.name
arguments = eval(tool_call.function.arguments)
result = calculator_add(**arguments)
print(result)
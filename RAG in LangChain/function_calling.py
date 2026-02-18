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

# Define tools with the calculator_add function
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

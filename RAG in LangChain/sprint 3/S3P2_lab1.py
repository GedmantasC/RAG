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

# Step 1: Define custom state by extending AgentState
class ShoppingState(AgentState):
    cart_items: list[dict] = []  # List of items: [{"name": "Laptop", "price": 1299}, ...]
    budget: float = 100.0         # User's budget

# Step 2: Create a simple tool (we'll make it stateful in the next section)
@tool
def search_products(query: str) -> str:
    """Search for products matching the query."""
    # Mock product search
    products = {
        "laptop": "MacBook Pro - $1,299",
        "phone": "iPhone 15 - $799",
        "headphones": "AirPods Pro - $249",
        "tablet": "iPad Air - $599"
    }

    results = [v for k, v in products.items() if query.lower() in k]
    return "\n".join(results) if results else "No products found."



# Create model
model = ChatOpenAI(model="gpt-4o")

shopping_agent = create_agent(
    model=model,
    tools=[search_products],
    state_schema=ShoppingState,  # Pass our custom state schema
    checkpointer=checkpointer
)

print("Shopping agent created with custom state!")
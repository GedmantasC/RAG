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

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#save our vectors in memory
vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
# changed the code and made it simple to have more than one scourse for RAG
# --- Load WEB docs ---
web_loader = WebBaseLoader(
    web_paths=(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://www.delfi.lt/en/politics/olekas-names-conditions-for-belarus-to-resume-potash-fertiliser-transit-120211817",
    )
)
web_docs = web_loader.load()

#load from my computer
word_paths = [
    r"C:\Users\Hp\Documents\TuringCollege\AI Enginering\Mokausi AIE\RAG\RAG in LangChain/cnn_page.docx"
]
word_docs = []
for p in word_paths:
    word_docs.extend(UnstructuredWordDocumentLoader(p).load())

#combine both scourses
docs = web_docs + word_docs

#here we add clear scourse for the metadata to know which one is used for the answer
for d in web_docs:
    d.metadata["source_type"] = "web"
for d in word_docs:
    d.metadata["source_type"] = "docx"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

#here we define how to talk with the llm
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the context to answer the question.
If you don't know, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
  retrieved_docs = vector_store.similarity_search(state["question"])
  return {"context": retrieved_docs}

#main part for talking with llm 
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)

    sources = []
    for doc in state["context"]:
        # WebBaseLoader usually has `source` in metadata; docx you can set it yourself
        sources.append(doc.metadata.get("source") or doc.metadata.get("file_path") or doc.metadata.get("source_type"))

    return {"answer": response.content + "\n\nSources:\n- " + "\n- ".join(map(str, sources))}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Display the compiled graph in the notebook
graph

#the idea is that we create a frame of prompt and as context for the prompt we provide our link. 
result = graph.invoke({"question": "What did Juozas Olekas told?", "context": [], "answer": ""})
print(result["answer"])
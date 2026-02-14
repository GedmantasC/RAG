'''Idea is to have a document, use it as a database and then ask questions. Program 
gives you parts of text that could be used to have results'''
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
from langchain_core.documents import Document
from langchain_core.runnables import chain

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# print(documents)


#here I load the pdf document that later will be used as DB
file_path = "https://raw.githubusercontent.com/langchain-ai/langchain/v0.3/docs/docs/example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)


#here we slpit the text, accordingly to the parameters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

print(all_splits[0])

#connect to openAI

# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

#make embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

#here we take a look to our vectors
assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

#save our vectors in memory
vector_store = InMemoryVectorStore(embeddings)

#here ve index the document
ids = vector_store.add_documents(documents=all_splits)

#provide question and get answer
#keep in mind that it provides vector that is the closed one to the one that we make
results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")

print(results[0])

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

#here before we make enbeding from our query and later calculate distance to have an answer
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print(results[0])

#this is more efficient way to have retriev
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
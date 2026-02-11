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



file_path = "https://raw.githubusercontent.com/langchain-ai/langchain/v0.3/docs/docs/example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
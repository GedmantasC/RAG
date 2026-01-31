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

'''Each document contains:
id: Unique identifier
name: Person's name
text: 2-4 sentence biography including birth/death years, nationality, profession, achievements, and legacy'''

file_id = '1WqlveKpz2RKTdhURBUqsdnX6UHx-z03J'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
response = requests.get(url)
documents = response.json()

# Preview the data structure
print("\nSample document:")
print(f"id: {documents[2]['id']}")
print(f"name: {documents[2]['name']}")
print(f"text: {documents[2]['text'][:400]}...")

texts = [d["text"] for d in documents]
names = [d["name"] for d in documents]

# Load API key
secrets = toml.load(".key/secrets.toml")

client = OpenAI(api_key=secrets["OPENAI_API_KEY"])
resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)
'''Idea is that we have a list of historical people and some text about each of them. By making embedings of the text, we plot similarity between them. We can see who are similar to whom'''
X = np.array([e.embedding for e in resp.data])

X_2d = TSNE(n_components=2, random_state=41, n_jobs=1, init='random').fit_transform(X)

plt.figure(figsize=(10, 7))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=30)

texts_obj = [plt.text(x, y, label, fontsize=9) for x, y, label in zip(X_2d[:, 0], X_2d[:, 1], names)]
adjust_text(texts_obj, force_text=(0,0.1))

plt.title("t-SNE of text-embedding-3-small embeddings")
plt.show()


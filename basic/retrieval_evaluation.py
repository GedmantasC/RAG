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

class TextVectorizer:
    '''In general all functions combined allows to vectorize text and latter to compare the similarity of the vectors accordingly'''
    def __init__(self, api_key=None):
        """
        Initialize the vectorizer with OpenAI API credentials.

        :param api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        # Use the API key from environment or passed parameter
        self.client = OpenAI(api_key=api_key)

        # Default embedding model
        self.model = "text-embedding-3-small"

    def vectorize(self, text):
        """
        Convert input text into a vector using OpenAI's embedding model.

        :param text: Input text to vectorize
        :return: Numpy array of the text embedding
        """
        try:
            # Request embedding from OpenAI
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )

            # Extract the embedding vector
            embedding = response.data[0].embedding
            return np.array(embedding)

        except Exception as e:
            print(f"Error during vectorization: {e}")
            return None


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

# plt.figure(figsize=(10, 7))
# plt.scatter(X_2d[:, 0], X_2d[:, 1], s=30)

texts_obj = [plt.text(x, y, label, fontsize=9) for x, y, label in zip(X_2d[:, 0], X_2d[:, 1], names)]
adjust_text(texts_obj, force_text=(0,0.1))

# plt.title("t-SNE of text-embedding-3-small embeddings")
# plt.show()

file_id = '1WjGorxTd2ywiFg8VXyfjTPa871pWnkrd'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
response = requests.get(url)
QUERIES = response.json()

print(f"Loaded {len(QUERIES)} queries:")
print(f"  - Easy: {len([q for q in QUERIES if q['difficulty'] == 'easy'])}")
print(f"  - Hard: {len([q for q in QUERIES if q['difficulty'] == 'hard'])}\n\n")

for k,v in QUERIES[0].items(): print(k, ':', v)

# Function to save embeddings to a SQLite database
def save_embeddings_to_db(embeddings, db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding BLOB
        )
    ''')

    for text, embedding in embeddings.items():
        # Convert the NumPy array to a bytes object
        embedding_bytes = embedding.tobytes()
        cursor.execute("INSERT INTO embeddings (text, embedding) VALUES (?, ?)", (text, embedding_bytes))

    conn.commit()
    conn.close()

embeddings_dict = {}
for text in texts:
  embeddings_dict[text] = vectorizer.vectorize(text)
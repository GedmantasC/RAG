
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import toml
from openai import OpenAI

# Load API key
secrets = toml.load(".key/secrets.toml")

client = OpenAI(api_key=secrets["OPENAI_API_KEY"])


# Function to generate embeddings using OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
  rsp = client.embeddings.create(
      model=model,
      input=text
    )
  return rsp.data[0].embedding

# Create a list of texts
texts = ["Hi", "car", "tourist", "hello", "vehicle"]
# Generate embeddings for the texts
embeddings = [get_embedding(text) for text in texts]

# Create a DataFrame with texts and their embeddings
flattened_df = pd.DataFrame({
    'text': texts,
    'embeddings': embeddings
})

# Create the search index using NearestNeighbors with the nearest 5 neighbors.
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(flattened_df['embeddings'].to_list())

# Query the index to find nearest neighbors
# We'll find neighbors for each embedding
for i, text in enumerate(texts):
  distances, indices = nbrs.kneighbors([embeddings[i]])

  print(f"\nNearest neighbors for '{text}':")
  for dist, idx in zip(distances[0], indices[0]):
      if idx != i:  # Exclude the original text itself
          print(f"- {flattened_df.iloc[idx]['text']} (Distance: {dist:.4f})")
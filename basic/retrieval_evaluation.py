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


file_id = '1WqlveKpz2RKTdhURBUqsdnX6UHx-z03J'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
response = requests.get(url)
documents = response.json()

# Preview the data structure
print("\nSample document:")
print(f"id: {documents[2]['id']}")
print(f"name: {documents[2]['name']}")
print(f"text: {documents[2]['text'][:400]}...")
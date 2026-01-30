import numpy as np
import sqlite3
from openai import OpenAI
import toml

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

    def compare_vectors(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors.

        :param vector1: First vector
        :param vector2: Second vector
        :return: Cosine similarity score
        """
        if vector1 is None or vector2 is None:
            return None

        # Compute cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)

        return dot_product / (norm_vector1 * norm_vector2)
    
# Load API key
secrets = toml.load(".key/secrets.toml")
    
text_vectorizer = TextVectorizer(api_key=secrets["OPENAI_API_KEY"])

embedding = text_vectorizer.vectorize("The food was delicious and the waiter...")

#here the similarity between sentences are calculated. The closer the score is to 1 the more similar senetences are
v1 = text_vectorizer.vectorize("The food was amazing")
v2 = text_vectorizer.vectorize("The meal was delicious")

similarity=text_vectorizer.compare_vectors(v1, v2)  # high similarity

print(embedding[:10])
print(similarity)
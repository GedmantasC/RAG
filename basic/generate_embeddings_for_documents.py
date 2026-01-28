from openai import OpenAI
import toml
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import bs4


# Load API key
secrets = toml.load(".key/secrets.toml")

client = OpenAI(api_key=secrets["OPENAI_API_KEY"])
model = "text-embedding-3-small"

car_embedding = client.embeddings.create(
  model=model,
  input="car",
)

bus_embedding = client.embeddings.create(
  model=model,
  input="bus",
)

sky_embedding = client.embeddings.create(
  model=model,
  input="sky",
)

dog_embedding = client.embeddings.create(
  model=model,
  input="dog",
)

print(car_embedding.data[0].embedding[:10])



# Step 1: Load the raw data or documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()

# Step 2: Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Step 3: Generate embeddings
# Use OpenAI Embeddings (requires OpenAI API key)
openai_embeddings = OpenAIEmbeddings(api_key=secrets["OPENAI_API_KEY"])

# Step 4: Store embeddings in a vector database (FAISS in this case)
vector_store = FAISS.from_documents(chunks, openai_embeddings)

# Step 5: Query the vector store (optional)
query = "What is the best way to restart a router?"
query_embedding = vector_store.similarity_search(query, k=3)

# Step 6: Output results
print("Relevant Chunks:")
for result in query_embedding:
    print(result.page_content)

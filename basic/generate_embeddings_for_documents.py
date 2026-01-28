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
'''the hole idea is that we can make coordinates from the word. To do that we are using model, which is trained with many words. This print statement print 10 first coordinates out of 1.5k that each word has.
If we print more coordinates of the words, similar words like car and bus, would have close coordinates.'''
print(car_embedding.data[0].embedding[:10])



# Step 1: Load the raw data or documents
#here we prepare to load the web page, without ads, bars etc.
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
#after taking text we make small chunks of 500 symbols and we have overlap of 50 symbols between chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(chunks)

# Step 3: Generate embeddings
# Use OpenAI Embeddings (requires OpenAI API key)
#prepare to embedit a.k.a to have coordinates of words and according to that later we will embedit
openai_embeddings = OpenAIEmbeddings(api_key=secrets["OPENAI_API_KEY"])

# Step 4: Store embeddings in a vector database (FAISS in this case)
#we make coordinates of each chunk and store it into FAISS - special place to keep such an info
vector_store = FAISS.from_documents(chunks, openai_embeddings)

# Step 5: Query the vector store (optional)
#query is embedded. FAISS compares it to all stored embeddings. Finds the top 3 most similar chunks, because k=3. 
query = "What is the best way to restart a router?"
query_embedding = vector_store.similarity_search(query, k=3)

# Step 6: Output results
#prints top3 closes chunks to our query
print("Relevant Chunks:")
for result in query_embedding:
    print(result.page_content)

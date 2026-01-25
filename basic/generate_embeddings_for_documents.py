from openai import OpenAI
import toml

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

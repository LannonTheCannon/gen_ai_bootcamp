from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import yaml

OPENAI_API_KEY = yaml.safe_load(open('../../../credentials.yml'))['openai']
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

vectorstore = Chroma(
    persist_directory="data/chroma_cannondale.db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
)

docs = vectorstore.similarity_search("Tell me about Trail 5")

print(docs)

# print(docs[0])

# print(docs[0].metadata)

#print(docs[0].metadata.get("main_image"))
# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# CHALLENGE 1: CREATE A DATA SCIENCE EXPERT USING THE INTRODUCTION TO STATISTICAL LEARNING WITH PYTHON PDF

# DIFFICULTY: BEGINNER

# SPECIFIC ACTIONS:
#  1. USE PDF LOADER TO LOAD THE PDF AND PROCESS THE TEXT
#  2. CREATE A VECTOR DATABASE TO STORE KNOWLEDGE FROM THE BOOK'S PDF
#  3. CREATE A WEB APP THAT INCORPORATES Q&A AND CHAT MEMORY


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import yaml
from pprint import pprint


OPENAI_API_KEY = yaml.safe_load(open('../credentials.yml'))['openai']

# PDF Loader 

loader = PyPDFLoader("challenges/solution_01_statistical_learning_pdf/pdf/ISLP_website.pdf")

# THIS TAKES 5 MINUTES...
documents = loader.load()

CHUNK_SIZE = 1000

text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    # chunk_overlap=100,
    separator="\n"
)

docs = text_splitter.split_documents(documents)

docs

len(docs)

docs[0]

pprint(dict(docs[5])["page_content"])

docs[5]

# Vector Database

embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=OPENAI_API_KEY
)

vectorstore = Chroma.from_documents(
    docs, 
    persist_directory="challenges/solution_01_statistical_learning_pdf/data/chroma_statistical_learning.db",
    embedding=embedding_function
)


vectorstore = Chroma(
    persist_directory="challenges/solution_01_statistical_learning_pdf/data/chroma_statistical_learning.db",
    embedding_function=embedding_function
)

retriever = vectorstore.as_retriever()

retriever

# RAG LLM Model

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(
    model = 'gpt-3.5-turbo',
    temperature = 0.7,
    api_key=OPENAI_API_KEY
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke("What are the top 3 things needed to do principal component analysis (pca)?")

pprint(result)


result = rag_chain.invoke("How many types of linear regression are common?")

pprint(result)

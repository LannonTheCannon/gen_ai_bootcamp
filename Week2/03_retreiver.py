# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goals: Intro to ... 
# - Document Retrieval
# - Augmenting LLMs with the Expert Information (CONTEXT)... i.e. Making an basic LLM a Custom Expert

# LIBRARIES 

# UPDATE: Chroma is now a separate package
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import yaml
import os

from pprint import pprint
from IPython.display import Markdown

# Key Parameters
RAG_DATABASE    = "Week2/data/chroma_3.db"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM             = "gpt-4o-mini"

# OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('credentials.yml'))['openai']


# 1.0 CREATE A RETRIEVER FROM THE VECTORSTORE 

embedding_function = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

vectorstore = Chroma(
    persist_directory=RAG_DATABASE,
    embedding_function=embedding_function
)

retriever = vectorstore.as_retriever()

retriever

# 2.0 USE THE RETRIEVER TO AUGMENT AN LLM

# * Prompt template 

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# * LLM Specification

model = ChatOpenAI(
    model = LLM,
    temperature = 0.7,
)

response = model.invoke("What are the top 3 things needed in a good social media marketing strategy for Facebook (Meta)? Site any sources used.")

pprint(response.content)
Markdown(response.content)

# * Combine with Lang Chain Expression Language (LCEL)
#   - Context: Give it access to the retriever
#   - Question: Provide the user question as a pass through from the invoke method
#   - Use LCEL to add a prompt template, model spec, and output parsing

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

rag_chain

# * Try it out:

# * Baseline
result_baseline = model.invoke("What are the top 3 things needed in a good social media marketing strategy for Facebook (Meta)? Site any sources used.")

Markdown(result_baseline.content)

# * RAG
result = rag_chain.invoke("What are the top 3 things needed in a good social media marketing strategy for Facebook (Meta)? Cite any sources used with the URL.")

Markdown(result)



# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# CHALLENGE 2: CREATE AN EXPERT IN CANNONDALE BICYCLES BIKE MODELS. GOAL IS TO USE THE LLM TO HELP RECOMMEND BICYCLES TO USERS.

# WEBSITE: https://www.cannondale.com/en-us

# DIFFICULTY: INTERMEDIATE

# SPECIFIC ACTIONS:
#  1. USE WEB LOADER TO LOAD WEBPAGES AND STORE THE TEXT WITH METADATA
#  2. CREATE A VECTOR DATABASE TO STORE KNOWLEDGE 
#  3. CREATE A WEB APP THAT INCORPORATES Q&A AND CHAT MEMORY

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import requests
from bs4 import BeautifulSoup
import html2text

import pandas as pd
import yaml

# OPENAI_API_KEY = yaml.safe_load(open('../../../credentials.yml'))['openai']
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

# Load the serialized string from the file
with open('challenges/solution_02_website_llm_web_loader/data/bikes.html', 'r', encoding='utf-8') as file:
    soup_string = file.read()

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

OPENAI_API_KEY = yaml.safe_load(open("../../../credentials.yml"))['openai']

# 1.0 Get Website Data

# https://www.cannondale.com/en-us/bikes

# I manually saved the website html as bikes.html

# 2.0 Beautiful Soup

# Load the serialized string from the file
with open('data/bikes.html', 'r', encoding='utf-8') as file:
    soup_string = file.read()

# Parse the string back into a BeautifulSoup object
soup = BeautifulSoup(soup_string, 'html.parser')

# 3.0 Parsing the Product Cards

# Find all product card divs
product_cards = soup.find_all('div', class_='product-card')

products = []


for card in product_cards:
    # Extract product title
    title_element = card.find('h4', class_='product-card__title')
    title = title_element.get_text(strip=True) if title_element else 'N/A'

    # Extract product subtitle
    subtitle_element = card.find('span', class_='product-card__subtitle')
    subtitle = subtitle_element.get_text(strip=True) if subtitle_element else 'N/A'

    # Extract product price
    price_element = card.find('span', class_='product-card__price-main')
    price = price_element.get_text(strip=True) if price_element else 'N/A'

    # Extract product description
    description_element = card.find('p', class_='product-card__description')
    description = description_element.get_text(strip=True) if description_element else 'N/A'

    # Extract product link
    link_element = card.find('a', class_='content product product-card__link')
    link = link_element['href'] if link_element else 'N/A'

    # Extract main image URL
    main_image_element = card.find('picture', class_='product-card__main-image')
    main_image = main_image_element.find('img')['src'] if main_image_element and main_image_element.find('img') else 'N/A'

    # Extract alternate image URL
    alternate_image_element = card.find('picture', class_='product-card__3Q-image')
    alternate_image = alternate_image_element.find('img')['src'] if alternate_image_element and alternate_image_element.find('img') else 'N/A'

    product_details = {
        'title': title,
        'subtitle': subtitle,
        'price': price,
        'description': description,
        'link': link,
        'main_image': main_image,
        'alternate_image': alternate_image
    }

    products.append(product_details)

products

len(products)

# I stored as a CSV
pd.DataFrame(products).to_csv("data/bikes.csv", index=False)

# 4.0 Make LangChain Documents

documents = []

for product in products:
    
    content = f"""
    title: {product.get("title")}
    subtitle: {product.get("subtitle")}
    price: {product.get("price")}
    description: {product.get("description")}
    """
    # print(content)
    
    doc = Document(page_content=content, metadata=product)
    
    documents.append(doc)

documents

len(documents)

print(documents[0].page_content)

print(documents[0].metadata)

# 5.0 Vector Database

embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=OPENAI_API_KEY
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
    persist_directory="data/chroma_cannondale.db",
)


# vectorstore = Chroma(
#     embedding_function=embedding_function,
#     persist_directory="challenges/solution_02_website_llm_web_loader/data/chroma_cannondale.db",
# )

retriever = vectorstore.as_retriever()

retriever

# 6.0 RAG LLM Model

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(
    model = 'gpt-3.5-turbo',
    temperature = 0,
    api_key=OPENAI_API_KEY
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke("What is a good mountain bike under $1000. Describe it.")

result


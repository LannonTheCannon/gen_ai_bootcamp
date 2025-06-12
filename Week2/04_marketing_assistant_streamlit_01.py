# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# streamlit run path_to_app.py

# Goal: Take what you've learned and show how to put it into a simple Q&A streamlit app.

# APP #1: SINGLE Q&A
# - GREAT FOR SIMPLE 1-OFF TASKS (E.G. WRITE ME AN EMAIL)
# - PROBLEM: NO CHAT CAPABILITY


question = """
Draft an email to a prospective client to introduce your social media marketing services. Give 3 tips based on experts in the space from our AI database and site your sources. Transition to the next problem which is how to convert leads into customers. See if the client would like to schedule a 15-minute call to discuss further. 
"""

# Imports 

# UPDATE: Chroma is now a separate package
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import yaml
import os

# * New: Add streamlit
import streamlit as st

# Key Parameters
RAG_DATABASE    = "Week2/data2/chroma_3.db"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM             = "gpt-4o-mini"

# *New: Initialize the Streamlit app
st.set_page_config(
    page_title="Your Books Assistant", 
    layout="wide"
)

# Load the API Key securely
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']

# Function to create the processing chain
def create_chain():
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vectorstore = Chroma(
        persist_directory=RAG_DATABASE,
        embedding_function=embedding_function
    )

    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(
        model=LLM,
        temperature=0.7,
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain



# Main app interface
st.title("Your Marketing AI Copilot")

# Layout for input and output
col1, col2 = st.columns(2)

with col1:
    st.header("How can I help you?")
    # file = st.file_uploader("Upload a file with your marketing question")
    question = st.text_area("Enter your Marketing question here:", height=300)
    submit_button = st.button("Submit")

with col2:
    st.header("AI Generated Answer")
    if submit_button and question:
        # Process the question through the chain
        with st.spinner("Processing your question..."):
            try:
                # Process the question through the chain
                chain = create_chain()
                answer = chain.invoke(question)
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        
        st.write("Your answer will appear here.")



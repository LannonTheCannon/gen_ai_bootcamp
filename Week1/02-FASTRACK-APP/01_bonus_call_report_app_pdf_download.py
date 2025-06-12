# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# CHALLENGE - CREATE A CALL REPORT APP
# ***
# GOAL: Exposure to using LLM's, Document Loaders, and Prompts

# streamlit run path_to_app
# streamlit run 02-FASTRACK-APP/01_bonus_call_report_app_pdf_download.py

import yaml
import streamlit as st

import subprocess
import os
from tempfile import NamedTemporaryFile
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# *NEW* - GRAB CHAT OLLAMA FIRST
# from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

# * NEW* - STRIP THINKING (Used for Chain-Of-Thought models)
# This function removes the <think>...</think> block from the content
import re
def strip_thinking(content: str) -> str:
    # Remove the <think>...</think> block
    return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)

# * APP INPUTS ----

MODEL_LIST = ['llama3.2:3b', 'deepseek-r1:8b']

# * STREAMLIT INTERFACE ----

# Streamlit Interface
st.set_page_config(layout='wide', page_title="Call Transcript Summarizer")
st.title('Earnings Call Transcript Summarizer')
col1, col2 = st.columns(2)

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose Ollama model (must be installed locally)",
    MODEL_LIST,
    index=0
)

# * FUNCTIONS ----

# Load API Key
# OPENAI_API_KEY = yaml.safe_load(open('../credentials.yml'))['openai']
# MODEL = "gpt-4o-mini"

def generate_pdf_with_quarto(markdown_text):
    with NamedTemporaryFile(delete=False, suffix=".qmd", mode='w') as md_file:
        md_file.write(markdown_text)  # Write string directly
        md_file_path = md_file.name

    pdf_file_path = md_file_path.replace('.qmd', '.pdf')
    
    # Use the Quarto command line instead of Python integration for more complex rendering
    subprocess.run(["quarto", "render", md_file_path, "--to", "pdf"], check=True)
    
    os.remove(md_file_path)  # Clean up the Markdown file
    return pdf_file_path

def move_file_to_downloads(pdf_file_path):
    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    destination_path = os.path.join(downloads_path, os.path.basename(pdf_file_path))
    shutil.move(pdf_file_path, destination_path)
    return destination_path

def load_and_summarize(file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        file_path = tmp.name
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        prompt_template = """
        Write a business report from the following earnings call transcript:
        {text}

        Use the following Markdown format:
        # Insert Descriptive Report Title

        ## Earnings Call Summary
        Use 3 to 7 numbered bullet points

        ## Important Financials:
        Describe the most important financials discussed during the call. Use 3 to 5 numbered bullet points.

        ## Key Business Risks
        Describe any key business risks discussed on the call. Use 3 to 5 numbered bullets.

        ## Conclusions
        Conclude with any overarching business actions that the company is pursuing that may have positive or negative implications and what those implications are. 
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # * NEW* - USE OLLAMA MODEL
        # model = ChatOpenAI(
        #     model=MODEL,
        #     temperature=0,
        #     api_key=OPENAI_API_KEY
        # )
        
        model = ChatOllama(
            model=model_option,
        )

        llm_chain = LLMChain(llm=model, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        response = stuff_chain.invoke(docs)
    finally:
        os.remove(file_path)

    return response['output_text']


with col1:
    st.subheader('Upload a PDF document:')
    uploaded_file = st.file_uploader("Choose a file", type="pdf", key="file_uploader")
    if uploaded_file:
        summarize_flag = st.button('Summarize Document', key="summarize_button")
        

if uploaded_file and summarize_flag:
    with col2:
        with st.spinner('Summarizing...'):
            summaries = load_and_summarize(uploaded_file)
            
            # * NEW* - STRIP THINKING
            summaries = strip_thinking(summaries)
            
            st.subheader('Summarization Result:')
            st.markdown(summaries)
            
            pdf_file = generate_pdf_with_quarto(summaries)
            download_path = move_file_to_downloads(pdf_file)
            st.markdown(f"**PDF Downloaded to your Downloads folder: {download_path}**")

else:
    with col2:
        st.write("No file uploaded. Please upload a PDF file to proceed.")




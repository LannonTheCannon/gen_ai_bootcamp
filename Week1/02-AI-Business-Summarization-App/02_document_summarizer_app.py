# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# FIRST AI-POWERED BUSINESS APP: PART 2
# ***
# GOAL: Exposure to using LLM's, Document Loaders, and Prompts

# streamlit run 03-First-AI-Business-App/02_document_summarizer_app.py


import yaml

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import streamlit as st
import os
from tempfile import NamedTemporaryFile

# Load API Key
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

# 1.0 LOAD AND SUMMARIZE FUNCTION
def load_and_summarize(file, use_template=False): 
    with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp: 
        tmp.write(file.getvalue())
        file_path = tmp.name 

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        model = ChatOpenAI(
                    model="gpt-3.5-turbo", 
                    temperature=0,
                    api_key=OPENAI_API_KEY,
                    max_tokens=None,
                )

        if use_template: 
            # Bullets
            prompt_template = """
            Write a business report from the following earning call transcript: 
            {text}

            Use the following markdown format: 
            # Insert Descriptive Report Title 

            ## Earnings Call Summary 
            Use 3 to 7 numbered bullet points to describe key points.

            ## Important Financials: 
            Describe the most important financials discussed during the call. Use 3 - 5 
            numbered bullets 

            ## Conclusions 
            Conclude with any overarching business actions that the company is purusing 
            that maye a positive or negative implications and what those implications are. 

            """

            prompt = PromptTemplate(input_variables=['text'], template=prompt_template)
            print(prompt)

            llm_chain = LLMChain(prompt=prompt, llm=model)

            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            response = stuff_chain.invoke(docs)

        else:
            # No bullets
            summarizer_chain = load_summarize_chain(llm=model, chain_type='stuff')
            response = summarizer_chain.invoke(docs)

    finally: 
        os.remove(file_path)

    return response['output_text']


# 2.0 STREAMLIT INTERFACE
st.title("PDF Children of Ruin Book Summarizer")
st.subheader('Upload file: ')
uploaded_file = st.file_uploader("Choose a file: (.pdf)", type="pdf")

if uploaded_file is not None:
    st.write('A file has been uploaded!')
    use_template = st.checkbox("Use numbered bullet points? (If not paragraph will be returned)")
    
    summary = load_and_summarize(uploaded_file, use_template)
    st.subheader('Summarization Results')
    st.markdown(summary)

            

else: 
    st.write("No file uploaded (Upload a pdf to proceed)")

# CONCLUSIONS:
#  1. WE CAN SEE HOW APPLICATIONS LIKE STREAMLIT ARE A NATURAL INTERFACE TO AUTOMATING THE LLM TASKS
#  2. BUT WE CAN DO MORE. 
#     - WHAT IF WE HAD A FULL DIRECTORY OF PDF'S?
#     - WHAT IF WE WANTED TO DO MORE COMPLEX ANALYSIS?

# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# streamlit run path_to_app.py

# Chat Q&A Framework for RAG Apps
# GOAL: CONNECT MESSAGE CONTEXT (MEMORY) WITH RAG

# Key Modifications:
# 1. Persistent Chat History: Utilizing Streamlit's session state, this setup remembers past messages across reruns of the app. Each user interaction and the corresponding assistant response are appended to a message list.
# 2. Using Chat Components for Display: Each message, whether from the user or the AI assistant, is displayed within a st.chat_message context, clearly distinguishing between the participants.

question = """
Draft an email to a prospective client to introduce your social media marketing services. Give 3 tips based on experts in the space from our AI database and site your sources and the URL links. Transition to the next problem which is how to convert leads into customers. See if the client would like to schedule a 15-minute call to discuss further. 
"""

prompt_2 = """
Modify the prospect's name to Dave. Modify my name is Matt and I'm the Founder of Business Science. 
"""

prompt_2 = """
Switch the 3rd point to focus on Facebook Ads. This is one of our strong suits.  
"""

# Imports 

# * UPDATE: Chroma is now a separate package
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st
import yaml
import os

# Key Parameters
RAG_DATABASE    = "Week2/data2/chroma_2.db" # SWAP THIS OUT FOR YOUR RAG DATABASE
EMBEDDING_MODEL = "text-embedding-3-large"
# LLM             = "gpt-4o-mini"

# * STREAMLIT APP SETUP ----

CHAT_LLM_OPTIONS = ["gpt-4o-mini", "gpt-4o"]

# * Page Setup

# Initialize the Streamlit app
st.set_page_config(page_title="Your Marketing AI Copilot")
st.title("Your Marketing AI Copilot")

# Load the API Key securely
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']


# * NEW: Set up model selection

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose OpenAI model",
    CHAT_LLM_OPTIONS,
    index=0
)

LLM = model_option

# * NEW: Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

def create_rag_chain():
    
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        # api_key=api_key,
        chunk_size=500,
    )
    vectorstore = Chroma(
        persist_directory=RAG_DATABASE,
        embedding_function=embedding_function
    )
    
    retriever = vectorstore.as_retriever()      
    
    llm = ChatOpenAI(
        model=LLM, 
        temperature=0.7, 
        # max_tokens=4000,
    )

    # * NEW: COMBINE CHAT HISTORY WITH RAG RETREIVER
    # * 1. Contextualize question: Integrates RAG
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # * 2. Answer question based on Chat Context
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # * Combine both RAG + Chat Message History
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

rag_chain = create_rag_chain()

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input("Enter your Marketing question here:", key="query_input"):
    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)     
           
        response = rag_chain.invoke(
            {"input": question}, 
            config={
                "configurable": {"session_id": "any"}
            },
        )
        # Debug response
        # print(response)
        # print("\n")
  
        st.chat_message("ai").write(response['answer'])

# * NEW: View the messages for debugging
# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)




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


# Imports 
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
import requests  # For validating image URLs

# Key Parameters
RAG_DATABASE = "data/chroma_cannondale.db"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_LLM_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]

# Initialize the Streamlit app
st.set_page_config(page_title="Your Cannondale Bike Recommendation AI", layout="wide")
st.title("Your Cannondale Bike Recommendation AI")

# Load the API Key securely
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../../../credentials.yml'))['openai']

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose OpenAI model",
    CHAT_LLM_OPTIONS,
    index=0
)
LLM = model_option

# Set up chat history memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you with bike models? Ask about a specific model to see its details and image!")

# Debugging expander for session messages
view_messages = st.expander("View the message contents in session state")

# Function to validate image URLs
def is_valid_image_url(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200 and 'image' in response.headers.get('content-type', '').lower()
    except requests.RequestException:
        return False
    
def create_rag_chain():
    # Initialize embeddings
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        chunk_size=500,
    )
    
    # Connect to vector store
    vectorstore = Chroma(
        persist_directory=RAG_DATABASE,
        embedding_function=embedding_function
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Retrieve top 1 document for simplicity
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=LLM, 
        temperature=0.7,
    )

    # Contextualize question: Integrates RAG with chat history
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question based on context
    qa_system_prompt = """You are an assistant for question-answering tasks about bike models. \
    Use the following pieces of retrieved context to answer the question concisely. \
    If an image URL is available, include it in the answer by stating 'Main Image URL: [URL]'. \
    If you don't know the answer, say so. Keep the answer to three sentences maximum.\

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Combine RAG + Chat Message History
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ), retriever  # Return retriever for image access


# Create RAG chain and retriever
rag_chain, retriever = create_rag_chain()

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Handle user input
if question := st.chat_input("Enter your question about a bike model:", key="query_input"):
    with st.spinner("Thinking..."):
        # Display user question
        st.chat_message("human").write(question)
        
        # Invoke RAG chain
        response = rag_chain.invoke(
            {"input": question}, 
            config={
                "configurable": {"session_id": "any"}
            },
        )
        
        # Get the answer
        answer = response['answer']
        
        # Display the answer
        st.chat_message("ai").write(answer)
        
        # Retrieve documents to access metadata (for image URLs)
        retrieved_docs = retriever.invoke(question)
        if retrieved_docs and 'main_image' in retrieved_docs[0].metadata:
            main_image_url = retrieved_docs[0].metadata.get('main_image', 'N/A')
            if main_image_url != 'N/A' and is_valid_image_url(main_image_url):
                st.image(main_image_url, caption=f"Image of {retrieved_docs[0].metadata.get('title', 'Bike Model')}", width=300)
            else:
                st.write("No valid image available for this bike model.")

# View messages for debugging
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
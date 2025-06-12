import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import yaml
import os

# --- Load API Key ---
OPENAI_API_KEY = yaml.safe_load(open('../../../credentials.yml'))['openai']
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- PDF Loader & Chunking ---
@st.cache_data
def load_and_split_pdf():
    loader = PyPDFLoader("../solution_01_statistical_learning_pdf/pdf/ISLP_website.pdf")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
    return splitter.split_documents(documents)

# --- Embed & Load VectorStore ---
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    return Chroma(
        persist_directory="challenges/solution_01_statistical_learning_pdf/data/chroma_statistical_learning.db",
        embedding_function=embeddings
    )

# --- Build RAG Chain with Memory ---
def build_rag_chain(retriever, msgs):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=OPENAI_API_KEY)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the user's latest question as a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer concisely using the context. Say 'I don't know' if unsure.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# --- Streamlit App ---
st.set_page_config(page_title="Statistical Learning RAG", layout="wide")
st.title("📘 Ask the Statistical Learning Expert")

# Set up chat memory
msgs = StreamlitChatMessageHistory(key="langchain_messages_stat_learn")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you with Statistical Learning today?")

# Render chat history
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Input & Response Handling
if question := st.chat_input("Ask a question based on the ISLP PDF:"):
    st.chat_message("human").write(question)

    with st.spinner("Thinking..."):
        # Load vectorstore and build RAG chain with memory
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever()
        rag_chain = build_rag_chain(retriever, msgs)

        # Run the full memory-enabled chain
        response = rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "stat_learn_session"}}
        )
        st.chat_message("ai").write(response['answer'])
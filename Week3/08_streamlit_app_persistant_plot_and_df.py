# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# streamlit run path_to_streamlit_app.py

# Goal: Create a streamlit chat app with persistent data frames and plots

# Imports

import streamlit as st
import plotly.express as px

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Initialize Streamlit App

def create_plot():
    df = px.data.iris()
    fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
    return fig

def get_iris_data():
    return px.data.iris()

st.set_page_config(page_title="Your Business Intelligence AI Copilot", layout="centered")
st.title("Your Business Intelligence AI Copilot")

st.markdown("""
            I'm a handy business intelligence agent that connects up to the leads_scored.db SQLite database that mimics an ERP System for a company. You can ask me Business Intelligence, Customer Analytics, and Data Visualization Questions. I will report the results. 
            """)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("Instructions")

with view_messages:
    st.write(
        """
        ### Example Questions
        - Make a plot
        - Make a data frame
        - Answer a question
        """
    )

# Initialize plot storage in session state
if "plots" not in st.session_state:
    st.session_state.plots = []

# Initialize dataframe storage in session state
if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Function to display chat messages including Plotly charts and dataframes
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(st.session_state.plots[plot_index], key=f"history_plot_{plot_index}")
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index], key=f"history_dataframe_{df_index}")
            else:
                st.write(msg.content)

# Render current messages from StreamlitChatMessageHistory
display_chat_history()

if question := st.chat_input("Enter your question here:", key="query_input"):
    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        if "data frame" in question.lower():
            
            response_text = "returning the Iris data frame..."
            iris_data = get_iris_data()

            # Store the dataframe and keep its index
            df_index = len(st.session_state.dataframes)
            st.session_state.dataframes.append(iris_data)

            # Store the response text and dataframe index in the messages
            msgs.add_ai_message(response_text)
            msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")

            st.chat_message("ai").write(response_text)
            st.dataframe(iris_data)
            
        elif "plot" in question.lower():
            
            response_text = "creating a plot..."
            response_plot = create_plot()

            # Store the plot and keep its index
            plot_index = len(st.session_state.plots)
            st.session_state.plots.append(response_plot)

            # Store the response text and plot index in the messages
            msgs.add_ai_message(response_text)
            msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")

            st.chat_message("ai").write(response_text)
            st.plotly_chart(response_plot)
            
        else:
            
            response_text = "Here's your answer..."
            msgs.add_ai_message(response_text)
            st.chat_message("ai").write(response_text)

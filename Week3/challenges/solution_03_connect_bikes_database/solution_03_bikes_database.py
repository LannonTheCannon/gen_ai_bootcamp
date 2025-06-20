# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# CHALLENGE 3: CONNECT YOUR BUSINESS INTELLIGENCE COPILOT TO A NEW DATABASE

# DIFFICULTY: BEGINNER

# SPECIFIC ACTIONS:
#  1. Allow the user to pass a SQL Connection URI through the Streamlit App (PATH_DB should be an input)
#  2. See how to BI Copilot can be used on ANY database

# streamlit run path_to_app

# SAMPLE QUESTIONS:

# what tables are in the database?
# what do the first 5 rows of the bikes table look like?
# Which table contains the transactions?
# Show me the orderlines table
# what does the bikeshop table look like?
# Which table contains the products?

# NOTE: Used gpt-4o for these...
# What are the total sales per year-month? Make sure to calculate a total price by multiplying the bike price by the quantity. Make a chart of sales over time.
# What is the sales by year-month for just Road bicycles. Make sure to calculate a total price by multiplying the bike price by the quantity. Make a chart of sales over time.
# Create a map plot of sales by US state. Make sure to calculate a total price by multiplying the bike price by the quantity.


# LIBRARIES

import streamlit as st
import plotly.express as px

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import os
import yaml

import pandas as pd
import sqlalchemy as sql
import plotly.io as pio

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from business_intelligence_agent import make_business_intelligence_agent

# AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']

# SQL DATABASE SETUP

PATH_DB = "sqlite:///database/leads_scored.db"

sql_engine = sql.create_engine(PATH_DB)

conn = sql_engine.connect()


# * STREAMLIT APP SETUP ----

st.set_page_config(page_title="Your Business Intelligence AI Copilot")
st.title("Your Business Intelligence AI Copilot")

st.markdown("""
            I'm a handy business intelligence agent that connects up to the leads_scored.db SQLite database that mimics an ERP System for a company. You can ask me Business Intelligence, Customer Analytics, and Data Visualization Questions. I will report the results. 
            """)

# * model selection

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose OpenAI model",
    MODEL_LIST,
    index=0
)

# * NEW Database option

db_option = st.sidebar.selectbox(
    "Select a Database",
    ["Bikeshop Database", "Lead Score Database"]
)

db_options = {
    "Bikeshop Database": "sqlite:///challenges/solution_03_connect_bikes_database/bikeshop_database.sqlite",
    "Lead Score Database": "sqlite:///database/leads_scored.db"
}

PATH_DB = db_options.get(db_option)


# * AGENT

app = make_business_intelligence_agent(
    path = PATH_DB,
    model = model_option
)


# * STREAMLIT 

example_questions = st.expander("Try out example questions")

with example_questions:
    """
    Example Questions:
    - what tables are in the database?
    - what do the first 5 rows of the bikes table look like?
    - Which table contains the transactions?
    - Show me the orderlines table
    - what does the bikeshop table look like?
    - Which table contains the products?

    NOTE: Used gpt-4o for these...
    - What are the total sales per year-month? Make sure to calculate a total price by multiplying the bike price by the quantity. Make a chart of sales over time.
    - What is the sales by year-month for just Road bicycles. Make sure to calculate a total price by multiplying the bike price by the quantity. Make a chart of sales over time.
    - Create a map plot of sales by US state. Make sure to calculate a total price by multiplying the bike price by the quantity.
    """
    
# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

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
        
        # Run the app
        inputs = {"user_question": question}
        
        error_occured = False
        try: 
            result = app.invoke(inputs)
        except Exception as e:
            error_occured = True
            print(e)
        
        if not error_occured:

            if result['routing_preprocessor_decision'] == 'table':
                # Table was requested
                
                response_text = f"Returning the table...\n\nSQL Query:\n```sql\n{result['sql_query']}\n```"
                
                response_df = pd.DataFrame(result['data'])

                # Store the dataframe and keep its index
                df_index = len(st.session_state.dataframes)
                
                st.session_state.dataframes.append(response_df)

                # Store the response text and dataframe index in the messages
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")

                st.chat_message("ai").write(response_text)
                st.dataframe(response_df)
                
            elif result['routing_preprocessor_decision'] == 'chart' and result['chart_plotly_error'] is False:
                # Chart was requested and  produced correctly
                
                response_text = f"Returning the plot...\n\nSQL Query:\n```sql\n{result['sql_query']}\n```"
                
                response_plot = pio.from_json(result["chart_plotly_json"])

                # Store the plot and keep its index
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(response_plot)

                # Store the response text and plot index in the messages
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")

                st.chat_message("ai").write(response_text)
                st.plotly_chart(response_plot)
            else:
                # Chart error occurred, return Table instead
                response_text = f"I apologize. There was an error during the plotting process. Returning the table instead...\n\nSQL Query:\n```sql\n{result['sql_query']}\n```"
                
                df = pd.DataFrame(result['data'])

                # Store the dataframe and keep its index
                df_index = len(st.session_state.dataframes)
                
                st.session_state.dataframes.append(df)

                # Store the response text and dataframe index in the messages
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")

                st.chat_message("ai").write(response_text)
                st.dataframe(df)
        else:
            # SQL error occurred
            response_text = f"An error occurred in generating the SQL. I apologize. Please try again or format the question differently and I'll try my best to provide a helpful answer."
            msgs.add_ai_message(response_text)
            st.chat_message("ai").write(response_text)
            

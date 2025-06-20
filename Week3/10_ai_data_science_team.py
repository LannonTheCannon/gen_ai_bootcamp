# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# Goal: Expose you to my AI Data Science Team

# YOUR AI DATA SCIENCE TEAM ARMY OF AGENTS:
# - SQLDataAnalyst
# - SQLDatabaseAgent
# - DataVisualizationAgent

# Requirements:
# pip install ai-data-science-team

# LIBRARIES
import sqlalchemy as sql
import pandas as pd

from langchain_openai import ChatOpenAI
import os
import yaml

from ai_data_science_team.multiagents import SQLDataAnalyst
from ai_data_science_team.agents import SQLDatabaseAgent, DataVisualizationAgent


# * Setup

MODEL    = "gpt-4o-mini"
LOG      = False
LOG_PATH = os.path.join(os.getcwd(), "logs/")

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model = MODEL)

sql_engine = sql.create_engine("sqlite:///database/leads_scored.db")

conn = sql_engine.connect()

# * Make the agent

sql_data_analyst = SQLDataAnalyst(
    model = llm,
    sql_database_agent = SQLDatabaseAgent(
        model = llm,
        connection = conn,
        n_samples = 1,
        log = LOG,
        log_path = LOG_PATH,
        bypass_recommended_steps=True,
    ),
    data_visualization_agent = DataVisualizationAgent(
        model = llm,
        n_samples = 1000,
        log = LOG,
        log_path = LOG_PATH,
    )
)

sql_data_analyst

sql_data_analyst.show(xray=1)

sql_data_analyst.get_state_keys()

# * Easy questions

sql_data_analyst.invoke_agent(
    user_instructions = "What is the average p1 lead score of leads by member rating in the database?",
)

sql_data_analyst.get_data_sql()

sql_data_analyst.get_sql_query_code(markdown=True)

sql_data_analyst.get_sql_database_function(markdown=True)

sql_data_analyst.get_workflow_summary(markdown=True)


# * Data visualization

sql_data_analyst.invoke_agent(
    user_instructions = "What are the total sales by month-year? Use suggested price as a proxy for revenue for each transaction and a quantity of 1. Make a chart of sales over time.",
)


sql_data_analyst.get_data_sql()

sql_data_analyst.get_sql_query_code(markdown=True)

sql_data_analyst.get_plotly_graph()

sql_data_analyst.get_response()['messages']

sql_data_analyst.get_workflow_summary(markdown=True)

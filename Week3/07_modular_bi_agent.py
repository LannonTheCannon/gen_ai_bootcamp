# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# Goal: Convert Agent to a reusable Python Module
# - Create a function to make the agent
# - Realistic Production Example


# LIBRARIES

from business_intelligence_agent import make_business_intelligence_agent

import pandas as pd
import plotly.io as pio
import yaml
import os

from pprint import pprint


# * MAKE THE AGENT

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']

PATH_DB = "sqlite:///database/leads_scored.db"
MODEL = "gpt-4.1-mini"


bi_agent = make_business_intelligence_agent(
    path = PATH_DB,
    model = MODEL
)

bi_agent

# * TEST THE AGENT

# Data Analysis Question

response = bi_agent.invoke({"user_question": "What is the average p1 lead score of leads in the database?"})

list(response.keys())

pprint(response["sql_query"])

pd.DataFrame(response["data"])


# Data Visualization Question

response = bi_agent.invoke({"user_question": "What are the total sales by month-year? Use suggested price as a proxy for revenue for each transaction and a quantity of 1. Make a chart of sales over time. Make the line green in the chart"})


response.keys()

pprint(response["sql_query"])

pd.DataFrame(response["data"])

pio.from_json(response["chart_plotly_json"])


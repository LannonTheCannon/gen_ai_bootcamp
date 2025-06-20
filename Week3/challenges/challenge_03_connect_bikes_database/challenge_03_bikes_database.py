# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# CHALLENGE 3: CONNECT YOUR BUSINESS INTELLIGENCE COPILOT TO A NEW DATABASE

# DIFFICULTY: BEGINNER

# SPECIFIC ACTIONS:
#  1. Allow the user to have a dropdown and select different SQL Connections through the Streamlit App 
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

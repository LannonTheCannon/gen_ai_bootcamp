# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# NOTE BEFORE WE START: 
#  THIS IS AN INTERMEDIATE-TO-ADVANCED PROJECT. 
#  IF YOU ARE JUST STARTING TO LEARN, YOU MIGHT BE UNCOMFORTABLE. THAT'S OK. 
#  SUGGEST YOU WORK THROUGH AI FAST TRACK AND RAG PROJECTS FIRST. 
#  CREATING THIS PROJECT IS SOMETHING TO LEARN FROM AND STRIVE FOR. 
#  HOWEVER, THE FINAL STREAMLIT APP CODE I GIVE YOU CAN BE APPLIED TO ANY DATABASE (JUST COPY AND PASTE).
#  THIS WILL BE DEMONSTRATED IN CHALLENGE 3. 

# Goal: Recap Machine Learning and Email Lead Scoring
#  - Dataset and project is from Python Course 2 (Machine Learning)

# H2O AUTOML -----
# 1. Covered in depth in Module 5 of Python Course 2
# 2. Perform Predictive Lead Scoring
# 3. Upload the lead scores into a SQL database that the AI will have access to 

# 1.0 Imports and Setups ---------------------------------------------------------------------------------
import pandas as pd
import pytimetk as tk
import sqlalchemy as sql
import h2o
from h2o.automl import H2OAutoML


# To check where my current working directory is! 
import os
print("Current working directory:", os.getcwd())


# 2.0 Connects to the Database ---------------------------------------------------------------------------------

# Inspect the Database 
# - Goal: Predictive Lead Scoring

# Connect to the database
sql_engine = sql.create_engine("sqlite:///database/leads_scored.db")
conn = sql_engine.connect()

# Table Names
metadata = sql.MetaData()
metadata.reflect(bind=sql_engine)
list(metadata.tables.keys())

# Goal: Predictive Lead Scoring (Check whats in the database)
pd.read_sql_table('leads_scored_h2o', conn).glimpse()


# 3.0 Prepares Data for Machine Learning  ---------------------------------------------------------------------------------
# PREPARE THE DATA FOR LEAD SCORING

# Read the tables
leads_df = pd.read_sql_table('leads', conn)
products_df = pd.read_sql_table('products', conn)
transactions_df = pd.read_sql_table('transactions', conn)

# Drop unnecessary columns
df = leads_df.drop(columns=['mailchimp_id', 'made_purchase', 'user_full_name'])

# Create the target from buyer's email addresses
target = transactions_df['user_email'].unique()
df['purchased'] = df['user_email'].isin(target).astype(int)

# 4.0 Sets up H2O AutoML ---------------------------------------------------------------------------------
# SET UP H2O AUTOML

# Initialize the H2O cluster
h2o.init()

# Convert the pandas DataFrame to an H2O Frame
hf = h2o.H2OFrame(df)

hf['purchased'] = hf['purchased'].asfactor()

hf.describe()

# 5.0 Defines Predictors and Targets  ---------------------------------------------------------------------------------

# Set the predictor names and the response column name
predictors = [
 'member_rating',
 'optin_time',
 'country_code',
 'optin_days',
 'email_provider'
]
response = "purchased"

# 6.0 Train the H2O AutoML Model ---------------------------------------------------------------------------------
# Train an H2O AutoML model
# - Note: Set max_runtime_secs = 60 * 10 to run for 10 minutes in production

automl = H2OAutoML(
    max_models=20, 
    seed=1, 
    max_runtime_secs=100 # 60 * 10
)

automl.train(x=predictors, y=response, training_frame=hf)

# View the AutoML Leaderboard
lb = automl.leaderboard
print(lb)

# 7.0 Save the Best Model and Make Predictions ---------------------------------------------------------------------------------
# Save the best model
h2o.save_model(
    automl.leader, 
    path = "models", 
    filename = 'best_model_h2o_' + str(automl.leader.model_id),
    force = True
)

# Load the production model
best_model_h2o = h2o.load_model("models/best_model_h2o_XGBoost_3_AutoML_1_20250623_121353")

# 8.0 Make Predictions and Explain the Model ---------------------------------------------------------------------------------
# Explain
best_model_h2o.explain(hf)


# 9.0 Make Predictions and Update the SQL Database ---------------------------------------------------------------------------------
# Update the SQL Database with the predictions
predictions_df = best_model_h2o.predict(hf).as_data_frame()

# This will create a new table called leads_scored_h2o
pd.concat([leads_df, predictions_df], axis=1) \
    .to_sql('leads_scored_h2o', con = conn, if_exists='replace', index=False) 


# 10.0 Summary and clean up ---------------------------------------------------------------------------------
# Clean up: Close the connection to the database and shutdown H2O
conn.close()
h2o.shutdown(prompt=False)


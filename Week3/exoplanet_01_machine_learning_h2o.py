import pandas as pd
import pytimetk as tk
import sqlalchemy as sql
import h2o
from h2o.automl import H2OAutoML
import os
import pprint 
from cleaning_agent.agent import make_data_cleaning_agent
from langchain_openai import ChatOpenAI

# 1.0 Check working directory
print("Current working directory:", os.getcwd())

# 2.0 Connect to the DB After Data Cleaning ---------------------------------------------------------------------------------
exo_df = pd.read_csv("dataset/nasa_exoplanets.csv")

# 2.1 Clean the Exoplanets Dataset 
# - instantiate your LLM
llm = ChatOpenAI(model='gpt-4.1')
data_cleaning_agent = make_data_cleaning_agent(llm)

cleaned_response = data_cleaning_agent.invoke({
    "user_instructions": None,
    "data_raw": exo_df.to_dict(),
    "max_retries": 3,
    "retry_count": 0
})

# 2.2 Convert the cleaned response back to a DataFrame
cleaned_data = cleaned_response['data_cleaned']
exo_df = pd.DataFrame.from_dict(cleaned_data)

sql_engine = sql.create_engine("sqlite:///database/exoplanets_v4.db") 
exo_df.to_sql('exoplanets', con=sql_engine, if_exists='replace', index=False)

conn = sql_engine.connect()

# Table names 
metadata = sql.MetaData()
metadata.reflect(bind=sql_engine)
print("Tables:", list(metadata.tables.keys()))

# Goal: Predictive Habitability Scoring (Check what's in the database)
print(pd.read_sql_table('exoplanets', conn).head())

# 3.0 Prepare Data for Machine Learning  ---------------------------------------------------------------------------------
# df = pd.read_sql_table('exoplanets', conn)
# df = df.drop(columns=['mass_wrt', 'radius_wrt', 'detection_method'])

drop_cols = [
    # Identifiers & References
    'hostname', 'pl_refname', 'st_refname', 'sy_refname', 'pl_bmassprov',
    # Discovery/Publication Info
    'discoverymethod', 'disc_year', 'disc_facility', 'rowupdate', 'pl_pubdate', 'releasedate',
    # Flags & Metadata
    'default_flag', 'pl_controv_flag', 'ttv_flag',
    'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 'pl_radjlim',
    'pl_bmasselim', 'pl_bmassjlim', 'pl_orbeccenlim', 'pl_insollim', 'pl_eqtlim',
    # Coordinates
    'rastr', 'ra', 'decstr', 'dec'
]
# Drop error/limit columns programmatically
drop_cols += [col for col in exo_df.columns if col.endswith(('err1', 'err2', 'lim'))]

exo_df = exo_df.drop(columns=[col for col in drop_cols if col in exo_df.columns])

# # Create the target column
# Example: create a 'habitable' target column

exo_df['habitable'] = (
    (exo_df['pl_rade'] > 0.5) & (exo_df['pl_rade'] < 2.5) &    # Earth-like size
    (exo_df['pl_orbper'] > 50) & (exo_df['pl_orbper'] < 500)   # Reasonable orbital period
).astype(int)
print(exo_df['habitable'].value_counts())
# df['habitable'] = (
#     (df['planet_type'] == 'Terrestrial') &  # add this line
#     (df['orbital_radius'] > 0.5) & (df['orbital_radius'] < 2.0) &
#     (df['mass_multiplier'] > 0.1) & (df['mass_multiplier'] < 10)
# ).astype(int)

# print("Habitable value counts:\n", df['habitable'].value_counts())

# 4.0 Set up H2O AutoML ---------------------------------------------------------------------------------
h2o.init() 

hf = h2o.H2OFrame(exo_df)
hf['habitable'] = hf['habitable'].asfactor()
hf.describe()

predictors = [
    'sy_snum',      # Number of stars in system
    'sy_pnum',      # Number of planets in system
    'soltype',      # Stellar type (categorical)
    'pl_orbper',    # Orbital period (days)
    'pl_rade',      # Planet radius (Earth radii)
    'pl_radj',      # Planet radius (Jupiter radii)
    'st_teff',      # Stellar effective temperature
    'st_rad',       # Stellar radius
    'st_mass',      # Stellar mass
    'st_met',       # Stellar metallicity
    'st_metratio',  # Stellar metallicity ratio
    'st_logg',      # Stellar surface gravity
    'sy_dist',      # Distance to system
    'sy_vmag',      # Visual magnitude
    'sy_kmag',      # K-band magnitude
    'sy_gaiamag'    # Gaia magnitude
]

response = "habitable"

# 5.0 Train the H2O AutoML Model ---------------------------------------------------------------------------------
automl = H2OAutoML(
    max_runtime_secs=120,
    max_models=10,
    seed=42,
    balance_classes=True
)
automl.train(
    x=predictors,
    y=response,
    training_frame=hf
)

lb = automl.leaderboard
print(lb.head())

# 6.0 Save the Best Model and Make Predictions ---------------------------------------------------
h2o.save_model(
    automl.leader,
    path="models",
    filename="exoplanet_habitable_model_" + str(automl.leader.model_id),
    force=True
)

best_model_h2o = h2o.load_model("models/exoplanet_habitable_model_StackedEnsemble_AllModels_1_AutoML_8_20250624_133123")

# 7.0 Explain the Model ---------------------------------------------------------------------------------
hf_explain = hf.drop('pl_name')
best_model_h2o.explain(hf)
# 8.0 Make Predictions and Update the SQL Database ---------------------------------------------------------------------------------
predictions_df = best_model_h2o.predict(hf).as_data_frame()

# Combine original features and predictions
full_pred_df = pd.concat([exo_df, predictions_df], axis=1)
full_pred_df.to_sql('exoplanets_predictions', con=conn, if_exists='replace', index=False)

# 9.0 Query: List all planets with habitable == 1 ----------------------------------------------------------
# Read back from SQL to ensure correct types
pred_df = pd.read_sql_table('exoplanets_predictions', conn)

# Check type and unique values for habitable
print("Unique values in 'habitable':", pred_df['habitable'].unique())
print("Type of 'habitable':", pred_df['habitable'].dtype)

# If needed, convert habitable to int
if pred_df['habitable'].dtype != int:
    pred_df['habitable'] = pred_df['habitable'].astype(int)

habitable_planets = pred_df[pred_df['predict'] == 1]
print(habitable_planets)
print("Number of habitable planets:", len(habitable_planets))

conn.close()
h2o.shutdown(prompt=False)
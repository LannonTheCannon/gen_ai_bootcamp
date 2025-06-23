import pandas as pd
import pytimetk as tk
import sqlalchemy as sql
import h2o
from h2o.automl import H2OAutoML
import os
import pprint 


# 1.0 Check working directory
print("Current working directory:", os.getcwd())

# 2.0 Connect to the Database ---------------------------------------------------------------------------------
exo_df = pd.read_csv("dataset/cleaned_5250.csv")
sql_engine = sql.create_engine("sqlite:///database/exoplanets.db") 
exo_df.to_sql('exoplanets', con=sql_engine, if_exists='replace', index=False)

conn = sql_engine.connect()

# Table names 
metadata = sql.MetaData()
metadata.reflect(bind=sql_engine)
print("Tables:", list(metadata.tables.keys()))

# Goal: Predictive Habitability Scoring (Check what's in the database)
print(pd.read_sql_table('exoplanets', conn).head())

# 3.0 Prepare Data for Machine Learning  ---------------------------------------------------------------------------------
df = pd.read_sql_table('exoplanets', conn)
df = df.drop(columns=['name', 'mass_wrt', 'radius_wrt', 'detection_method'])

# Create the target column
df['habitable'] = (
    (df['planet_type'] == 'Terrestrial') &  # add this line
    (df['orbital_radius'] > 0.5) & (df['orbital_radius'] < 2.0) &
    (df['mass_multiplier'] > 0.1) & (df['mass_multiplier'] < 10)
).astype(int)

print("Habitable value counts:\n", df['habitable'].value_counts())

# 4.0 Set up H2O AutoML ---------------------------------------------------------------------------------
h2o.init() 

hf = h2o.H2OFrame(df)
hf['habitable'] = hf['habitable'].asfactor()
hf.describe()

predictors = [
    "distance",
    "stellar_magnitude",
    # "planet_type",  # removed to avoid data leakage
    "discovery_year",
    "mass_multiplier",
    "orbital_period",
    "orbital_radius",
    "eccentricity",
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
    filename="exoplanet_habitable_model" + str(automl.leader.model_id),
    force=True
)

best_model_h2o = h2o.load_model("models/exoplanet_habitable_modelXGBoost_1_AutoML_2_20250623_155756")

# 7.0 Explain the Model ---------------------------------------------------------------------------------
best_model_h2o.explain(hf)

# 8.0 Make Predictions and Update the SQL Database ---------------------------------------------------------------------------------
predictions_df = best_model_h2o.predict(hf).as_data_frame()

# Combine original features and predictions
full_pred_df = pd.concat([df, predictions_df], axis=1)
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

habitable_planets = pred_df[pred_df['habitable'] == 1]
print(habitable_planets)
print("Number of habitable planets:", len(habitable_planets))

conn.close()
h2o.shutdown(prompt=False)
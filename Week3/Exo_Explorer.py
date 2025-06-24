import streamlit as st 
import pandas as pd
import pytimetk as tk
import sqlalchemy as sql
import h2o
from h2o.automl import H2OAutoML
from cleaning_agent.agent import make_data_cleaning_agent
from langchain_openai import ChatOpenAI
import os

# === Streamlit Setup ===
st.set_page_config(page_title="Exo_Explorer", layout="wide")
st.title("Explore NASA Exoplanets Dataset")
st.markdown("Use ML to **predict habitability scores** of all the known exoplanets in the dataset!")

# === 1.0 Load Raw Data ===
print("Current working directory:", os.getcwd())
exo_df = pd.read_csv("dataset/nasa_exoplanets.csv") 

st.subheader('Raw Exoplanets Data')
st.write(exo_df.head())

st.subheader('Raw Exoplanets Data Description')
st.write(exo_df.describe())

# === 2.0 Session State Initialization ===
st.session_state.setdefault('exo_df_cleaned', None)
st.session_state.setdefault('cleaning_code', None)
st.session_state.setdefault('sql_engine', None)
st.session_state.setdefault('conn', None)

# === 3.0 Data Cleaning ===
st.subheader('1. Begin Data Cleaning')
if st.button('Clean the Data'):
    st.write("Cleaning the data...")
    llm = ChatOpenAI(model='gpt-4.1')
    data_cleaning_agent = make_data_cleaning_agent(llm)
    cleaned_response = data_cleaning_agent.invoke({
        "user_instructions": None,
        "data_raw": exo_df.to_dict(),
        "max_retries": 3,
        "retry_count": 0
    })
    st.session_state.exo_df_cleaned = pd.DataFrame.from_dict(cleaned_response['data_cleaned'])
    st.session_state.cleaning_code = cleaned_response.get("data_cleaner_function", "No code returned.")
    st.success("Data cleaned successfully!")

if st.session_state.exo_df_cleaned is not None:
    tab1, tab2 = st.tabs(["Cleaned Data Preview", "Cleaning Code"])
    with tab1:
        st.subheader('Cleaned Exoplanets Data')
        st.write(st.session_state.exo_df_cleaned.head())
    with tab2:
        st.subheader('Generated Cleaning Code')
        st.code(st.session_state.cleaning_code, language="python")

# === 4.0 Save Cleaned Data to SQL ===
st.subheader('2. Save Cleaned Data to SQL Database')
if st.button('Save Cleaned Data to SQL Database'):
    engine = sql.create_engine("sqlite:///database/exoplanets_v4.db") 
    conn = engine.connect()
    st.session_state.exo_df_cleaned.to_sql('exoplanets', con=engine, if_exists='replace', index=False)
    st.session_state.sql_engine = engine
    st.session_state.conn = conn
    st.success("Cleaned data saved to SQL database!")

    metadata = sql.MetaData()
    metadata.reflect(bind=engine)
    st.write("Tables in the database:", list(metadata.tables.keys()))
    st.subheader('Cleaned Data from SQL')
    st.write(pd.read_sql_table('exoplanets', conn).head())

# === 5.0 Prepare Data for ML ===
st.subheader('3. Prepare Data for Machine Learning')
if st.button('Prepare Data for ML'):
    conn = st.session_state.get('conn')
    if conn is None:
        st.error("Please save the cleaned data to the SQL database first.")
        st.stop()

    drop_cols = [
        'hostname', 'pl_refname', 'st_refname', 'sy_refname', 'pl_bmassprov',
        'discoverymethod', 'disc_year', 'disc_facility', 'rowupdate', 'pl_pubdate', 'releasedate',
        'default_flag', 'pl_controv_flag', 'ttv_flag',
        'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 'pl_radjlim',
        'pl_bmasselim', 'pl_bmassjlim', 'pl_orbeccenlim', 'pl_insollim', 'pl_eqtlim',
        'rastr', 'ra', 'decstr', 'dec'
    ]
    df = pd.read_sql_table('exoplanets', conn)
    exo_df_cleaned = df.drop(columns=[col for col in drop_cols if col in df.columns])
    st.session_state.exo_df_cleaned = exo_df_cleaned
    st.write("Data prepared for ML:")
    st.write(exo_df_cleaned.head())

# === 6.0 Initialize H2O AutoML ===
st.subheader('4. Set up H2O AutoML')
if st.button('Initialize H2O AutoML'):
    h2o.init()
    hf = h2o.H2OFrame(st.session_state.exo_df_cleaned)
    hf['habitable'] = hf['habitable'].asfactor()
    st.session_state.hf = hf

    st.write("H2O Frame initialized.")
    st.write(hf.describe())

    possible_predictors = [
        'pl_rade', 'pl_orbper', 'pl_bmassj', 'pl_eqt', 'st_mass', 'st_teff',
        'pl_orbsmax', 'pl_bmasse', 'st_rad', 'st_met', 'st_logg', 'sy_dist',
        'sy_vmag', 'sy_kmag', 'sy_gaiamag', 'sy_snum', 'sy_pnum', 'soltype'
    ]
    predictors = [col for col in possible_predictors if col in st.session_state.exo_df_cleaned.columns]
    response = "habitable"
    if response in predictors:
        predictors.remove(response)

    st.session_state.predictors = predictors
    st.session_state.response = response

    st.write("Predictors:", predictors)
    st.write("Response:", response)

# === 7.0 Train H2O Model ===
st.subheader('5. Train the H2O AutoML Model')
if st.button('Train H2O AutoML Model'):
    automl = H2OAutoML(max_models=20, seed=1, max_runtime_secs=100)
    automl.train(
        x=st.session_state.predictors, 
        y=st.session_state.response, 
        training_frame=st.session_state.hf
    )
    st.session_state.automl = automl
    best_model = automl.leader
    st.session_state.best_model = best_model
    h2o.save_model(model=best_model, path="models/best_exoplanet_model", force=True)

    st.write("Model trained!")
    st.subheader('AutoML Leaderboard')
    st.write(automl.leaderboard)
    st.success("Best model saved!")

# === 8.0 Explain the Model ===
st.subheader('6. Explain the Model')
if st.button('Explain the Best Model'):
    best_model = h2o.load_model("models/best_exoplanet_model")
    hf_explain = st.session_state.hf.drop('pl_name') if 'pl_name' in st.session_state.hf.columns else st.session_state.hf
    explanation = best_model.explain(hf_explain)
    st.write("Model explanation:")
    st.write(explanation)

# === 9.0 Make Predictions & Update SQL ===
st.subheader('7. Make Predictions and Update SQL Database')
if st.button('Make Predictions and Update SQL Database'):
    best_model = h2o.load_model("models/best_exoplanet_model")
    predictions_df = best_model.predict(st.session_state.hf).as_data_frame()
    exo_df_cleaned = st.session_state.exo_df_cleaned.copy()
    exo_df_cleaned['predicted_habitability'] = predictions_df['predict']
    exo_df_cleaned.to_sql('exoplanets_with_predictions', con=st.session_state.sql_engine, if_exists='replace', index=False)

    st.success("Predictions saved to SQL!")
    st.subheader('Exoplanets with Predictions')
    st.write(pd.read_sql_table('exoplanets_with_predictions', st.session_state.conn).head())

# === 10.0 Query Habitable Planets ===
st.subheader('8. Query Habitable Planets')
if st.button('List Habitable Planets'):
    query = "SELECT * FROM exoplanets_with_predictions WHERE predicted_habitability = 1"
    results = pd.read_sql_query(query, st.session_state.conn)
    st.write("Habitable Planets Found:", len(results))
    st.write(results)

# === 11.0 Final Cleanup ===
if st.button('Close Connection and Shutdown H2O'):
    if st.session_state.conn:
        st.session_state.conn.close()
    h2o.shutdown(prompt=False)
    st.subheader('9. Summary and Clean Up')
    st.write("This app has demonstrated cleaning, ML modeling, prediction, and explanation for NASA's Exoplanets dataset using H2O AutoML.")
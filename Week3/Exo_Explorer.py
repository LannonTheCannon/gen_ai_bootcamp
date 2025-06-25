import streamlit as st
import pandas as pd
import sqlalchemy as sql
import h2o
from h2o.automl import H2OAutoML
import os
from cleaning_agent.agent import make_data_cleaning_agent
from langchain_openai import ChatOpenAI
from collections import Counter

st.set_page_config(page_title="Exo Explorer", layout="wide")
st.title("Exo Explorer: Predict Exoplanet Habitability")

# === Load and Clean Data ===
exo_df = pd.read_csv("dataset/nasa_exoplanets.csv")

st.subheader("Raw Dataset Preview")
st.write(exo_df.head())

with st.expander("📖 Column Reference Guide (click to expand)"):
    st.markdown("""
    Here’s what each of the dataset’s columns mean:

    **Planet & Host Info**
    - `pl_name`: Planet Name  
    - `hostname`: Host Star Name  
    - `default_flag`: Default Parameter Set  
    - `sy_snum`: Number of Stars in System  
    - `sy_pnum`: Number of Planets in System  

    **Discovery & Classification**
    - `discoverymethod`: Discovery Method  
    - `disc_year`: Discovery Year  
    - `disc_facility`: Discovery Facility  
    - `soltype`: Solution Type  
    - `pl_controv_flag`: Controversial Flag  
    - `pl_refname`: Parameter Reference  

    **Orbital Parameters**
    - `pl_orbper`: Orbital Period [days]  
    - `pl_orbsmax`: Semi-Major Axis (Average Orbit Distance) [AU]  
    - `pl_orbeccen`: Orbital Eccentricity  

    **Planet Physical Characteristics**
    - `pl_rade`: Radius [Earth radii]  
    - `pl_radj`: Radius [Jupiter radii]  
    - `pl_bmasse`: Mass [Earth masses]  
    - `pl_bmassj`: Mass [Jupiter masses]  
    - `pl_eqt`: Equilibrium Temperature [K]  
    - `pl_insol`: Insolation (Stellar Energy Flux) [Earth flux]  

    **Host Star Properties**
    - `st_teff`: Effective Temperature [K]  
    - `st_rad`: Radius [Solar radii]  
    - `st_mass`: Mass [Solar masses]  
    - `st_met`: Metallicity [dex]  
    - `st_logg`: Surface Gravity [log10(cm/s²)]  
    - `st_spectype`: Spectral Type  

    **Position & Distance**
    - `ra`, `dec`: Right Ascension / Declination [deg]  
    - `sy_dist`: Distance from Earth [parsecs]  

    **Magnitudes**
    - `sy_vmag`: Visual Magnitude  
    - `sy_kmag`: Infrared Magnitude (Ks Band)  
    - `sy_gaiamag`: Gaia Magnitude  

    **Timestamps & References**
    - `rowupdate`, `pl_pubdate`, `releasedate`: Update & Publication Dates  
    - `st_refname`, `sy_refname`: Reference Names  

    _(Note: Uncertainty/error/limit columns have been omitted for clarity.)_
    """)

llm = ChatOpenAI(model='gpt-4.1')
data_cleaning_agent = make_data_cleaning_agent(llm)

# if st.button("Run Data Cleaning"):
#     with st.spinner("Cleaning the dataset..."):
#         cleaned_response = data_cleaning_agent.invoke({
#             "user_instructions": None,
#             "data_raw": exo_df.to_dict(),
#             "max_retries": 3,
#             "retry_count": 0
#         })
#         exo_df = pd.DataFrame.from_dict(cleaned_response['data_cleaned'])
#         st.success("Data cleaned successfully!")

# if st.button("Run Data Cleaning"):
#     with st.spinner("Cleaning the dataset..."):
#         cleaned_response = data_cleaning_agent.invoke({
#             "user_instructions":"""
#                 Identify and condense duplicate records based on all columns. 
#                 For numeric fields, compute the average. For categorical fields, use the most frequent value.
#                 If a value remains missing after aggregation, keep it as NaN.
#                 Do not drop duplicates blindly—combine them meaningfully.
#                 Also perform your usual cleaning steps except outlier removal.
#             """,
#             "data_raw":exo_df.to_dict(),
#             "max_retries":3,
#             "retry_count":0
#         })
#         exo_df = pd.DataFrame.from_dict(cleaned_response['data_cleaned'])
#         st.success("Data cleaned successfully!")

        

# === Drop unnecessary columns ===
drop_cols = [
    'hostname', 'pl_refname', 'st_refname', 'sy_refname', 'pl_bmassprov',
    'discoverymethod', 'disc_year', 'disc_facility', 'rowupdate', 'pl_pubdate', 'releasedate',
    'default_flag', 'pl_controv_flag', 'ttv_flag',
    'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 'pl_radjlim',
    'pl_bmasselim', 'pl_bmassjlim', 'pl_orbeccenlim', 'pl_insollim', 'pl_eqtlim',
    'rastr', 'ra', 'decstr', 'dec'
]
drop_cols += [col for col in exo_df.columns if col.endswith(('err1', 'err2', 'lim'))]
exo_df = exo_df.drop(columns=[col for col in drop_cols if col in exo_df.columns])

# === Define habitability ===
def label_habitable(row):
    try:
        return int(
            (row.get('pl_rade', 99) <= 5) and                            # Earth-sized
            (0.38 <= row.get('pl_orbsmax', -1) <= 2.0) and               # Orbital distance in AU
            (row.get('pl_insol', -1) >= 0.25 and row.get('pl_insol') <= 2.0) and  # Stellar flux (Earth flux)
            (3500 <= row.get('st_teff', -1) <= 8000)                     # Star temperature
        )
    except:
        return 0
exo_df['habitable'] = exo_df.apply(label_habitable, axis=1)

# === Check label distribution ===
label_counts = Counter(exo_df['habitable'])
st.write("Label distribution:", label_counts)
st.markdown("The 'habitable' column indicates whether a planet is potentially habitable (1) or not (0).")

if len(label_counts) < 2:
    st.warning("Only one class found in 'habitable'. Forcing one example to be habitable.")
    exo_df.loc[0, 'habitable'] = 1

# === Save to SQL ===

os.makedirs("database", exist_ok=True)

engine = sql.create_engine("sqlite:///database/exoplanets_v4.db")
conn = engine.connect()
exo_df.to_sql('exoplanets', con=engine, if_exists='replace', index=False)

# === Train Model ===
st.subheader("Train H2O AutoML Model")

if st.button("Start Training"):
    h2o.init()
    hf = h2o.H2OFrame(exo_df)
    hf['habitable'] = hf['habitable'].asfactor()

    predictors = [
        col for col in [
            'sy_snum', 'sy_pnum', 'soltype', 'pl_orbper', 'pl_rade', 'pl_radj',
            'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg',
            'sy_dist', 'sy_vmag', 'sy_kmag', 'sy_gaiamag',
            'st_metratio'  # <- only include if it exists
        ] if col in exo_df.columns
    ]
    response = "habitable"

    automl = H2OAutoML(
        max_runtime_secs=60,
        max_models=5,
        seed=42,
        balance_classes=True
    )
    automl.train(x=predictors, y=response, training_frame=hf)
    lb = automl.leaderboard
    st.write("AutoML Leaderboard", lb.as_data_frame())
    if lb.nrows == 0:
        st.error("AutoML did not train any models. Please check your data.")

    best_model = automl.leader
    if best_model is not None and hassattr(best_model, 'model_id'):
        model_path = h2o.save_model(best_model, path="models", force=True)
        st.write(f"Best model saved to: {model_path}")
    else: 
        st.error("No valid model found. Please check the training process.")
    
    predictions_df = best_model.predict(hf).as_data_frame()

    # 'predict' is the class (0 or 1), 'p1' is the probability of class 1 (habitable)
    exo_df['predicted_habitability'] = predictions_df['predict']
    exo_df['probability_habitable'] = predictions_df['p1']  # p1 = probability of class 1
    exo_df.to_sql('exoplanets_predictions', con=conn, if_exists='replace', index=False)
    st.success("Predictions saved to database.")

    # habitable_planets = exo_df[exo_df['predicted_habitability'] == 1]
    # st.subheader("Predicted Habitable Planets")
    # st.write(habitable_planets)
    # st.write(f"Total predicted habitable planets: {len(habitable_planets)}")

    st.subheader("🌍 Planets with Highest Predicted Habitability")
    high_conf_planets = exo_df[exo_df['probability_habitable'] > 0.90]

    st.subheader("🌟 Planets with >90% Probability of Being Habitable")
    st.write(
        high_conf_planets
        .sort_values(by='probability_habitable', ascending=False)
        .style.format({'probability_habitable': '{:.2%}'})
    )
    st.write(f"Total planets with >90% habitability probability: {len(high_conf_planets)}")
    # Show cases where model disagrees with label
    false_positives = exo_df[
        (exo_df['habitable'] == 0) & 
        (exo_df['predicted_habitability'] == 1)
    ]

    st.subheader("🔍 Model-Discovered Habitable Planets (False Positives)")
    st.write(
    false_positives
    .sort_values(by='probability_habitable', ascending=False)
    .style.format({'probability_habitable': '{:.2%}'})
    )
    st.write(f"Total predicted habitable planets with rule-based label = 0: {len(false_positives)}")

    st.download_button(
    label="📥 Download Predictions CSV",
    data=exo_df.to_csv(index=False).encode('utf-8'),
    file_name='exo_predictions.csv',
    mime='text/csv'
    )

    h2o.shutdown(prompt=False)
    conn.close()

import streamlit as st
import plotly.express as px
import pandas as pd

# Main app interface
st.title("Your Marketing AI Copilot")

value = st.number_input("Enter a number", min_value=0, max_value=100, value=50)

submit_buttton_2 = st.button("Submit Number")
if submit_buttton_2:
    st.write("You entered:", value)

# Layout for input and output
col1, col2 = st.columns(2)

with col1:
    st.header("How can I help you?")
    question = st.text_area("Enter your Marketing question here:", height=150)
    submit_button = st.button("Submit")
    
    if submit_button:
        st.write("You asked:", question)

with col2:
    st.header("AI Generated Answer")
    
    # Sample data for the Plotly graph
    data = {
        "Category": ["A", "B", "C", "D"],
        "Value": [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)
    
    # Create a bar chart using Plotly Express
    fig = px.bar(df, x="Category", y="Value", title="Sample Plotly Bar Chart")
    
    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)
    
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

# Page config
st.set_page_config(page_title="Streamlit Starter App", layout="wide")

# Title and markdown
st.title("🚀 Streamlit Starter App")
st.markdown("""
This app showcases the most popular Streamlit features.  
Use this as your base template for building apps!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Upload", "Charts", "Custom Logic"])

# Home page
if page == "Home":
    st.header("💬 Inputs & Widgets")

    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=0, max_value=120)
    color = st.selectbox("Pick your favorite color", ["Red", "Blue", "Green"])
    agree = st.checkbox("I agree to the terms")

    if st.button("Submit"):
        if agree:
            st.success(f"Welcome, {name}! You're {age} and love {color}.")
        else:
            st.warning("You must agree to the terms.")

    st.header("📸 Image Display")
    st.image("https://picsum.photos/600/300", caption="Random Image", use_column_width=True)

# Data upload page
elif page == "Data Upload":
    st.header("📁 Upload a CSV")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("🧾 Data Preview", df.head())

        if st.checkbox("Show Summary Stats"):
            st.write(df.describe())

        st.download_button("📥 Download CSV", uploaded_file, file_name="your_file.csv")

# Charts page
elif page == "Charts":
    st.header("📊 Charts in Streamlit")

    df = pd.DataFrame({
        "x": np.arange(10),
        "y": np.random.randn(10).cumsum(),
        "z": np.random.randn(10).cumsum()
    })

    chart_type = st.radio("Select Chart Type", ["Line Chart", "Bar Chart", "Matplotlib", "Plotly"])

    if chart_type == "Line Chart":
        st.line_chart(df.set_index("x"))
    elif chart_type == "Bar Chart":
        st.bar_chart(df.set_index("x"))
    elif chart_type == "Matplotlib":
        fig, ax = plt.subplots()
        ax.plot(df["x"], df["y"], label="y")
        ax.plot(df["x"], df["z"], label="z")
        ax.legend()
        st.pyplot(fig)
    elif chart_type == "Plotly":
        fig = px.line(df, x="x", y=["y", "z"], title="Plotly Line Chart")
        st.plotly_chart(fig)

# Custom logic page
elif page == "Custom Logic":
    st.header("🧠 Custom Python Execution")

    st.markdown("Try running a simple calculation and see the result.")

    num1 = st.number_input("First number", value=10)
    num2 = st.number_input("Second number", value=5)
    operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide"])

    if st.button("Calculate"):
        result = None
        if operation == "Add":
            result = num1 + num2
        elif operation == "Subtract":
            result = num1 - num2
        elif operation == "Multiply":
            result = num1 * num2
        elif operation == "Divide":
            result = num1 / num2 if num2 != 0 else "Error: Division by zero"

        st.success(f"Result: {result}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")

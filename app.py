
%%writefile app.py
import os
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Used Car Valuation Engine",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Valuation Engine")
st.markdown("Estimate the fair market value of a pre-owned vehicle using machine learning.")
st.write("---")

# Load Pipeline
MODEL_PATH = "used_car_xgboost_pipeline.joblib"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file '{path}' not found! Make sure your .joblib file is in /content.")
        return None
    return joblib.load(path)

pipeline = load_model(MODEL_PATH)

# User Controls & Predictions
if pipeline is not None:
    st.sidebar.header("Vehicle Specifications")

    car_age = st.sidebar.slider("Vehicle Age (Years)", 0, 20, 4)
    mileage = st.sidebar.slider("Total Mileage (Miles)", 0, 200000, 35000, 1000)
    engine_size = st.sidebar.select_slider("Engine Size (Liters)", options=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0], value=2.0)
    brand = st.sidebar.selectbox("Manufacturer Brand", ["Toyota", "Ford", "BMW", "Honda", "Mercedes"])
    transmission = st.sidebar.radio("Transmission Type", ["Automatic", "Manual"], horizontal=True)

    st.subheader("Selected Specifications")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Brand:** {brand}")
        st.write(f"**Age:** {car_age} years old")
        st.write(f"**Engine:** {engine_size}L")
    with col2:
        st.write(f"**Transmission:** {transmission}")
        st.write(f"**Mileage:** {mileage:,} miles")

    st.write("---")

    if st.button("Calculate Market Value", type="primary"):
        input_data = pd.DataFrame([{
            "car_age": car_age,
            "mileage": mileage,
            "engine_size": engine_size,
            "brand": brand,
            "transmission": transmission
        }])

        predicted_price = pipeline.predict(input_data)[0]

        st.subheader("Estimated Market Valuation")
        st.metric(label="Fair Market Price (USD)", value=f"${predicted_price:,.2f}")
        st.success("Valuation calculated successfully!")
     

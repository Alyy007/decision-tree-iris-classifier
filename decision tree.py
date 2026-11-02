import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🌸",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open("decision_tree_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title
st.title("🌸 Iris Flower Classification")
st.write(
    "This application predicts the species of an Iris flower "
    "using a trained Decision Tree model."
)

st.divider()

# User inputs
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

# Prediction button
if st.button("Predict Flower Species", type="primary"):

    # Prepare input data
    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Species names
    species_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    predicted_species = species_names[int(prediction)]

    # Display result
    st.success(
        f"Predicted Flower Species: {predicted_species}"
    )

    # Display probabilities if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        st.subheader("Prediction Confidence")

        for i, species in enumerate(species_names):
            st.write(
                f"{species}: {probabilities[i] * 100:.2f}%"
            )

        st.progress(float(probabilities[int(prediction)]))

st.divider()

st.caption(
    "Machine Learning Project | Decision Tree Classifier"
)

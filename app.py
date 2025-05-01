import streamlit as st
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO

# Load model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Configuration
st.set_page_config(page_title="IRIS CLASSIFIER", layout="centered")
st.title("🌿 IRIS FLOWER CLASSIFIER")
st.markdown("""
*Predicts iris species using Random Forest model*  
**Key features:**  
- Petal Width is the most important feature  
- Sepal measurements help with borderline cases  
""")

# Input Section
with st.expander("📏 ENTER FLOWER MEASUREMENTS", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0, 0.1)

# Prediction Logic
if st.button("🔍 PREDICT SPECIES"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(input_data)[0]
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'][pred]
    
    # Display Results
    st.success(f"## PREDICTION: {species.split('-')[1].upper()}")
    
    # Decision Rules
    st.markdown("### How the model decided:")
    if petal_width <= 0.8:
        st.markdown("- 🌸 **Petal Width ≤ 0.8cm** → Strong Setosa indicator")
    elif petal_width <= 1.7:
        st.markdown("- 🌺 **Petal Width 0.8-1.7cm** → Likely Versicolor")
    else:
        st.markdown("- 🌹 **Petal Width > 1.7cm** → Virginica territory")
        if petal_length > 4.8:
            st.markdown("- 📏 **Long petals (>4.8cm)** confirms Virginica")
    
    # Show flower image
    try:
        img_url = {
            'Iris-setosa': 'https://upload.wikimedia.org/wikipedia/commons/1/11/Iris_setosa_3.jpg',
            'Iris-versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
            'Iris-virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
        }[species]
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, width=300)
    except:
        st.warning("Couldn't load flower image")

# Model Information
with st.expander("ℹ️ ABOUT THE MODEL"):
    st.markdown("""
    **Model Details:**
    - Type: Random Forest Classifier
    - Accuracy: ~96% on test data
    - Key Features: Petal measurements most significant
    
    **Common Confusions:**
    - Large Versicolor vs small Virginica flowers
    - Borderline cases around 1.7cm petal width
    """)

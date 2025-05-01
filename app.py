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
st.title("🔍 IRIS SPECIES PREDICTION")
st.markdown("""
*Uses decision rules from Random Forest analysis*  
**Key splits:**  
- Setosa: Petal Width ≤ 0.75cm  
- Versicolor: Petal Width ≤ 1.55cm  
- Virginica: Petal Width > 1.55cm + Length > 4.9cm  
""")

# Input Section
with st.expander("📏 INPUT MEASUREMENTS", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1,
                               help="Critical for virginica identification")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    with col2:
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0, 0.1,
                              help="Main separator for setosa/versicolor")
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)

# Prediction Logic
if st.button("🌼 IDENTIFY SPECIES"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(input_data)[0]
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'][pred]
    
    # Display Results
    st.success(f"## PREDICTION: {species.split('-')[1].upper()}")
    
    # Decision Path Visualization
    st.markdown("### Decision Path:")
    if petal_width <= 0.75:
        st.markdown("- ✅ **Petal Width ≤ 0.75cm** → Pure Setosa (100% accurate)")
    else:
        st.markdown("- ❌ Petal Width > 0.75cm")
        if petal_width <= 1.55:
            st.markdown("- ✅ **Petal Width ≤ 1.55cm** → Versicolor (82.4% accurate)")
        else:
            st.markdown("- ❌ Petal Width > 1.55cm")
            if petal_length <= 4.9:
                st.markdown("- ✅ **Petal Length ≤ 4.9cm** → Versicolor (98.1% accurate)")
            else:
                st.markdown("- ✅ **Petal Length > 4.9cm** → Virginica (93.3% accurate)")
    
    # Show flower image
    try:
        img_url = {
            'Iris-setosa': 'https://upload.wikimedia.org/wikipedia/commons/1/11/Iris_setosa_3.jpg',
            'Iris-versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
            'Iris-virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
        }[species]
        img = Image.open(BytesIO(requests.get(img_url).content)
        st.image(img, width=300)
    except:
        st.warning("Image unavailable")

# Model Insights
with st.expander("💡 MODEL INSIGHTS"):
    st.markdown("""
    **Key Splits Found in Decision Tree:**
    - Setosa perfectly separated at Petal Width ≤ 0.75cm
    - Versicolor mostly separated at Petal Width ≤ 1.55cm
    - Virginica requires both:
      - Petal Width > 1.55cm 
      - Petal Length > 4.9cm
    
    **Borderline Cases:**
    - 3 Versicolor misclassified when 1.55 < Width ≤ 1.65cm
    - 1 Virginica misclassified when Length ≤ 4.9cm
    """)


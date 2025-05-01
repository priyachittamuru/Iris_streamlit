import streamlit as st
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO

# ---- Configuration ----
st.set_page_config(
    page_title="IRIS CLASSIFIER",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4a7c59;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3a5a40;
        transform: scale(1.05);
    }
    .stSlider>div>div>div>div {
        background-color: #4a7c59 !important;
    }
    .stExpander>div>div>div {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9f5db 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .feature-importance {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Load Model ----
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        return pickle.load(f)
model = load_model()

# ---- Header Section ----
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/1200px-Kosaciec_szczecinkowaty_Iris_setosa.jpg", 
             width=120, caption="Iris Flower")
with col2:
    st.title("Iris Flower Classifier")
    st.markdown("""
    <div style='color: #555; font-size: 16px;'>
    Predict iris species using our advanced Random Forest model with 96% accuracy
    </div>
    """, unsafe_allow_html=True)

# ---- Input Section ----
with st.expander("📏 Enter Flower Measurements", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1,
                                help="Length of the sepal from base to tip")
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1,
                               help="Width of the sepal at its widest point")
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1,
                                help="Length of the petal from base to tip")
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0, 0.1,
                               help="Width of the petal at its widest point")

# ---- Prediction Section ----
if st.button("🔍 Predict Species", use_container_width=True):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'][pred]
    
    # Results Card
    with st.container():
        st.markdown(f"""
        <div class='prediction-card'>
            <h2 style='color: #4a7c59; text-align: center;'>Prediction Result</h2>
            <div style='text-align: center; margin: 20px 0;'>
                <h3 style='color: #2d6a4f;'>{species.split('-')[1].upper()}</h3>
                <div style='height: 10px; background: #e9ecef; border-radius: 5px; margin: 15px 0;'>
                    <div style='width: {proba[pred]*100:.1f}%; height: 100%; background: #4a7c59; border-radius: 5px;'></div>
                </div>
                <p>Confidence: {proba[pred]*100:.1f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Visual Explanation
        with st.expander("📊 How the model decided"):
            tab1, tab2 = st.tabs(["Decision Rules", "Feature Importance"])
            
            with tab1:
                st.markdown("### Key Decision Factors")
                if petal_width <= 0.8:
                    st.success("🌸 **Petal Width ≤ 0.8cm** → Strong Setosa indicator")
                elif petal_width <= 1.7:
                    st.warning("🌺 **Petal Width 0.8-1.7cm** → Likely Versicolor")
                else:
                    st.error("🌹 **Petal Width > 1.7cm** → Virginica territory")
                    if petal_length > 4.8:
                        st.info("📏 **Long petals (>4.8cm)** confirms Virginica")
            
            with tab2:
                st.markdown("### Relative Feature Importance")
                features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
                importance = model.feature_importances_
                for feat, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
                    st.markdown(f"""
                    <div class='feature-importance'>
                        <strong>{feat}</strong>
                        <div style='height: 10px; background: #e9ecef; border-radius: 5px; margin-top: 5px;'>
                            <div style='width: {imp*100:.1f}%; height: 100%; background: #4a7c59; border-radius: 5px;'></div>
                        </div>
                        <small>{imp*100:.1f}% importance</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Flower Image
        try:
            img_url = {
                'Iris-setosa': 'https://upload.wikimedia.org/wikipedia/commons/1/11/Iris_setosa_3.jpg',
                'Iris-versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
                'Iris-virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
            }[species]
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(img, caption=f"{species.split('-')[1]} Sample", width=300)
                st.markdown(f"[Source Image]({img_url})")
        except:
            st.warning("Couldn't load flower image")

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This app predicts iris flower species using measurements:
    - Sepal length/width
    - Petal length/width
    
    **Model Details:**
    - Random Forest Classifier
    - 96% accuracy
    - Trained on classic Iris dataset
    """)
    
    st.markdown("---")
    st.markdown("### Quick Guide")
    st.markdown("""
    1. Adjust the sliders
    2. Click predict
    3. See explanation
    """)
    
    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit")

# ---- Footer ----
st.markdown("---")
st.markdown("""
<small>Note: This is a demonstration app. For scientific use, please verify with additional data.</small>
""", unsafe_allow_html=True)

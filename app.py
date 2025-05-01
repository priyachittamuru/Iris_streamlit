import streamlit as st
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO

# Load the pre-trained model
with open('random_forest_model_iris.pkl', 'rb') as f:
    model = pickle.load(f)

# Iris species information with botanical descriptions
species_data = {
    'Iris-setosa': {
        'image': 'https://upload.wikimedia.org/wikipedia/commons/1/11/Iris_setosa_3.jpg',
        'description': 'Distinctive features:\n'
                     '- Small petals (typically 1-2 cm)\n'
                     '- Short, broad sepals (3-5 cm)\n'
                     '- Compact flower structure\n'
                     '- Bright white/blue coloration'
    },
    'Iris-versicolor': {
        'image': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
        'description': 'Distinctive features:\n'
                     '- Medium-sized petals (3-4 cm)\n'
                     '- Elongated sepals (5-7 cm)\n'
                     '- Purple/blue veined patterns\n'
                     '- Graceful, spreading form'
    },
    'Iris-virginica': {
        'image': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg',
        'description': 'Distinctive features:\n'
                     '- Large petals (4-7 cm)\n'
                     '- Broad, drooping sepals (6-8 cm)\n'
                     '- Pale purple coloration\n'
                     '- Tall, robust stems'
    }
}

# Streamlit UI
st.set_page_config(page_title="Iris Botanical Classifier", page_icon="🌿")

st.title("🌿 Iris Botanical Classifier")
st.markdown("""
Predict Iris species based on precise morphological measurements.
This tool helps botanists and gardeners identify iris varieties.
""")

# Sidebar for measurements
with st.sidebar:
    st.header("✏️ Measurement Input")
    st.caption("Use precise caliper measurements for best results")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
        petal_length = st.number_input("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
    with col2:
        sepal_width = st.number_input("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
        petal_width = st.number_input("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)
    
    if st.button("Identify Species", type="primary"):
        st.session_state['predict_clicked'] = True

# Main content
if 'predict_clicked' in st.session_state:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = list(species_data.keys())[prediction]
    
    # Results display
    st.success(f"## Identification: {species.split('-')[1]}")
    
    col_img, col_desc = st.columns([1, 2])
    with col_img:
        try:
            response = requests.get(species_data[species]['image'])
            img = Image.open(BytesIO(response.content))
            st.image(img, width=250)
        except:
            st.warning("Image unavailable")
    
    with col_desc:
        st.markdown(f"**Botanical Characteristics:**")
        st.markdown(species_data[species]['description'])
        
        # Measurement comparison
        st.markdown(f"**Your Measurements:**")
        st.markdown(f"- Sepals: {sepal_length} × {sepal_width} cm")
        st.markdown(f"- Petals: {petal_length} × {petal_width} cm")

# Species reference guide
st.markdown("---")
st.subheader("📚 Iris Species Reference")

tab1, tab2, tab3 = st.tabs(["Setosa", "Versicolor", "Virginica"])
with tab1:
    st.markdown(species_data['Iris-setosa']['description'])
    st.image(species_data['Iris-setosa']['image'], width=300)
with tab2:
    st.markdown(species_data['Iris-versicolor']['description'])
    st.image(species_data['Iris-versicolor']['image'], width=300)
with tab3:
    st.markdown(species_data['Iris-virginica']['description'])
    st.image(species_data['Iris-virginica']['image'], width=300)

# Footer
st.markdown("""
<style>
.footer {
    font-size: small;
    color: gray;
    text-align: center;
    margin-top: 2rem;
}
</style>
<div class="footer">
Botanical Classification System | Created with Streamlit
</div>
""", unsafe_allow_html=True)
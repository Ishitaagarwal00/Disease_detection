import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from PIL import Image

# Load models
@st.cache_resource
def load_model(file_path, model_type='keras'):
    if model_type == 'keras':
        return tf.keras.models.load_model(file_path)
    elif model_type == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)

malaria_model = load_model('malaria (1).h5', model_type='keras')
pneumonia_model = load_model('pneumonia.h5', model_type='keras')
diabetes_model = load_model('diabetes.pkl', model_type='pickle')
breast_cancer_model = load_model('brest_cancer (1).pkl', model_type='pickle')

# Streamlit UI
st.title("Disease Prediction App")

option = st.selectbox("Select Disease to Predict", ["Diabetes", "Breast Cancer", "Malaria", "Pneumonia"])

if option in ["Malaria", "Pneumonia"]:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = image.resize((150, 150))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        model = malaria_model if option == "Malaria" else pneumonia_model
        prediction = model.predict(image)
        result = "Positive" if prediction[0][0] > 0.5 else "Negative"
        st.write(f"Prediction: {result}")

elif option in ["Diabetes", "Breast Cancer"]:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data.head())
        
        model = diabetes_model if option == "Diabetes" else breast_cancer_model
        prediction = model.predict(data)
        result = ["Positive" if p == 1 else "Negative" for p in prediction]
        data['Prediction'] = result
        st.write("Prediction Results:")
        st.dataframe(data)

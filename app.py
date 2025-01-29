import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Constants
IMAGE_SIZE = 256
CLASS_NAMES = ['Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy']

# Custom CSS for UI styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #121212;
            color: white;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #ffcc00;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 18px;
            text-align: center;
            color: #bbb;
            margin-bottom: 20px;
        }
        .upload-box {
            border: 2px dashed #ffcc00;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            background-color: #222;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px;
            margin-top: 10px;
            background-color: #ffcc00;
            color: black;
            font-weight: bold;
        }
        .prediction-text {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            color: #16a085;
            margin-top: 15px;
        }
        .confidence-text {
            font-size: 18px;
            text-align: center;
            color: #d35400;
            margin-top: 10px;
        }
        .stProgress > div > div > div {
            background-color: #16a085 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model with caching
@st.cache_resource
def load_trained_model():
    return load_model("models/version_1.keras")  # Adjust the path if necessary

model = load_trained_model()

# Preprocessing function
def preprocess_image(uploaded_img):
    img = Image.open(uploaded_img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Create batch
    return img, img_array

# Prediction function
def predict(model, preprocessed_img):
    predictions = model.predict(preprocessed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, float(confidence)  # Convert confidence to Python float

# App Title
st.markdown("<h1 class='main-title'>Plant Disease System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-title'>A Plant Disease System for Sustainable Agriculture focuses on early detection, prevention, and management of plant diseases to ensure the health of crops while minimizing the environmental impact of agricultural practices.It promotes sustainable farming practices such as precision agriculture, integrated pest management, and biocontrol.The system helps reduce chemical use, increase crop yields, and protect the environment. It also provides farmers with real-time insights, disease forecasting, and personalized recommendations.</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload a potato leaf image to detect diseases.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

# Buttons
show_image = st.button("Show Image")
predict_btn = st.button("Predict")

# Logic for "Show Image" button
if show_image and uploaded_file is not None:
    img, _ = preprocess_image(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

# Logic for "Predict" button
if predict_btn and uploaded_file is not None:
    img, preprocessed_img = preprocess_image(uploaded_file)
    predicted_class, confidence = predict(model, preprocessed_img)

    # Display prediction
    st.markdown(f"<p class='prediction-text'>Prediction: {predicted_class}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='confidence-text'>Confidence: {confidence}%</p>", unsafe_allow_html=True)

    # Progress bar for confidence
    st.progress(confidence / 100)  # Ensure confidence is a standard float

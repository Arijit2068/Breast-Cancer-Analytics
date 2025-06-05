import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


# Set page config
st.set_page_config(page_title="Breast Cancer Detection", layout="centered")

# Title
st.title("Breast Cancer Classification")
st.write("Upload a mammogram image to classify it into one of the following:")
st.markdown("- **MALIGNANT**")
st.markdown("- **BENIGN**")
st.markdown("- **BENIGN_WITHOUT_CALLBACK**")

# Class names (must match training order)
class_names = ["BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"]

# Load the model
@st.cache_resource
def load_model_cached():
    model = load_model("model.h5")  # Adjust filename if needed
    return model

model = load_model_cached()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image",width=250)

    # Preprocess image
    img = img.resize((224, 224))  # Adjust if your model uses a different input shape
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize if used during training
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Show result
    st.success(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Show all class probabilities
    st.subheader("Class Probabilities:")
    for i, prob in enumerate(predictions):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set page config
st.set_page_config(page_title="Breast Cancer Detection", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
            border-radius: 12px;
        }
        h1, h2, h3, .stMarkdown, .stText, .stMetric {
            color: #007bff;
        }
        .stButton>button {
            background-color: #1f77b4 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        .uploadedFile {
            color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

# Main container
st.title("🔬 Breast Cancer Classification 🩺")
st.markdown("Upload a mammogram image to classify it into one of the following:")
st.markdown("- 🧬 **MALIGNANT**")
st.markdown("- 🩺 **BENIGN**")
st.markdown("- 📋 **BENIGN_WITHOUT_CALLBACK**")
st.markdown("---")

# Class names (must match training order)
class_names = ["BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"]

# Load model using cache
@st.cache_resource
def load_model_cached():
    return load_model("model.h5")

model = load_model_cached()

# File uploader
uploaded_file = st.file_uploader("📤 Upload a mammogram image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image",width=230)

    with col2:
        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

        # Display results
        # st.success(f"🎯 **Prediction:** `{predicted_class}`")
        # st.metric(label="Confidence", value=f"{confidence:.2f} %")
        # # Display prediction with larger font
        # st.markdown(
        #     f"""
        #     <div style='font-size:34px; font-weight:bold; color:green;'>
        #         🎯 Prediction: <code style='border:red;'>{predicted_class}</code>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
        # Styled prediction box
        st.markdown(
            f"""
            <div style='
                font-size:28px; 
                font-weight:bold; 
                color:#52cc29; 
                border: 2px solid red; 
                padding: 10px; 
                border-radius: 10px; 
                background-color: #222423;
                margin-bottom: 10px;
            '>
            🎯 Prediction: <span style='font-size:30px; color:red;'>{predicted_class}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display confidence in red
        st.markdown(
            f"""
            <div style='font-size:30px; font-weight:bold; color:#52cc29;'>
                Confidence: {confidence:.2f} %
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(predictions):
        st.write(f"**{class_names[i]}**: {prob*100:.2f}%")
        st.progress(float(prob))



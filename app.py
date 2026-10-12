import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained CNN model
st.write("Loading brain tumor model...")
model = tf.keras.models.load_model("brain_tumor_cnn.h5", compile=False)

# Page settings
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# Project Header
st.title("🧠 AI-BASED BRAIN TUMOR DETECTION")
st.subheader("MRI Image Classification using CNN")

st.write("### College Project")

st.markdown("""
**Developed By:**
- Devraj Patil
- Pradnya Kantekure
- Aditya Patil

**Department:** E&TC

**College:** SKNCOE
""")



st.divider()

# Project Overview
st.write("### 🔬 Project Overview")

st.write(
    "This application uses a Convolutional Neural Network (CNN) "
    "to classify brain MRI images into two categories: "
    "Normal Brain (No Tumor) and Tumor Brain."
)

st.divider()

# Upload MRI Image
st.write("### 📤 Upload Brain MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)



@st.cache_resource
def load_image_checker():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

image_checker = load_image_checker()

# Prediction
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("❌ Invalid image file. Please upload a valid image.")
        st.stop()

    st.image(image, caption="Uploaded Image", width="stretch")

    # Check whether the image looks like a common everyday object
    checker_image = image.resize((224, 224))
    checker_array = tf.keras.utils.img_to_array(checker_image)
    checker_array = np.expand_dims(checker_array, axis=0)
    checker_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        checker_array.copy()
    )

    checker_prediction = image_checker.predict(checker_array, verbose=0)
    labels = tf.keras.applications.mobilenet_v2.decode_predictions(
        checker_prediction, top=5
    )[0]

    everyday_objects = {
        "dog", "cat", "bird", "car", "truck", "bus", "bicycle",
        "motorcycle", "airplane", "boat", "person", "banana",
        "apple", "orange", "flower", "guitar", "keyboard",
        "laptop", "desktop_computer", "cellular_telephone",
        "coffee_mug", "chair", "sofa", "television", "remote_control"
    }

    obvious_object = any(
        label[1] in everyday_objects and label[2] >= 0.50
        for label in labels
    )

    if obvious_object:
        st.error("❌ Invalid Image: Please upload a brain MRI image.")
        st.stop()

    # Existing brain tumor CNN prediction
    image_resized = image.resize((128, 128))
    image_array = np.expand_dims(np.array(image_resized), axis=0)

    prediction = model.predict(image_array, verbose=0)[0][0]

    st.divider()
    st.write("### 🧠 Classification Result")

    if prediction >= 0.5:
        st.error("🔴 Tumor Brain")
        st.write(f"Model Confidence: {prediction * 100:.2f}%")
    else:
        st.success("🟢 Normal Brain (No Tumor)")
        st.write(f"Model Confidence: {(1 - prediction) * 100:.2f}%")



# Disclaimer
st.divider()

st.info(
    "⚠️ This application is an educational/research prototype "
    "and is not intended for medical diagnosis."
)

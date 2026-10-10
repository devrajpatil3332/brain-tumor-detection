import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained CNN model
model = tf.keras.models.load_model("brain_tumor_cnn.h5")

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


# Prediction
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("❌ Invalid image file. Please upload a valid image.")
        st.stop()

    st.image(
        image,
        caption="Uploaded MRI Image",
        width="stretch"
    )

    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0][0]

    st.divider()
    st.write("### 🧠 Classification Result")

    if prediction >= 0.5:
        st.error("🔴 Tumor Brain")
        st.write(f"Model Confidence: {prediction * 100:.2f}%")
    else:
        st.success("🟢 Normal Brain (No Tumor)")
        st.write(f"Model Confidence: {(1 - prediction) * 100:.2f}%")

    # Preprocessing
    image_resized = image.resize((128, 128))

    image_array = np.array(image_resized)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # CNN Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    st.divider()

    st.write("### 🧠 Classification Result")

    if prediction >= 0.5:

        st.error("🔴 Tumor Brain")

        st.write(
            f"Model Confidence: {prediction * 100:.2f}%"
        )

    else:

        st.success("🟢 Normal Brain (No Tumor)")

        st.write(
            f"Model Confidence: {(1 - prediction) * 100:.2f}%"
        )

# Disclaimer
st.divider()

st.info(
    "⚠️ This application is an educational/research prototype "
    "and is not intended for medical diagnosis."
)

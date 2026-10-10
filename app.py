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

# Load a general zero-shot image classifier
@st.cache_resource
def load_image_checker():
    from transformers import pipeline
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=-1
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
        caption="Uploaded Image",
        width="stretch"
    )

    with st.spinner("Checking whether this looks like a brain MRI..."):
        try:
            checker = load_image_checker()
            results = checker(
                image,
                candidate_labels=[
                    "a brain MRI scan image",
                    "a photograph of an everyday object or scene",
                    "a picture of a person, animal, or plant",
                    "a drawing or screenshot"
                ]
            )
        except Exception as e:
            st.error("Image checker could not load. Please try again later.")
            st.stop()

    top_result = results[0]

    if top_result["label"] != "a brain MRI scan image" or top_result["score"] < 0.50:
        st.error("❌ Invalid Image — Please upload a Brain MRI image.")
        st.stop()

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

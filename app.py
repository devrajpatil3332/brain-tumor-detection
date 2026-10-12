


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_brain_model():
    return tf.keras.models.load_model(
        "brain_tumor_cnn.h5",
        compile=False
    )

@st.cache_resource
def load_image_checker():
    return tf.keras.applications.MobileNetV2(
        weights="imagenet"
    )

# ---------------- PROJECT HEADER ----------------
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

# ---------------- PROJECT OVERVIEW ----------------
st.write("### 🔬 Project Overview")
st.write(
    "This application uses a CNN to classify brain MRI images "
    "into Normal Brain (No Tumor) and Tumor Brain categories."
)

st.divider()

# ---------------- IMAGE UPLOAD ----------------
st.write("### 📤 Upload Brain MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("❌ Invalid image file. Please upload a valid image.")
        st.stop()

    st.image(image, caption="Uploaded Image", width="stretch")

    with st.spinner("Checking image..."):
        # MobileNetV2 preprocessing
        checker_image = image.resize((224, 224))
        checker_array = tf.keras.utils.img_to_array(checker_image)
        checker_array = np.expand_dims(checker_array, axis=0)
        checker_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            checker_array
        )

        checker = load_image_checker()
        checker_prediction = checker.predict(
            checker_array,
            verbose=0
        )

        labels = tf.keras.applications.mobilenet_v2.decode_predictions(
            checker_prediction,
            top=5
        )[0]

    # Common non-MRI labels from ImageNet
    everyday_objects = {
        # Vehicles
        "car", "sports_car", "racer", "limousine", "convertible",
        "jeep", "minivan", "pickup", "truck", "trailer_truck",
        "bus", "school_bus", "ambulance", "fire_engine",
        "bicycle", "mountain_bike", "motorcycle", "airplane", "boat",

        # People and activities
        "person", "groom", "bridegroom", "soccer_player",
        "basketball", "swimmer", "scuba_diver", "ballet_dancer",

        # Animals
        "dog", "cat", "bird", "horse", "cow", "sheep", "goat",
        "pig", "rabbit", "hamster", "monkey", "lion", "tiger",
        "bear", "elephant", "zebra", "snake", "turtle", "fish",

        # Nature
        "tree", "pine_tree", "oak_tree", "flower", "daisy",
        "sunflower", "rose", "lakeside", "seashore", "alp",
        "valley", "volcano", "mountain", "cliff", "forest",
        "rainforest", "park", "beach", "coral_reef",

        # Food and household objects
        "banana", "apple", "orange", "pizza", "hamburger",
        "guitar", "keyboard", "laptop", "desktop_computer",
        "cellular_telephone", "coffee_mug", "chair", "sofa",
        "television", "remote_control", "bottle", "vase",
        "clock", "book", "backpack", "handbag", "umbrella",

        # Decorations and other objects
        "figurine", "mask", "altar", "pot", "sculpture",
        "statue", "ornament", "gasmask"
    }

    # Reject when a known everyday-object label has enough score
    obvious_object = any(
        label[1] in everyday_objects and float(label[2]) >= 0.03
        for label in labels
    )

    if obvious_object:
        st.error(
            "❌ Invalid Image: Please upload a brain MRI image."
        )
        st.stop()

    # ---------------- CNN PREDICTION ----------------
    with st.spinner("Analyzing brain image..."):
        image_resized = image.resize((128, 128))
        image_array = np.expand_dims(
            np.array(image_resized),
            axis=0
        )

        prediction = float(
            load_brain_model().predict(
                image_array,
                verbose=0
            )[0][0]
        )

    st.divider()
    st.write("### 🧠 Classification Result")

    if prediction >= 0.5:
        st.error("🔴 Tumor Brain")
        st.write(f"Model Confidence: {prediction * 100:.2f}%")
    else:
        st.success("🟢 Normal Brain (No Tumor)")
        st.write(f"Model Confidence: {(1 - prediction) * 100:.2f}%")

# ---------------- DISCLAIMER ----------------
st.divider()
st.info(
    "⚠️ This application is an educational/research prototype "
    "and is not intended for medical diagnosis."
)

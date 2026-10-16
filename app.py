
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="AI Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def load_brain_model():
    return tf.keras.models.load_model(
        "brain_tumor_cnn.h5",
        compile=False
    )


def generate_gradcam(model, image_array):
    try:
        inputs = tf.keras.Input(shape=(128, 128, 3))
        x = inputs
        conv_output = None

        for layer in model.layers:
            x = layer(x)
            if layer.name == "conv2d_1":
                conv_output = x

        if conv_output is None:
            return None

        grad_model = tf.keras.Model(
            inputs=inputs,
            outputs=[conv_output, x]
        )

        image_tensor = tf.convert_to_tensor(
            image_array, dtype=tf.float32
        )

        with tf.GradientTape() as tape:
            conv_features, output = grad_model(
                image_tensor, training=False
            )
            score = output[:, 0]

        gradients = tape.gradient(score, conv_features)

        if gradients is None:
            return None

        weights = tf.reduce_mean(
            gradients, axis=(1, 2), keepdims=True
        )

        heatmap = tf.reduce_sum(
            conv_features * weights, axis=-1
        )[0]

        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (
            tf.reduce_max(heatmap) + 1e-8
        )

        return heatmap.numpy()

    except Exception as error:
        print("Grad-CAM error:", error)
        return None


def create_overlay(image, heatmap):
    original = np.array(
        image.resize((128, 128)).convert("RGB")
    )

    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (128, 128)
    )

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap_uint8 = np.uint8(255 * heatmap)

    colored = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original, 0.55, colored, 0.45, 0
    )

    return original, overlay


st.title("🧠 AI-Based Brain Tumor Detection")
st.subheader("MRI Image Classification using CNN")

st.markdown("""
**Developed by:** Devraj Patil, Pradnya Kantekure, Aditya Patil

**Department:** E&TC | **College:** SKNCOE
""")

st.divider()
st.write("Upload a brain MRI image to classify it.")

uploaded_file = st.file_uploader(
    "Choose MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Invalid image file.")
        st.stop()

    st.image(image, caption="Uploaded MRI", width="stretch")

    resized = image.resize((128, 128))
    image_array = np.expand_dims(
        np.asarray(resized, dtype=np.float32),
        axis=0
    )

    with st.spinner("Analyzing MRI..."):
        try:
            model = load_brain_model()
            prediction = float(
                model.predict(image_array, verbose=0)[0][0]
            )
        except Exception as error:
            st.error("Could not load or run the model.")
            st.exception(error)
            st.stop()

    st.divider()
    st.subheader("📋 Classification Result")

    if prediction >= 0.5:
        st.error("🔴 Tumor Brain")
    else:
        st.success("🟢 Normal Brain (No Tumor)")

    st.caption(f"Raw model output: {prediction:.6f}")

    st.divider()
    st.subheader("🔥 Grad-CAM Heatmap")

    with st.spinner("Generating heatmap..."):
        heatmap = generate_gradcam(model, image_array)

    if heatmap is not None:
        original, overlay = create_overlay(image, heatmap)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original,
                caption="Original MRI",
                width="stretch"
            )

        with col2:
            st.image(
                overlay,
                caption="Grad-CAM Heatmap Overlay",
                width="stretch"
            )

        buffer = BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")

        st.download_button(
            "📥 Download Heatmap",
            data=buffer.getvalue(),
            file_name="gradcam_result.png",
            mime="image/png"
        )
    else:
        st.warning("Grad-CAM could not be generated.")

st.divider()
st.warning(
    "Educational project only. Not intended for medical diagnosis. "
    "Heatmap activation does not confirm tumor location."
)

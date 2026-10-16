
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI-Based Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
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


# ---------------- IMAGE CHECKER ----------------

EVERYDAY_OBJECTS = {
    "car", "sports_car", "racer", "limousine", "convertible",
    "jeep", "minivan", "pickup", "truck", "trailer_truck",
    "bus", "school_bus", "ambulance", "fire_engine",
    "bicycle", "mountain_bike", "motorcycle", "airplane", "boat",
    "person", "groom", "bridegroom", "soccer_player",
    "basketball", "swimmer", "scuba_diver", "ballet_dancer",
    "dog", "cat", "bird", "horse", "cow", "sheep", "goat",
    "pig", "rabbit", "hamster", "monkey", "lion", "tiger",
    "bear", "elephant", "zebra", "snake", "turtle", "fish",
    "tree", "pine_tree", "oak_tree", "flower", "daisy",
    "sunflower", "rose", "lakeside", "seashore", "alp",
    "valley", "volcano", "mountain", "cliff", "forest",
    "rainforest", "park", "beach", "coral_reef",
    "banana", "apple", "orange", "pizza", "hamburger",
    "guitar", "keyboard", "laptop", "desktop_computer",
    "cellular_telephone", "coffee_mug", "chair", "sofa",
    "television", "remote_control", "bottle", "vase",
    "clock", "book", "backpack", "handbag", "umbrella",
    "figurine", "mask", "altar", "pot", "sculpture",
    "statue", "ornament", "gasmask"
}


def looks_like_everyday_object(image):
    checker = load_image_checker()

    checker_image = image.resize((224, 224))
    checker_array = tf.keras.utils.img_to_array(
        checker_image
    )
    checker_array = np.expand_dims(
        checker_array, axis=0
    )
    checker_array = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            checker_array
        )
    )

    predictions = checker.predict(
        checker_array, verbose=0
    )

    labels = (
        tf.keras.applications.mobilenet_v2.decode_predictions(
            predictions, top=10
        )[0]
    )

    return any(
        label[1] in EVERYDAY_OBJECTS
        and float(label[2]) >= 0.03
        for label in labels
    )


# ---------------- GRAD-CAM ----------------

def generate_gradcam(model, image_array):
    try:
        # Rebuild a connected graph through the CNN layers
        inputs = tf.keras.Input(
            shape=(128, 128, 3)
        )
        x = inputs
        conv_output = None

        for layer in model.layers:
            x = layer(x)

            if layer.name == "conv2d_1":
                conv_output = x

        if conv_output is None:
            print("Last convolution layer not found.")
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

        gradients = tape.gradient(
            score, conv_features
        )

        if gradients is None:
            print("Gradients unavailable.")
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


# ---------------- HEATMAP OVERLAY ----------------

def make_heatmap_overlay(original_image, heatmap):
    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (128, 128),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (
        np.max(heatmap) + 1e-8
    )

    # Colorize normalized activation map
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    original = np.asarray(
        original_image.resize((128, 128)).convert("RGB"),
        dtype=np.uint8
    )

    overlay = cv2.addWeighted(
        original, 0.55,
        heatmap_color, 0.45,
        0
    )

    return original, overlay


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🧠 Brain MRI")
    st.markdown("### Navigation")
    st.markdown("🏠 Home")
    st.markdown("📘 About Project")
    st.markdown("🧠 Model Details")
    st.markdown("📊 Performance")
    st.markdown("🔥 Grad-CAM Heatmap")

    st.divider()
    st.caption("E&TC | SKNCOE")


# ---------------- HEADER ----------------

st.markdown(
    """
    <div style="text-align:center;">
        <h1>🧠 AI-BASED BRAIN TUMOR DETECTION</h1>
        <h3>MRI Image Classification using CNN</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        text-align:center;
        background:#eaf3ff;
        padding:15px;
        border-radius:10px;
        margin-bottom:20px;">
        <b>Developed By:</b>
        Devraj Patil | Pradnya Kantekure | Aditya Patil
        <br>
        <b>Department:</b> E&TC
        &nbsp; | &nbsp;
        <b>College:</b> SKNCOE
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- PROJECT OVERVIEW ----------------

with st.expander("📘 About Project"):
    st.write(
        "This educational project uses a Convolutional Neural "
        "Network (CNN) to classify brain MRI images into two "
        "categories: Tumor Brain and Normal Brain."
    )

with st.expander("🧠 Model Details"):
    st.write("Input image size: 128 × 128 × 3")
    st.write("Architecture: CNN with two convolution layers")
    st.write("Output: Single model score, threshold 0.5")

# ---------------- UPLOAD MRI ----------------

st.markdown("## 📤 Upload Brain MRI Image")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(
            uploaded_file
        ).convert("RGB")
    except Exception:
        st.error(
            "Invalid image file. Please upload a valid image."
        )
        st.stop()

    # Check whether upload looks like a common object.
    # ImageNet checking is imperfect and cannot validate MRI reliably.
    with st.spinner("Checking uploaded image..."):
        try:
            if looks_like_everyday_object(image):
                st.error(
                    "This image may not be a brain MRI. "
                    "Please upload a brain MRI image."
                )
                st.stop()
        except Exception as error:
            print("Image checker error:", error)
            st.warning(
                "Automatic image checking was unavailable. "
                "Please make sure the image is a brain MRI."
            )

    # Preserve model's original preprocessing: RGB pixels 0-255
    image_resized = image.resize((128, 128))

    image_array = np.expand_dims(
        np.asarray(
            image_resized,
            dtype=np.float32
        ),
        axis=0
    )

    # ---------------- CLASSIFICATION ----------------

    with st.spinner("Analyzing MRI image..."):
        try:
            model = load_brain_model()

            prediction = float(
                model.predict(
                    image_array, verbose=0
                )[0][0]
            )

        except Exception as error:
            st.error("Could not run the CNN model.")
            st.exception(error)
            st.stop()

    st.divider()
    st.markdown("## 📋 Classification Result")

    image_col, result_col = st.columns(2)

    with image_col:
        st.markdown("### 🖼️ Uploaded MRI Image")
        st.image(
            image,
            caption="Original MRI Image",
            width="stretch"
        )

    with result_col:
        st.markdown("### ⚙️ CNN Prediction")

        if prediction >= 0.5:
            st.error("🔴 Tumor Brain")
        else:
            st.success("🟢 Normal Brain (No Tumor)")

        st.metric(
            "Raw Model Output",
            f"{prediction:.6f}"
        )

        st.caption(
            "Raw model output is not a calibrated medical "
            "confidence score."
        )

    # ---------------- GRAD-CAM ----------------

    st.divider()
    st.markdown(
        "## 🔥 Grad-CAM Heatmap (AI Explanation)"
    )

    with st.spinner(
        "Generating Grad-CAM visualization..."
    ):
        heatmap = generate_gradcam(
            model, image_array
        )

    if heatmap is not None:
        original, overlay = make_heatmap_overlay(
            image, heatmap
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original,
                caption="Original MRI Image",
                width="stretch"
            )

        with col2:
            st.image(
                overlay,
                caption="Grad-CAM Heatmap Overlay",
                width="stretch"
            )

        st.markdown(
            """
            <div style="
                background:#eaf3ff;
                padding:12px;
                border-radius:8px;
                margin-top:8px;">
                💡 Highlighted regions show areas influencing the
                model output. Red/yellow generally indicates stronger
                activation, while blue indicates weaker activation.
            </div>
            """,
            unsafe_allow_html=True
        )

        output_buffer = BytesIO()
        Image.fromarray(overlay).save(
            output_buffer, format="PNG"
        )

        st.download_button(
            label="📥 Download Grad-CAM Image",
            data=output_buffer.getvalue(),
            file_name="gradcam_result.png",
            mime="image/png"
        )

    else:
        st.warning(
            "Grad-CAM could not be generated. "
            "Check the terminal for error details."
        )

# ---------------- DISCLAIMER ----------------

st.divider()

st.warning(
    "Educational/research prototype only. Not intended for "
    "clinical use or medical diagnosis. Predictions and heatmaps "
    "may be inaccurate. A highlighted region does not confirm "
    "the presence or exact location of a tumor."
)

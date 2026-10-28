
import os
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, Image as PDFImage
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0b1220,#14243a);
    color: #f4f7fb;
}
[data-testid="stSidebar"] {
    background: #091321;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(110deg,#18365f,#176174);
    border: 1px solid #31506d;
    margin-bottom: 22px;
}
.hero h1 { color: white; }
.hero p { color: #d7e9f7; font-size: 17px; }
.info-card {
    background: #17253a;
    border: 1px solid #2a405c;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}
</style>
""", unsafe_allow_html=True)


MODEL_PATH = "brain_tumor_cnn.h5"
IMAGE_SIZE = (128, 128)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} was not found in the repository."
        )
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )


# ---------------- PREDICTION ----------------
def predict_image(model, image_array):
    result = model.predict(image_array, verbose=0)

    if isinstance(result, (list, tuple)):
        result = result[0]

    result = np.asarray(result)

    if result.size != 1:
        raise ValueError(
            f"Expected one binary output, received shape {result.shape}. "
            "This app currently supports a binary classifier."
        )

    return float(result.reshape(-1)[0])


# ---------------- GRAD-CAM ----------------
def get_last_conv_layer(model):
    """Find a directly accessible Conv2D layer."""
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    if conv_layers:
        return conv_layers[-1]

    # Some models wrap the CNN inside a nested model.
    # Inspect nested layers, but report failure if the graph
    # does not expose the intermediate output correctly.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            nested_convs = [
                sublayer for sublayer in layer.layers
                if isinstance(sublayer, tf.keras.layers.Conv2D)
            ]
            if nested_convs:
                return nested_convs[-1]

    return None


def make_gradcam(model, image_array):
    try:
        conv_layer = get_last_conv_layer(model)

        if conv_layer is None:
            return None, (
                "No accessible convolutional layer was found. "
                "The model architecture must be checked."
            )

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )

        image_tensor = tf.convert_to_tensor(
            image_array, dtype=tf.float32
        )

        with tf.GradientTape() as tape:
            conv_features, output = grad_model(
                image_tensor, training=False
            )
            output = tf.convert_to_tensor(output)
            score = (
                output[:, 0]
                if len(output.shape) > 1
                else output
            )

        gradients = tape.gradient(score, conv_features)

        if gradients is None:
            return None, "Grad-CAM gradients were unavailable."

        weights = tf.reduce_mean(
            gradients, axis=(1, 2), keepdims=True
        )

        heatmap = tf.reduce_sum(
            conv_features * weights, axis=-1
        )[0]

        heatmap = tf.maximum(heatmap, 0)
        heatmap = tf.math.divide_no_nan(
            heatmap, tf.reduce_max(heatmap)
        )

        return heatmap.numpy(), None

    except Exception as error:
        return None, (
            "Grad-CAM could not be generated for this model. "
            f"Details: {type(error).__name__}: {error}"
        )


def make_overlay(image, heatmap):
    original = np.asarray(
        image.resize(IMAGE_SIZE).convert("RGB")
    )

    heatmap = cv2.resize(
        heatmap.astype(np.float32), IMAGE_SIZE
    )
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    colored = cv2.cvtColor(
        colored, cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original, 0.60, colored, 0.40, 0
    )

    return original, overlay


# ---------------- PDF REPORT ----------------
def create_pdf(filename, label, raw_output, timestamp, overlay=None):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportHeading",
        parent=styles["Title"],
        textColor=colors.HexColor("#15385b"),
        fontSize=19
    ))

    story = [
        Paragraph(
            "NeuroVision AI - Brain MRI Report",
            styles["ReportHeading"]
        ),
        Spacer(1, 12),
        Paragraph(
            "Educational research prototype only. "
            "Not a clinical diagnosis.",
            styles["Normal"]
        ),
        Spacer(1, 15)
    ]

    rows = [
        ["Image filename", str(filename)],
        ["Analysis time", timestamp],
        ["Model output label", label],
        ["Raw model output", f"{raw_output:.6f}"],
        ["Threshold assumption", "0.5"]
    ]

    table = Table(
        rows, colWidths=[48 * mm, 112 * mm]
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1),
         colors.HexColor("#eaf1f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))

    story.extend([table, Spacer(1, 15)])

    if overlay is not None:
        image_buffer = BytesIO()
        Image.fromarray(overlay).save(
            image_buffer, format="PNG"
        )
        image_buffer.seek(0)

        story.append(
            Paragraph("Grad-CAM visualization", styles["Heading2"])
        )
        story.append(PDFImage(
            image_buffer, width=115 * mm, height=115 * mm
        ))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "The raw model output is not a clinically calibrated "
        "probability. This report assumes output >= 0.5 means "
        "the tumor class and output < 0.5 means the normal class. "
        "Verify this mapping against the original training labels. "
        "Grad-CAM highlights model activation and does not confirm "
        "tumor presence or location. Do not use this report for "
        "medical decisions.",
        styles["BodyText"]
    ))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 NeuroVision AI")
    st.caption("Explainable Brain MRI Research")

    page = st.radio(
        "Navigation",
        ["Dashboard", "MRI Analysis", "About Project"]
    )

    st.divider()
    st.warning(
        "Educational prototype only. "
        "Not intended for clinical diagnosis."
    )


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>🧠 NeuroVision AI</h1>
        <p>Brain MRI Classification with Explainable AI</p>
        <p>Deep Learning • CNN • Grad-CAM • Visual Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 CNN</h3>
            <p>Binary MRI classifier</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🔥 Grad-CAM</h3>
            <p>Visual model explanation</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>📄 PDF Report</h3>
            <p>Downloadable analysis report</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Project Overview")
    st.write(
        "Upload an MRI image to view the existing model output "
        "and, when supported, a Grad-CAM visualization."
    )
    st.info("Select **MRI Analysis** in the left sidebar to begin.")


# ---------------- MRI ANALYSIS ----------------
elif page == "MRI Analysis":
    st.markdown("""
    <div class="hero">
        <h1>📊 MRI Analysis</h1>
        <p>Upload a JPG or PNG brain MRI image.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Could not open this image file.")
            st.stop()

        model_input = np.expand_dims(
            np.asarray(
                image.resize(IMAGE_SIZE), dtype=np.float32
            ),
            axis=0
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image, caption="Uploaded MRI",
                use_container_width=True
            )

        with st.spinner("Running CNN model..."):
            try:
                model = load_model()
                raw_output = predict_image(model, model_input)
            except Exception as error:
                st.error("The model could not be loaded or run.")
                st.exception(error)
                st.stop()

        # Assumes the original binary model uses this mapping.
        label = (
            "Tumor class"
            if raw_output >= 0.5
            else "Normal class"
        )

        with col2:
            st.subheader("Classification Result")

            if raw_output >= 0.5:
                st.error("🔴 Model output: Tumor class")
            else:
                st.success("🟢 Model output: Normal class")

            st.metric("Raw model output", f"{raw_output:.6f}")
            st.caption(
                "This is a raw model output, not a clinically "
                "calibrated probability. Verify the class mapping "
                "against the original training labels."
            )

        st.divider()
        st.subheader("🔥 Grad-CAM Explanation")

        with st.spinner("Attempting Grad-CAM..."):
            heatmap, cam_error = make_gradcam(
                model, model_input
            )

        overlay = None

        if heatmap is not None:
            original, overlay = make_overlay(image, heatmap)

            cam_col1, cam_col2 = st.columns(2)

            with cam_col1:
                st.image(
                    original, caption="MRI input",
                    use_container_width=True
                )

            with cam_col2:
                st.image(
                    overlay, caption="Grad-CAM overlay",
                    use_container_width=True
                )

            image_buffer = BytesIO()
            Image.fromarray(overlay).save(
                image_buffer, format="PNG"
            )

            st.download_button(
                "📥 Download Grad-CAM Image",
                data=image_buffer.getvalue(),
                file_name="gradcam_overlay.png",
                mime="image/png"
            )
        else:
            st.warning(cam_error)
            st.caption(
                "The model prediction is available, but Grad-CAM "
                "requires a compatible, accessible convolutional layer."
            )

        st.divider()
        st.subheader("📄 Download PDF Report")

        if not REPORTLAB_AVAILABLE:
            st.error(
                "ReportLab is missing. Add 'reportlab' to "
                "requirements.txt and redeploy."
            )
        else:
            pdf_data = create_pdf(
                uploaded_file.name,
                label,
                raw_output,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                overlay
            )

            if pdf_data:
                st.download_button(
                    "Download PDF Report",
                    data=pdf_data,
                    file_name="neurovision_mri_report.pdf",
                    mime="application/pdf"
                )

        st.warning(
            "Educational prototype only. This output is not a diagnosis. "
            "Grad-CAM does not prove tumor presence or location. "
            "Consult a qualified medical professional for health concerns."
        )

    else:
        st.info("Upload an MRI image above to start analysis.")


# ---------------- ABOUT PROJECT ----------------
else:
    st.markdown("""
    <div class="hero">
        <h1>ℹ️ About NeuroVision AI</h1>
        <p>Explainable AI-based Brain MRI Classification</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Objectives")
    st.markdown("""
    - Classify MRI images using the existing trained CNN.
    - Display the binary model output.
    - Attempt Grad-CAM visualization when the architecture supports it.
    - Export a PDF report for educational demonstration.
    """)

    st.subheader("Technology Stack")
    st.write(
        "Python • Streamlit • TensorFlow/Keras • OpenCV • "
        "Pillow • ReportLab"
    )

    st.subheader("Project Team")
    st.write("Devraj Patil • Pradnya Kantekure • Aditya Patil")
    st.write("Department: E&TC | College: SKNCOE")

    st.warning(
        "This is an academic prototype, not a validated medical device. "
        "Do not use its output for diagnosis or treatment."
    )

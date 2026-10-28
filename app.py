
import os
from io import BytesIO, StringIO
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor AI Analyzer",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH = "brain_tumor_cnn.h5"
IMAGE_SIZE = (128, 128)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f6f8fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #132743, #1d5c83);
        color: white;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .small-note {
        color: #64748b;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' sapadla nahi."
        )

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Model initialize kara
    dummy_input = tf.zeros(
        (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        dtype=tf.float32
    )

    loaded_model(dummy_input, training=False)

    return loaded_model


# --------------------------------------------------
# MODEL ARCHITECTURE DIAGNOSTICS
# --------------------------------------------------

def get_model_summary(model):
    stream = StringIO()
    model.summary(print_fn=lambda line: stream.write(line + "\n"))
    return stream.getvalue()


def describe_layers(model):
    details = []

    for index, layer in enumerate(model.layers):
        try:
            output_shape = str(layer.output.shape)
        except Exception:
            output_shape = "Unavailable"

        details.append({
            "Index": index,
            "Layer": layer.name,
            "Type": layer.__class__.__name__,
            "Output shape": output_shape
        })

    return details


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(pil_image):
    image = pil_image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Training madhye normalization vaparla asel tar
    # ithe tech normalization vaprane avashyak aahe.
    return np.expand_dims(image_array, axis=0)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_image(model, image_array):
    prediction = model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(prediction)

    if prediction.size != 1:
        raise ValueError(
            "Ha app single-output binary classification model "
            "expect karto. Model output shape: "
            + str(prediction.shape)
        )

    score = float(prediction.reshape(-1)[0])

    if not np.isfinite(score):
        raise ValueError("Model ne valid prediction dili nahi.")

    # IMPORTANT:
    # He mapping model training labels var avalambun aahe.
    # Training madhye 1 = Tumor asel tarach hi mapping barobar aahe.
    if score >= 0.5:
        label = "Tumor"
        confidence = score * 100
    else:
        label = "Normal"
        confidence = (1.0 - score) * 100

    return label, confidence, score


# --------------------------------------------------
# FIND LAST CONVOLUTION LAYER
# --------------------------------------------------

def find_last_conv_layer(model):
    """
    Direct Conv2D layer shodhto.
    Nested models madhye asel tar tyanchyat suddha shodhto.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

        if isinstance(layer, tf.keras.Model):
            nested_layer = find_last_conv_layer(layer)
            if nested_layer is not None:
                return nested_layer

    return None


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------


def make_gradcam(model, image_array):
    try:
        image_tensor = tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )

        # Build the model first
        _ = model(image_tensor, training=False)

        # Screenshot madhil second Conv2D layer
        conv_layer = model.get_layer("conv2d_1")

        # Feature maps and prediction must share the same graph
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                conv_layer.output,
                model.outputs[0]
            ]
        )

        with tf.GradientTape() as tape:
            conv_features, predictions = grad_model(
                image_tensor,
                training=False
            )

            # Binary classifier: one output neuron
            score = predictions[:, 0]

        gradients = tape.gradient(
            score,
            conv_features
        )

        if gradients is None:
            return None, (
                "Gradient None aahe. Model input/output graph "
                "punha verify karava lagel."
            )

        # Average gradients across spatial dimensions
        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2),
            keepdims=True
        )

        # Weighted sum of convolution feature maps
        heatmap = tf.reduce_sum(
            conv_features * weights,
            axis=-1
        )[0]

        # Keep positive activations
        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)

        heatmap = tf.math.divide_no_nan(
            heatmap,
            max_value
        )

        heatmap = heatmap.numpy()

        if not np.isfinite(heatmap).all():
            return None, "Heatmap madhye invalid values aahet."

        if np.max(heatmap) <= 0:
            return None, (
                "Heatmap blank aahe. Dusri Conv2D layer "
                "test karavi lagel."
            )

        return heatmap, None

    except Exception as error:
        return None, (
            f"Grad-CAM error: {type(error).__name__}: {error}"
        )

def make_overlay(original_image, heatmap):
    try:
        original = original_image.convert("RGB")
        original_array = np.asarray(original)

        resized_heatmap = cv2.resize(
            heatmap,
            (original_array.shape[1], original_array.shape[0])
        )

        heatmap_uint8 = np.uint8(
            255 * np.clip(resized_heatmap, 0, 1)
        )

        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        colored_heatmap = cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_BGR2RGB
        )

        overlay = cv2.addWeighted(
            original_array,
            0.60,
            colored_heatmap,
            0.40,
            0
        )

        return Image.fromarray(overlay)

    except Exception:
        return None


# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------

def create_pdf_report(
    label,
    confidence,
    raw_score,
    uploaded_filename
):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    page_width, page_height = A4

    pdf.setTitle("Brain MRI AI Analysis Report")

    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawString(
        50,
        page_height - 60,
        "Brain MRI AI Analysis Report"
    )

    pdf.setFont("Helvetica", 11)

    y = page_height - 110

    report_lines = [
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Uploaded image: {uploaded_filename}",
        "",
        f"Model output category: {label}",
        f"Displayed model confidence: {confidence:.2f}%",
        f"Raw model output: {raw_score:.6f}",
        "",
        "Project: Explainable AI-based brain tumor classification",
        "Method: CNN classification with Grad-CAM visualization",
        "",
        "IMPORTANT MEDICAL DISCLAIMER",
        "This report is generated by an academic AI prototype.",
        "It is not a medical diagnosis and must not be used",
        "to make treatment or healthcare decisions.",
        "A qualified radiologist or medical professional must",
        "interpret MRI images and confirm any findings."
    ]

    for line in report_lines:
        if y < 60:
            pdf.showPage()
            y = page_height - 60
            pdf.setFont("Helvetica", 11)

        pdf.drawString(50, y, line)
        y -= 23

    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🧠 Brain Tumor MRI Analyzer</h1>
        <p>Explainable AI-based brain tumor classification using CNN
        and Grad-CAM</p>
        <p>Final-Year ENTC Major Project</p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🧠 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "MRI Analysis",
        "About Project",
        "Model Diagnostics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Academic prototype only. Not intended for clinical diagnosis."
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.subheader("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Model", "CNN")

    with col2:
        st.metric("Explainability", "Grad-CAM")

    with col3:
        st.metric("Input", "MRI Image")

    st.markdown("### Project Overview")

    st.write(
        "This application demonstrates binary classification "
        "of an uploaded MRI image using a trained CNN model. "
        "Grad-CAM is used to visualize image regions that influence "
        "the model's output, when the model architecture supports it."
    )

    st.info(
        "Grad-CAM visualization may not be available for every "
        "saved model architecture. A heatmap is not proof that "
        "the model has correctly identified a tumor."
    )


# --------------------------------------------------
# MRI ANALYSIS
# --------------------------------------------------

elif page == "MRI Analysis":

    st.subheader("MRI Image Analysis")

    st.write(
        "Upload an MRI image to run the trained CNN model."
    )

    uploaded_file = st.file_uploader(
        "Upload MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            original_image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns(2)

            with left:
                st.markdown("#### Uploaded MRI")
                st.image(
                    original_image,
                    caption="Uploaded MRI image",
                    use_container_width=True
                )

            with st.spinner("Loading CNN model..."):
                model = load_model()

            image_array = preprocess_image(original_image)

            with st.spinner("Analyzing image..."):
                label, confidence, raw_score = predict_image(
                    model,
                    image_array
                )

            with right:
                st.markdown("#### Classification Result")

                st.metric(
                    "Predicted Category",
                    label
                )

                st.metric(
                    "Model Score",
                    f"{confidence:.2f}%"
                )

                st.caption(
                    "This percentage is the model's output-derived "
                    "score, not a clinically validated probability."
                )

            if label == "Tumor":
                st.warning(
                    "The model classified this image as Tumor. "
                    "This is not a diagnosis; please consult a "
                    "qualified medical professional."
                )
            else:
                st.success(
                    "The model classified this image as Normal. "
                    "This does not rule out disease or replace "
                    "professional medical review."
                )

            st.markdown("---")
            st.subheader("Grad-CAM Explainability")

            with st.spinner("Generating Grad-CAM heatmap..."):
                heatmap, gradcam_error = make_gradcam(
                    model,
                    image_array
                )

            if heatmap is not None:
                overlay = make_overlay(
                    original_image,
                    heatmap
                )

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("#### Grad-CAM Heatmap")
                    st.image(
                        heatmap,
                        caption="Activation intensity",
                        clamp=True,
                        use_container_width=True
                    )

                with col_b:
                    st.markdown("#### MRI + Heatmap Overlay")

                    if overlay is not None:
                        st.image(
                            overlay,
                            caption="Highlighted model activation regions",
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            "Overlay image tayar karta aala nahi."
                        )

                st.caption(
                    "Grad-CAM highlights regions associated with the "
                    "model score. It does not independently confirm "
                    "a tumor's presence or location."
                )

            else:
                st.error("Grad-CAM tayar karta aala nahi.")
                st.code(gradcam_error or "Unknown Grad-CAM error.")

                st.info(
                    "Sidebar madhun Model Diagnostics page open kara. "
                    "Tithe model architecture summary milu shakto."
                )

            st.markdown("---")
            st.subheader("Download Report")

            if REPORTLAB_AVAILABLE:
                pdf_bytes = create_pdf_report(
                    label,
                    confidence,
                    raw_score,
                    uploaded_file.name
                )

                if pdf_bytes:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name="brain_mri_analysis_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning(
                    "PDF report sathi requirements.txt madhye "
                    "'reportlab' add kara."
                )

        except Exception as error:
            st.error(
                f"Analysis madhye error aala: "
                f"{type(error).__name__}: {error}"
            )

            st.info(
                "Model file, input image size ani model output "
                "format check kara."
            )


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.subheader("About the Project")

    st.markdown(
        """
        **Project Title:** Explainable AI-Based Brain Tumor
        Classification from MRI Using CNN and Grad-CAM

        **Objectives**
        - Classify MRI images using a trained CNN.
        - Display the model's predicted category.
        - Generate Grad-CAM visualization where supported.
        - Provide a downloadable PDF summary.

        **Technologies**
        - Python
        - TensorFlow / Keras
        - Streamlit
        - OpenCV
        - NumPy
        - Matplotlib-compatible heatmap coloring
        - ReportLab for PDF reports

        **Important limitation**

        The output depends on the model's training data, preprocessing,
        class-label mapping and architecture. It is an academic
        demonstration, not a clinically validated diagnostic tool.
        """
    )


# --------------------------------------------------
# MODEL DIAGNOSTICS
# --------------------------------------------------

elif page == "Model Diagnostics":

    st.subheader("CNN Model Diagnostics")

    st.write(
        "Grad-CAM error solve karayla model architecture samjun "
        "ghene mahatvache aahe."
    )

    if st.button("Load Model and Show Architecture"):

        try:
            with st.spinner("Loading model architecture..."):
                model = load_model()

            st.success("Model successfully loaded.")

            st.markdown("### Model Summary")
            st.code(
                get_model_summary(model),
                language="text"
            )

            st.markdown("### Layer Details")

            st.dataframe(
                describe_layers(model),
                use_container_width=True
            )

            conv_layer = find_last_conv_layer(model)

            if conv_layer is not None:
                st.write(
                    "Last convolution layer candidate:",
                    conv_layer.name
                )
                st.write(
                    "Layer type:",
                    conv_layer.__class__.__name__
                )
            else:
                st.warning(
                    "Conv2D layer sapadli nahi. Grad-CAM sathi "
                    "model architecture inspect karavi lagel."
                )

            st.markdown("### Model Input / Output")

            st.write("Input shape:", str(model.input_shape))
            st.write("Output shape:", str(model.output_shape))

            st.warning(
                "Architecture summary madhye nested models disat "
                "astil ani gradients disconnected yet astil, tar "
                "Grad-CAM code la actual model graph nusar adjust "
                "karava lagel."
            )

        except Exception as error:
            st.error(
                f"Model diagnostics failed: "
                f"{type(error).__name__}: {error}"
            )

st.markdown("---")

st.caption(
    "Educational prototype | CNN + Grad-CAM | "
    "Not for clinical diagnosis"
)

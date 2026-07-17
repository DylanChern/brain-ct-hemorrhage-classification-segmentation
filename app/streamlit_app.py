"""
Brain CT Hemorrhage Classification — Streamlit Demo App
Loads the trained SoftMax CNN (CNNImprovedModel_best.keras) and classifies
a brain CT slice into one of 7 classes: epidural, intraparenchymal,
intraventricular, subarachnoid, subdural, normal, multi.
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import os

# ---------------------------------------------------------------------------
# 1. Class mapping — confirmed from the training notebook's class_map dict.
#    This MUST match training order exactly, or predictions will show the
#    wrong label even if the model itself is correct.
# ---------------------------------------------------------------------------
CLASS_MAP = {
    "epidural": 0,
    "intraparenchymal": 1,
    "intraventricular": 2,
    "subarachnoid": 3,
    "subdural": 4,
    "normal": 5,
    "multi": 6,
}
# Invert it: index -> label, since the model outputs are ordered by index
IDX_TO_LABEL = {v: k for k, v in CLASS_MAP.items()}

# Human-friendly display names for the UI
DISPLAY_NAMES = {
    "epidural": "Epidural Hemorrhage",
    "intraparenchymal": "Intraparenchymal Hemorrhage",
    "intraventricular": "Intraventricular Hemorrhage",
    "subarachnoid": "Subarachnoid Hemorrhage",
    "subdural": "Subdural Hemorrhage",
    "normal": "Normal (No Hemorrhage)",
    "multi": "Multiple Hemorrhage Types",
}

# Sample image filenames — must exist in sample_images/ alongside this script
SAMPLE_IMAGES = {
    "epidural": "epidural_sample.png",
    "intraparenchymal": "intraparenchymal_sample.png",
    "intraventricular": "intraventricular_sample.png",
    "subarachnoid": "subarachnoid_sample.png",
    "subdural": "subdural_sample.png",
    "multi": "multi_sample.jpg",
    "normal": "normal_sample.jpg",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn", "CNNImprovedModel_best.keras")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")


# ---------------------------------------------------------------------------
# 2. Model loading — cached so it only loads once per session, not on every
#    click. This matters a lot for Streamlit Cloud, where reloading a Keras
#    model on every interaction would be painfully slow.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# 3. Preprocessing — must exactly mirror what was done at training time:
#    grayscale, resize to 128x128, normalize to [0, 1], add batch + channel dims.
# ---------------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("L")           # force grayscale (1 channel)
    img = img.resize((128, 128))           # match training input size
    arr = np.array(img, dtype=np.float32) / 255.0   # normalize to 0-1
    arr = arr.reshape(1, 128, 128, 1)      # (batch, height, width, channels)
    return arr


def predict(model, pil_image: Image.Image):
    x = preprocess_image(pil_image)
    probs = model.predict(x, verbose=0)[0]  # shape (7,)
    pred_idx = int(np.argmax(probs))
    pred_label = IDX_TO_LABEL[pred_idx]
    return pred_label, probs


# ---------------------------------------------------------------------------
# 4. UI layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Brain CT Hemorrhage Classifier", layout="centered")

st.title("Brain CT Hemorrhage Classifier")
st.write(
    "A SoftMax CNN trained to classify brain CT slices into 7 categories: "
    "5 hemorrhage subtypes, normal, or multiple hemorrhage types. "
    "Built as part of a MATH 7243 (Machine Learning I) final project at Northeastern University."
)
st.caption(
    "⚠️ This is an academic demo, not a diagnostic tool. The underlying model achieved "
    "~59% validation accuracy — see the project README for full results and limitations."
)

st.divider()

# --- Image source selection ---
source = st.radio("Choose an image source:", ["Use a sample image", "Upload your own"])

image = None

if source == "Use a sample image":
    class_choice = st.selectbox(
        "Pick a hemorrhage type to see a sample scan:",
        options=list(SAMPLE_IMAGES.keys()),
        format_func=lambda k: DISPLAY_NAMES[k],
    )
    sample_path = os.path.join(SAMPLE_DIR, SAMPLE_IMAGES[class_choice])
    image = Image.open(sample_path)
    st.caption(
        f"Sample image sourced from Radiopaedia.org (CC BY-NC-SA 3.0) — "
        f"see README for full case attribution."
    )
else:
    uploaded_file = st.file_uploader("Upload a brain CT image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# --- Display + predict ---
if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input image", use_container_width=True)

    with st.spinner("Loading model and running prediction..."):
        model = load_model()
        pred_label, probs = predict(model, image)

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### {DISPLAY_NAMES[pred_label]}")
        st.write(f"Confidence: **{probs[CLASS_MAP[pred_label]] * 100:.1f}%**")

    st.divider()
    st.subheader("Confidence by class")
    # Build a sorted list of (display_name, probability) for a clean bar chart
    chart_data = {
        DISPLAY_NAMES[label]: float(probs[idx])
        for label, idx in CLASS_MAP.items()
    }
    st.bar_chart(chart_data)
else:
    st.info("Select a sample image or upload your own to get a prediction.")

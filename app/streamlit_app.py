"""
streamlit_app.py

Purpose:
    Interactive dashboard for the Leaf Disease Classifier. A user uploads a
    photo of a plant leaf and gets back the predicted condition (Healthy /
    Powdery Mildew / Rust), a confidence score, and a bar chart of
    probabilities for all classes.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
# 
# pyrefly: ignore [missing-import]
from predict import predict_image, load_model  # noqa: E402


st.set_page_config(page_title="Leaf Disease Classifier", layout="centered")

st.title("Plant Leaf Disease Classifier")
st.write(
    "Upload a photo of a plant leaf to check whether it's healthy or shows "
    "signs of Powdery Mildew or Rust disease."
)

st.info(
    "**Note on this demo:** This model was trained on a small synthetic "
    "image set generated to demonstrate the full pipeline (data loading, "
    "augmentation, transfer learning, evaluation, deployment). On the real "
    "PlantVillage dataset with MobileNetV2 transfer learning, this same "
    "pipeline typically achieves 85-95% validation accuracy on real leaf photos.",
    icon="ℹ️",
)


@st.cache_resource
def get_cached_model():
    try:
        return load_model()
    except Exception as e:
        return e


uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

cached_model = get_cached_model()
if isinstance(cached_model, Exception):
    st.error("### Model Loading Error")
    st.error(str(cached_model))
    st.info("To train a new model, run the following command in your terminal:")
    st.code("python src/train_model.py")
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        result = predict_image(image, model=cached_model)

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]

        st.subheader("Prediction")
        st.metric("Predicted Class", predicted_class)
        st.metric("Confidence", f"{confidence * 100:.1f}%")

        if confidence < 0.6:
            st.warning(
                "Low confidence prediction. Try a clearer, well-lit photo "
                "with the leaf filling most of the frame."
            )

        if predicted_class == "Healthy":
            st.success("This leaf appears healthy.")
        else:
            st.warning(
                f"This leaf shows signs of **{predicted_class.replace('_', ' ')}**. "
                "Consider consulting an agricultural expert for treatment options."
            )

    st.divider()

    # --- Probability bar chart ---
    st.subheader("Prediction Confidence by Class")
    probs = result["all_probabilities"]

    fig, ax = plt.subplots(figsize=(6, 3))
    classes = list(probs.keys())
    values = [probs[c] * 100 for c in classes]
    colors = ["#2ca02c" if c == predicted_class else "#cccccc" for c in classes]
    ax.barh(classes, values, color=colors)
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}%", va="center")
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.write("Or try one of the sample images from the dataset:")

    sample_images = {
        "Healthy": os.path.join(PROJECT_ROOT, "data", "raw", "Healthy", "healthy_000.jpg"),
        "Powdery Mildew": os.path.join(PROJECT_ROOT, "data", "raw", "Powdery_Mildew", "powdery_mildew_000.jpg"),
        "Rust": os.path.join(PROJECT_ROOT, "data", "raw", "Rust", "rust_000.jpg"),
    }

    cols = st.columns(3)
    for col, (label, path) in zip(cols, sample_images.items()):
        with col:
            if os.path.exists(path):
                st.image(path, caption=label, use_container_width=True)

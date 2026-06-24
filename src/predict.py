"""
predict.py

Purpose:
    Provide a single, reusable function that takes an image (as a file path,
    PIL Image, or numpy array) and returns the predicted disease class with
    a confidence score and per-class probabilities.

    This is the file the Streamlit app calls.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from dataset import IMG_SIZE, CLASS_NAMES
from tf_compat import require_tensorflow, tf


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leaf_disease_model.keras")


def load_model(path: str = MODEL_PATH) -> tf.keras.Model:
    """Load the trained Keras model from disk."""
    require_tensorflow()
    return tf.keras.models.load_model(path)


def preprocess_image(image) -> np.ndarray:
    """
    Convert an input image into the format the model expects:
    a batch of shape (1, IMG_SIZE, IMG_SIZE, 3) with raw pixel values [0, 255].

    Why [0, 255] and not normalized here?
    - The model's preprocessing (rescaling/normalization) is built INTO the
      model itself (see train_model.py's Rescaling layer, or
      mobilenet_v2.preprocess_input for the transfer-learning path). This
      means predict.py just needs to hand over a resized image — the model
      handles the rest. Same principle as the churn project's Pipeline:
      keep training and inference preprocessing in sync by sharing it.

    Accepts:
        - a file path (str)
        - a PIL.Image
        - a numpy array
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)  # add batch dimension -> (1, H, W, 3)

    return array


def predict_image(image, model=None) -> dict:
    """
    Predict the disease class for a single leaf image.

    Args:
        image: file path, PIL Image, or numpy array
        model: optional pre-loaded model (avoids reloading on every call)

    Returns:
        dict with:
          - predicted_class: str
          - confidence: float (0-1), the probability of the predicted class
          - all_probabilities: dict mapping each class name to its probability
    """
    if model is None:
        model = load_model()

    array = preprocess_image(image)
    probabilities = model.predict(array, verbose=0)[0]  # shape: (num_classes,)

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    all_probabilities = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": {k: round(v, 4) for k, v in all_probabilities.items()},
    }


if __name__ == "__main__":
    # Quick manual test on a sample image from the dataset
    sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "Rust", "rust_000.jpg")
    result = predict_image(sample_path)
    print(f"Image: {sample_path}")
    print("Prediction:", result)

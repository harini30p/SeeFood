"""Inference module for the SeeFood (Hotdog vs Not Hotdog) application.

Loads the pre-trained Keras classification model on module startup and provides
the `predict_image` function to classify Pillow images.
"""

from pathlib import Path
from typing import Any, Dict
from PIL import Image
import tensorflow as tf

# Handle imports whether running from the project root or directly inside the backend directory
try:
    from backend.preprocessing import preprocess_image
except ImportError:
    from preprocessing import preprocess_image

# ---------------------------------------------------------------------------
# Model Path Resolution
# ---------------------------------------------------------------------------
# Path(__file__).resolve() gives the absolute path to backend/inference.py.
# .parent gives 'backend/', and .parent.parent gives the project root directory.
# This ensures that no matter where python/uvicorn is executed from, the model
# path consistently resolves to <PROJECT_ROOT>/models/seefood_best.keras.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "seefood_best.keras"

# Check if model exists and raise a clear error if missing
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at expected location: '{MODEL_PATH}'. "
        "Please make sure 'seefood_best.keras' is saved in the 'models/' folder at the project root."
    )

# ---------------------------------------------------------------------------
# Model Loading (Loaded once at module initialization)
# ---------------------------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image: Image.Image) -> Dict[str, Any]:
    """Run Hotdog vs Not Hotdog classification on a Pillow image.

    Args:
        image (Image.Image): Input image loaded with Pillow.

    Returns:
        Dict[str, Any]: Prediction results containing:
            - 'label' (str): "Hotdog" or "Not Hotdog"
            - 'hotdog_probability' (float): Sigmoid probability of being a hotdog (0.0 to 1.0)
            - 'confidence' (float): Confidence percentage for the assigned label (0.0 to 100.0)
    """
    # 1. Preprocess the Pillow image into a batch of shape (1, 224, 224, 3)
    preprocessed_batch = preprocess_image(image)

    # 2. Run prediction silently (verbose=0 suppresses TensorFlow progress bar)
    prediction = model.predict(preprocessed_batch, verbose=0)

    # 3. Extract the single scalar sigmoid probability output
    # Since the output shape is (1, 1), extract float value
    hotdog_prob = float(prediction[0][0])

    # 4. Determine label and confidence based on sigmoid threshold (0.5)
    if hotdog_prob >= 0.5:
        label = "Hotdog"
        # Confidence is the probability of it being a hotdog
        confidence = hotdog_prob * 100.0
    else:
        label = "Not Hotdog"
        # Confidence is the probability of it NOT being a hotdog
        confidence = (1.0 - hotdog_prob) * 100.0

    return {
        "label": label,
        "hotdog_probability": hotdog_prob,
        "confidence": confidence,
    }

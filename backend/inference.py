"""Inference module for the SeeFood (Hotdog vs Not Hotdog) application.

Loads the pre-trained Keras classification model on module startup, constructs
a reusable Grad-CAM explainability model, loads provisional decision thresholds,
and provides `predict_image` to classify and explain Pillow images with uncertainty handling.
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf

logger = logging.getLogger("seefood.inference")

# Handle imports whether running from the project root or directly inside backend/
try:
    from backend.preprocessing import preprocess_image
except ImportError:
    from preprocessing import preprocess_image

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "seefood_best.keras"
CONFIG_PATH = PROJECT_ROOT / "config" / "thresholds.json"

# Check if model exists and raise a clear error if missing
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at expected location: '{MODEL_PATH}'. "
        "Please make sure 'seefood_best.keras' is saved in the 'models/' folder at the project root."
    )

# ---------------------------------------------------------------------------
# Decision Threshold Configuration (Safe loading with provisional fallback)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "threshold_low": 0.35,
    "threshold_high": 0.65,
    "status": "provisional",
    "calibrated_on_validation_set": False,
}


def load_threshold_config(config_file: Optional[Path] = None) -> Dict[str, Any]:
    """Safely load and validate decision thresholds from configuration.

    Validates:
      - File exists and is valid JSON
      - threshold_low and threshold_high are numeric
      - Both values are within [0.0, 1.0]
      - threshold_low < threshold_high

    If missing or invalid, falls back to safe provisional defaults without crashing.
    """
    path = config_file or CONFIG_PATH
    if not path.exists():
        logger.warning(
            "Threshold config file '%s' not found. Using provisional defaults: %s",
            path,
            DEFAULT_THRESHOLDS,
        )
        return DEFAULT_THRESHOLDS.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        low = data.get("threshold_low")
        high = data.get("threshold_high")

        # Validate types and range
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(f"Thresholds must be numeric, got low={low}, high={high}")

        low = float(low)
        high = float(high)

        if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0):
            raise ValueError(f"Thresholds must be in range [0, 1], got low={low}, high={high}")

        if low >= high:
            raise ValueError(f"threshold_low ({low}) must be strictly less than threshold_high ({high})")

        return {
            "threshold_low": low,
            "threshold_high": high,
            "status": str(data.get("status", "provisional")),
            "calibrated_on_validation_set": bool(data.get("calibrated_on_validation_set", False)),
        }
    except Exception as exc:
        logger.warning(
            "Failed to load or validate '%s': %s. Using provisional defaults.",
            path,
            exc,
        )
        return DEFAULT_THRESHOLDS.copy()


# Global loaded configuration
THRESHOLDS: Dict[str, Any] = load_threshold_config()


def classify_probability(
    hotdog_prob: float,
    threshold_low: float = 0.35,
    threshold_high: float = 0.65,
) -> Tuple[str, Optional[float]]:
    """Determine class label and confidence percentage using dual thresholds.

    Rules:
      - hotdog_prob >= threshold_high -> 'Hotdog', confidence = hotdog_prob * 100
      - hotdog_prob <= threshold_low  -> 'Not Hotdog', confidence = (1 - hotdog_prob) * 100
      - threshold_low < hotdog_prob < threshold_high -> 'Uncertain', confidence = None
        (Confidence is explicitly None for Uncertain to avoid misleading users with invented values).

    Args:
        hotdog_prob (float): Model sigmoid output in [0, 1].
        threshold_low (float): Lower threshold for definitive 'Not Hotdog'.
        threshold_high (float): Upper threshold for definitive 'Hotdog'.

    Returns:
        Tuple[str, Optional[float]]: (label, confidence_percentage)
    """
    if hotdog_prob >= threshold_high:
        return "Hotdog", hotdog_prob * 100.0
    elif hotdog_prob <= threshold_low:
        return "Not Hotdog", (1.0 - hotdog_prob) * 100.0
    else:
        return "Uncertain", None


# ---------------------------------------------------------------------------
# Model Loading (Loaded once at module initialization)
# ---------------------------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Grad-CAM Model Construction (Constructed once at module initialization)
# ---------------------------------------------------------------------------
def _build_gradcam_model(base_seq_model: tf.keras.Model) -> tf.keras.Model:
    """Build a connected multi-output model for computing Grad-CAM gradients."""
    backbone = base_seq_model.get_layer("mobilenetv2_1.00_224")
    target_conv_layer = backbone.get_layer("out_relu")

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_conv_layer.output, backbone.output],
        name="backbone_grad_extractor",
    )

    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    conv_outputs, backbone_out = backbone_grad_model(input_tensor)

    y = base_seq_model.get_layer("global_average_pooling2d")(backbone_out)
    y = base_seq_model.get_layer("dropout")(y, training=False)
    final_output = base_seq_model.get_layer("dense")(y)

    return tf.keras.Model(
        inputs=input_tensor,
        outputs=[conv_outputs, final_output],
        name="seefood_gradcam_model",
    )


gradcam_model = _build_gradcam_model(model)


# ---------------------------------------------------------------------------
# Grad-CAM & Overlay Helper Functions
# ---------------------------------------------------------------------------
def _apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a 2D float array in [0, 1] to a 3-channel RGB image using Jet colormap."""
    r = np.clip(1.5 - np.abs(4.0 * heatmap - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * heatmap - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * heatmap - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


def generate_gradcam_overlay(
    image: Image.Image,
    preprocessed_batch: np.ndarray,
    is_hotdog: bool,
    alpha: float = 0.45,
) -> str:
    """Generate a Grad-CAM heatmap overlay on the original image encoded as base64 JPEG.

    Args:
        image (Image.Image): Original user-provided Pillow image.
        preprocessed_batch (np.ndarray): Preprocessed batch of shape (1, 224, 224, 3).
        is_hotdog (bool): True if targeting Hotdog gradients; False for Not Hotdog gradients.
        alpha (float): Opacity blend factor for the heatmap overlay (0.0 to 1.0).

    Returns:
        str: Base64-encoded data URL ('data:image/jpeg;base64,...').
    """
    input_tensor = tf.convert_to_tensor(preprocessed_batch)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(input_tensor)
        prob = predictions[0][0]
        score = prob if is_hotdog else (1.0 - prob)

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        grads = tf.zeros_like(conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_0 = conv_outputs[0]
    heatmap_tensor = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs_0), axis=-1)
    heatmap_tensor = tf.maximum(heatmap_tensor, 0.0)

    heatmap = heatmap_tensor.numpy()
    if not np.isfinite(heatmap).all():
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)

    max_val = np.max(heatmap)
    if max_val > 0.0:
        heatmap = heatmap / max_val
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_img = Image.fromarray((heatmap * 255.0).astype(np.uint8))
    heatmap_resized = heatmap_img.resize((224, 224), resample=Image.Resampling.BICUBIC)
    heatmap_norm = np.array(heatmap_resized, dtype=np.float32) / 255.0

    colored_heatmap = _apply_jet_colormap(heatmap_norm)

    orig_rgb = image.convert("RGB").resize((224, 224))
    orig_array = np.array(orig_rgb, dtype=np.float32)

    overlay_array = (1.0 - alpha) * orig_array + alpha * colored_heatmap.astype(np.float32)
    overlay_array = np.clip(overlay_array, 0.0, 255.0).astype(np.uint8)

    overlay_pil = Image.fromarray(overlay_array)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# Public Prediction API
# ---------------------------------------------------------------------------
def predict_image(image: Image.Image) -> Dict[str, Any]:
    """Run classification and Grad-CAM explainability with uncertainty handling.

    Args:
        image (Image.Image): Input image loaded with Pillow.

    Returns:
        Dict[str, Any]: Prediction results containing:
            - 'label' (str): "Hotdog", "Not Hotdog", or "Uncertain"
            - 'hotdog_probability' (float): Sigmoid probability of being a hotdog (0.0 to 1.0)
            - 'confidence' (Optional[float]): Confidence percentage (0.0 to 100.0) for decisive
              labels, or None when label is "Uncertain".
            - 'gradcam_base64' (str): JPEG Data URL of the Grad-CAM heatmap overlay
    """
    # 1. Preprocess the Pillow image into a batch of shape (1, 224, 224, 3)
    preprocessed_batch = preprocess_image(image)

    # 2. Run prediction silently (verbose=0 suppresses TensorFlow progress bar)
    prediction = model.predict(preprocessed_batch, verbose=0)

    # 3. Extract the single scalar sigmoid probability output
    hotdog_prob = float(prediction[0][0])

    # 4. Determine label and confidence based on provisional dual thresholds
    label, confidence = classify_probability(
        hotdog_prob,
        threshold_low=THRESHOLDS["threshold_low"],
        threshold_high=THRESHOLDS["threshold_high"],
    )

    # 5. Determine Grad-CAM target direction:
    # - If label is "Hotdog" -> target hotdog gradients.
    # - If label is "Not Hotdog" -> target not-hotdog gradients.
    # - If label is "Uncertain" -> target whichever binary class has the higher probability
    #   (hotdog_prob >= 0.5) using the existing Grad-CAM pipeline, so the user can see what
    #   features the model was primarily attending to during its hesitation.
    if label == "Hotdog":
        target_hotdog = True
    elif label == "Not Hotdog":
        target_hotdog = False
    else:
        target_hotdog = hotdog_prob >= 0.5

    gradcam_base64 = generate_gradcam_overlay(
        image=image,
        preprocessed_batch=preprocessed_batch,
        is_hotdog=target_hotdog,
    )

    return {
        "label": label,
        "hotdog_probability": hotdog_prob,
        "confidence": confidence,
        "gradcam_base64": gradcam_base64,
    }

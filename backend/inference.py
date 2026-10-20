"""Inference module for the SeeFood (Hotdog vs Not Hotdog) application.

Loads the pre-trained Keras classification model on module startup, constructs
a reusable Grad-CAM explainability model, and provides `predict_image` to classify
and explain Pillow images.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image
import tensorflow as tf

# Handle imports whether running from the project root or directly inside backend/
try:
    from backend.preprocessing import preprocess_image
except ImportError:
    from preprocessing import preprocess_image

# ---------------------------------------------------------------------------
# Model Path Resolution
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Grad-CAM Model Construction (Constructed once at module initialization)
# ---------------------------------------------------------------------------
# The outer model is a Keras Sequential model:
#   [0] mobilenetv2_1.00_224 (Functional backbone)
#   [1] global_average_pooling2d
#   [2] dropout
#   [3] dense (Sigmoid)
#
# Because the backbone is a nested Functional sub-model, we create a sub-graph
# that outputs both the target feature map ('out_relu', shape: 7x7x1280) and the
# final prediction probability, allowing tf.GradientTape to compute gradients.
def _build_gradcam_model(base_seq_model: tf.keras.Model) -> tf.keras.Model:
    """Build a connected multi-output model for computing Grad-CAM gradients."""
    backbone = base_seq_model.get_layer("mobilenetv2_1.00_224")
    target_conv_layer = backbone.get_layer("out_relu")

    # Intermediate model that exposes both the target conv feature map and backbone output
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_conv_layer.output, backbone.output],
        name="backbone_grad_extractor",
    )

    # Reconstruct top classification head on top of the backbone output
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
    """Map a 2D float array in [0, 1] to a 3-channel RGB image using Jet colormap.

    Implements the standard Jet color mapping using pure NumPy arithmetic.

    Args:
        heatmap (np.ndarray): 2D float array with values normalized to [0, 1].

    Returns:
        np.ndarray: uint8 RGB array of shape (H, W, 3).
    """
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
        is_hotdog (bool): True if predicted class is Hotdog, False if Not Hotdog.
        alpha (float): Opacity blend factor for the heatmap overlay (0.0 to 1.0).

    Returns:
        str: Base64-encoded data URL ('data:image/jpeg;base64,...').
    """
    input_tensor = tf.convert_to_tensor(preprocessed_batch)

    with tf.GradientTape() as tape:
        # Forward pass through the unified Grad-CAM model
        conv_outputs, predictions = gradcam_model(input_tensor)
        prob = predictions[0][0]

        # For Hotdog, maximize prob; for Not Hotdog, maximize (1.0 - prob)
        score = prob if is_hotdog else (1.0 - prob)

    # Compute gradients of the target class score with respect to the conv layer outputs
    grads = tape.gradient(score, conv_outputs)

    # Safety check: if gradients are None (e.g., zero activation), handle gracefully
    if grads is None:
        grads = tf.zeros_like(conv_outputs)

    # Global average pooling of gradients across spatial dimensions (height and width)
    # Shape becomes (1280,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each channel of the 7x7 conv output by its corresponding gradient importance
    conv_outputs_0 = conv_outputs[0]  # shape (7, 7, 1280)
    heatmap_tensor = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs_0), axis=-1)

    # Apply ReLU: only positive contributions matter to the class
    heatmap_tensor = tf.maximum(heatmap_tensor, 0.0)

    # Convert to NumPy array
    heatmap = heatmap_tensor.numpy()

    # Safely normalize the heatmap to [0, 1]
    # Check for NaN, Inf, or zero-variance maps
    if not np.isfinite(heatmap).all():
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)

    max_val = np.max(heatmap)
    if max_val > 0.0:
        heatmap = heatmap / max_val
    else:
        heatmap = np.zeros_like(heatmap)

    # Resize heatmap from 7x7 to 224x224 using Pillow's bicubic interpolation
    heatmap_img = Image.fromarray((heatmap * 255.0).astype(np.uint8))
    heatmap_resized = heatmap_img.resize((224, 224), resample=Image.Resampling.BICUBIC)
    heatmap_norm = np.array(heatmap_resized, dtype=np.float32) / 255.0

    # Colorize the heatmap using the Jet colormap (uint8 RGB)
    colored_heatmap = _apply_jet_colormap(heatmap_norm)

    # Prepare original image resized to 224x224 in RGB format
    orig_rgb = image.convert("RGB").resize((224, 224))
    orig_array = np.array(orig_rgb, dtype=np.float32)

    # Blend original image and colored heatmap:
    # blended = (1 - alpha) * original + alpha * colored_heatmap
    overlay_array = (1.0 - alpha) * orig_array + alpha * colored_heatmap.astype(np.float32)
    overlay_array = np.clip(overlay_array, 0.0, 255.0).astype(np.uint8)

    # Convert overlay array back to Pillow Image and encode as JPEG base64 Data URL
    overlay_pil = Image.fromarray(overlay_array)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# Public Prediction API
# ---------------------------------------------------------------------------
def predict_image(image: Image.Image) -> Dict[str, Any]:
    """Run Hotdog vs Not Hotdog classification and Grad-CAM explainability.

    Args:
        image (Image.Image): Input image loaded with Pillow.

    Returns:
        Dict[str, Any]: Prediction results containing:
            - 'label' (str): "Hotdog" or "Not Hotdog"
            - 'hotdog_probability' (float): Sigmoid probability of being a hotdog (0.0 to 1.0)
            - 'confidence' (float): Confidence percentage for the assigned label (0.0 to 100.0)
            - 'gradcam_base64' (str): JPEG Data URL of the Grad-CAM heatmap overlay
    """
    # 1. Preprocess the Pillow image into a batch of shape (1, 224, 224, 3)
    preprocessed_batch = preprocess_image(image)

    # 2. Run prediction silently (verbose=0 suppresses TensorFlow progress bar)
    prediction = model.predict(preprocessed_batch, verbose=0)

    # 3. Extract the single scalar sigmoid probability output
    hotdog_prob = float(prediction[0][0])

    # 4. Determine label and confidence based on sigmoid threshold (0.5)
    if hotdog_prob >= 0.5:
        label = "Hotdog"
        confidence = hotdog_prob * 100.0
        is_hotdog = True
    else:
        label = "Not Hotdog"
        confidence = (1.0 - hotdog_prob) * 100.0
        is_hotdog = False

    # 5. Generate Grad-CAM heatmap overlay for the predicted class
    gradcam_base64 = generate_gradcam_overlay(
        image=image,
        preprocessed_batch=preprocessed_batch,
        is_hotdog=is_hotdog,
    )

    return {
        "label": label,
        "hotdog_probability": hotdog_prob,
        "confidence": confidence,
        "gradcam_base64": gradcam_base64,
    }

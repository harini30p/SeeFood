"""Image preprocessing utilities for the SeeFood (Hotdog vs Not Hotdog) application.

This module provides functions to prepare raw images for inference using a
MobileNetV2 neural network.
"""

import numpy as np
import tensorflow as tf
from PIL import Image

# Target image dimensions required by MobileNetV2 (height, width)
IMG_SIZE = (224, 224)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess a Pillow image for MobileNetV2 model inference.

    Steps performed:
        1. Convert image to RGB (handles RGBA, grayscale, etc.).
        2. Resize to the target dimensions (224x224).
        3. Convert the PIL Image to a float32 NumPy array.
        4. Normalize pixel values using MobileNetV2's preprocess_input (scales to [-1, 1]).
        5. Expand dimensions to add a batch size of 1.

    Args:
        image (PIL.Image.Image): The input image loaded via Pillow.

    Returns:
        np.ndarray: Preprocessed image batch with shape (1, 224, 224, 3) and dtype float32.
    """
    # Step 1: Ensure the image is in RGB format (removes alpha channel if present or expands grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Step 2: Resize the image to 224x224 pixels
    image = image.resize(IMG_SIZE)

    # Step 3: Convert the PIL image to a NumPy array with float32 data type
    img_array = np.array(image, dtype=np.float32)

    # Step 4: Scale pixel values using MobileNetV2 specific preprocessing (ranges from -1 to 1)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Step 5: Add a batch dimension at axis 0 -> shape becomes (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch

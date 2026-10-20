"""FastAPI backend server for SeeFood (Hotdog vs Not Hotdog) application.

Exposes REST API endpoints for server status, health check, and image prediction.
"""

import io
import logging
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# Flexible import: works when running from project root (uvicorn backend.main:app)
# or directly from within the backend directory.
try:
    from backend.inference import model, predict_image
except ImportError:
    from inference import model, predict_image

# Configure logging
logger = logging.getLogger("seefood.api")
logging.basicConfig(level=logging.INFO)

# Supported image formats (identified by Pillow)
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

# Initialize FastAPI application
app = FastAPI(
    title="SeeFood API",
    description="A beginner-friendly Hotdog vs Not Hotdog image classification API powered by MobileNetV2 and FastAPI.",
    version="1.0.0",
)

# Configure CORS Middleware for frontend client integration (e.g. Vite React on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get(
    "/",
    summary="Root endpoint",
    description="Returns a welcome message and documentation links.",
    tags=["General"],
)
def read_root() -> Dict[str, str]:
    """Provide a welcome message and links to interactive API documentation."""
    return {
        "message": "Welcome to the SeeFood API! Ready to classify Hotdogs.",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get(
    "/health",
    summary="Health check endpoint",
    description="Checks service health and confirms if the deep learning model is loaded.",
    tags=["System"],
)
def health_check() -> Dict[str, Any]:
    """Verify backend and model status."""
    is_model_loaded = model is not None and hasattr(model, "predict")
    return {
        "status": "healthy" if is_model_loaded else "degraded",
        "model_loaded": is_model_loaded,
    }


@app.post(
    "/predict",
    summary="Predict hotdog or not hotdog",
    description="Upload an image (JPEG, PNG, or WebP) to classify whether it contains a hotdog.",
    tags=["Inference"],
)
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
) -> Dict[str, Any]:
    """Process uploaded image and return classification results with Grad-CAM explainability.

    Validates file content using Pillow, executes MobileNetV2 inference,
    and returns predicted label, confidence, hotdog probability, and
    the Grad-CAM heatmap base64 data URL.
    """
    # 1. Read binary content from the uploaded file
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read the uploaded file.",
        ) from exc

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # 2. Inspect and validate image using Pillow
    try:
        # Wrap bytes in BytesIO
        image_stream = io.BytesIO(contents)
        image = Image.open(image_stream)

        # image.verify() verifies header integrity; then re-open for decoding
        image.verify()
        image_stream.seek(0)
        image = Image.open(image_stream)
        image.load()  # Force load pixel data to ensure file is not corrupted
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Invalid or corrupt image uploaded: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or readable image.",
        ) from exc

    # 3. Check for supported image format
    image_format = (image.format or "").upper()
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image format '{image_format}'. "
                f"Supported formats are: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}."
            ),
        )

    # 4. Perform model inference
    try:
        result = predict_image(image)
        return result
    except Exception as exc:
        # Log error internally without exposing raw stack trace to caller
        logger.error("Prediction error encountered: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the image for prediction.",
        ) from exc

"""Pytest fixtures for SeeFood automated testing.

Provides a shared FastAPI TestClient and helper fixtures for generating
synthetic in-memory images in various formats without disk I/O.
"""

import io
from typing import Callable
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def create_image_bytes() -> Callable[..., bytes]:
    """Factory fixture to create in-memory image bytes of specified format and size.

    Args:
        fmt (str): Image format ('JPEG', 'PNG', 'WEBP', 'GIF', etc.).
        size (tuple): Image (width, height) in pixels.
        color (str | tuple): Image fill color.

    Returns:
        Callable that produces bytes for the specified image.
    """
    def _create(fmt: str = "JPEG", size: tuple = (224, 224), color: str = "red") -> bytes:
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    return _create

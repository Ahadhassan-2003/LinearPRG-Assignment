"""Shared pytest fixtures for the image translation pipeline test suite."""

from __future__ import annotations

import io

import pytest
from PIL import Image


@pytest.fixture()
def small_png_bytes() -> bytes:
    """Return raw bytes for a 50×50 white PNG image.

    This fixture is used across multiple test modules to avoid duplicating
    image-creation boilerplate and to ensure no filesystem access is required.

    Returns:
        Raw PNG bytes of a solid white 50×50 image.
    """
    img = Image.new("RGB", (50, 50), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

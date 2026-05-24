"""Unit tests for app/utils/image_utils.py."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app.utils.image_utils import (
    hex_to_rgb,
    image_to_base64,
    image_to_bytes,
    load_image_from_bytes,
    validate_image_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 100, height: int = 80, color: str = "RGB") -> bytes:
    """Create a minimal in-memory PNG and return its raw bytes."""
    img = Image.new(color, (width, height), (45, 80, 22))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# load_image_from_bytes
# ---------------------------------------------------------------------------


def test_load_image_from_bytes_valid() -> None:
    """Round-trip: create a PIL image, serialise to bytes, reload and verify."""
    raw = _make_png_bytes(width=120, height=90)
    image, fmt, width, height = load_image_from_bytes(raw)

    assert isinstance(image, Image.Image)
    assert fmt == "PNG"
    assert width == 120
    assert height == 90


def test_load_image_from_bytes_invalid() -> None:
    """Random bytes that are not a valid image must raise ValueError."""
    junk = b"\x00\x01\x02\x03this is not an image"
    with pytest.raises(ValueError, match="Could not decode image bytes"):
        load_image_from_bytes(junk)


# ---------------------------------------------------------------------------
# hex_to_rgb
# ---------------------------------------------------------------------------


def test_hex_to_rgb() -> None:
    """Known hex values must map to their expected RGB tuples."""
    assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert hex_to_rgb("#000000") == (0, 0, 0)
    assert hex_to_rgb("#2D5016") == (45, 80, 22)


def test_hex_to_rgb_lowercase() -> None:
    """Lowercase hex digits must also be accepted."""
    assert hex_to_rgb("#ffffff") == (255, 255, 255)
    assert hex_to_rgb("#2d5016") == (45, 80, 22)


def test_hex_to_rgb_invalid_raises() -> None:
    """Malformed strings must raise ValueError."""
    with pytest.raises(ValueError):
        hex_to_rgb("FFFFFF")  # missing '#'
    with pytest.raises(ValueError):
        hex_to_rgb("#FFF")  # too short
    with pytest.raises(ValueError):
        hex_to_rgb("#GGGGGG")  # non-hex chars


# ---------------------------------------------------------------------------
# image_to_base64 round-trip
# ---------------------------------------------------------------------------


def test_image_to_base64_roundtrip() -> None:
    """Encoding then decoding must reproduce the original bytes exactly."""
    original_bytes = _make_png_bytes()
    b64_string = image_to_base64(original_bytes)

    # Must be a plain string with no data-URI prefix
    assert isinstance(b64_string, str)
    assert not b64_string.startswith("data:")

    # Decoding must reproduce the exact original bytes
    recovered = base64.b64decode(b64_string)
    assert recovered == original_bytes


# ---------------------------------------------------------------------------
# validate_image_size
# ---------------------------------------------------------------------------


def test_validate_image_size_passes() -> None:
    """A small, valid image must not raise any exception."""
    small_bytes = _make_png_bytes(width=10, height=10)
    # Should complete without error
    validate_image_size(small_bytes, max_mb=10)


def test_validate_image_size_fails() -> None:
    """Bytes exceeding the limit must raise ValueError with a clear message."""
    # 11 MB of zeros — well above the 10 MB default limit
    oversized = b"\x00" * (11 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
        validate_image_size(oversized, max_mb=10)

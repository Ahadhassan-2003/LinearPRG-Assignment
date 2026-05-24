"""Image utility functions for the image translation pipeline.

All functions use Pillow (PIL) exclusively. No external dependencies are
introduced beyond what is already declared in pyproject.toml.
"""

from __future__ import annotations

import base64
import io
from collections import Counter

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_image_from_bytes(
    image_bytes: bytes,
) -> tuple[Image.Image, str, int, int]:
    """Open a PIL Image from raw bytes and return key metadata.

    Args:
        image_bytes: Raw bytes of the image file (e.g. read from an upload).

    Returns:
        A 4-tuple ``(pil_image, format_string, width, height)`` where
        ``format_string`` is Pillow's uppercase format identifier such as
        ``"PNG"`` or ``"JPEG"``.

    Raises:
        ValueError: If ``image_bytes`` cannot be decoded as a valid image.
    """
    try:
        buf = io.BytesIO(image_bytes)
        image = Image.open(buf)
        image.load()  # force full decode so we catch corrupt data early
        fmt = image.format or "PNG"
        width, height = image.size
        return image, fmt, width, height
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Serialise a PIL Image to raw bytes in the specified format.

    Args:
        image: A PIL ``Image`` object to serialise.
        format: Pillow format string (e.g. ``"PNG"``, ``"JPEG"``).
            Defaults to ``"PNG"``.

    Returns:
        Raw bytes of the encoded image.
    """
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Base64 / data-URI helpers
# ---------------------------------------------------------------------------


def image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes as a plain base64 string.

    Args:
        image_bytes: Raw bytes of any image file.

    Returns:
        A base64-encoded string with **no** data-URI prefix.
    """
    return base64.b64encode(image_bytes).decode("ascii")


def image_bytes_to_base64_uri(
    image_bytes: bytes,
    mime_type: str = "image/png",
) -> str:
    """Build a RFC-2397 data URI from raw image bytes.

    The resulting string is suitable for embedding directly in a vision-model
    API request or in an HTML ``<img>`` ``src`` attribute.

    Args:
        image_bytes: Raw bytes of the image to encode.
        mime_type: MIME type of the image, e.g. ``"image/png"`` or
            ``"image/jpeg"``. Defaults to ``"image/png"``.

    Returns:
        A data URI of the form ``"data:<mime_type>;base64,<data>"``.
    """
    b64 = image_to_base64(image_bytes)
    return f"data:{mime_type};base64,{b64}"


# ---------------------------------------------------------------------------
# Color sampling
# ---------------------------------------------------------------------------


def sample_background_color(
    image: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    sample_padding: int = 5,
) -> str:
    """Estimate the dominant background colour around a bounding box.

    A border strip of ``sample_padding`` pixels is taken *outside* the supplied
    bounding box (clamped to the image canvas). The most frequently occurring
    quantised RGB colour in that strip is returned.

    Args:
        image: Source PIL ``Image`` from which to sample.
        x: Left edge of the text bounding box in pixels.
        y: Top edge of the text bounding box in pixels.
        w: Width of the text bounding box in pixels.
        h: Height of the text bounding box in pixels.
        sample_padding: Number of pixels to expand outward from the bbox
            when defining the border sampling region. Defaults to ``5``.

    Returns:
        A hex colour string such as ``"#2D5016"`` representing the dominant
        colour in the sampled border. Returns ``"#FFFFFF"`` if sampling fails
        for any reason (e.g. the box is entirely outside the image canvas).
    """
    try:
        img_w, img_h = image.size
        rgb_image = image.convert("RGB")

        # Outer rectangle (clamped to canvas)
        outer_x1 = max(0, x - sample_padding)
        outer_y1 = max(0, y - sample_padding)
        outer_x2 = min(img_w, x + w + sample_padding)
        outer_y2 = min(img_h, y + h + sample_padding)

        # Inner rectangle (the actual bbox, clamped)
        inner_x1 = max(0, x)
        inner_y1 = max(0, y)
        inner_x2 = min(img_w, x + w)
        inner_y2 = min(img_h, y + h)

        if outer_x2 <= outer_x1 or outer_y2 <= outer_y1:
            return "#FFFFFF"

        pixels: list[tuple[int, int, int]] = []

        for py in range(outer_y1, outer_y2):
            for px in range(outer_x1, outer_x2):
                # Skip pixels that are inside the inner (text) box
                if inner_x1 <= px < inner_x2 and inner_y1 <= py < inner_y2:
                    continue
                pixels.append(rgb_image.getpixel((px, py)))  # type: ignore[arg-type]

        if not pixels:
            return "#FFFFFF"

        # Quantise to multiples of 8 to group near-identical colours together
        quantised = [
            (r & 0xF8, g & 0xF8, b & 0xF8) for r, g, b in pixels
        ]
        dominant = Counter(quantised).most_common(1)[0][0]
        return "#{:02X}{:02X}{:02X}".format(*dominant)

    except Exception:
        return "#FFFFFF"


# ---------------------------------------------------------------------------
# Color conversion
# ---------------------------------------------------------------------------


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a CSS hex colour string to an RGB integer triple.

    Args:
        hex_color: A string of the form ``"#RRGGBB"`` (case-insensitive).

    Returns:
        A tuple ``(R, G, B)`` where each component is an integer in ``[0, 255]``.

    Raises:
        ValueError: If ``hex_color`` is not a valid ``#RRGGBB`` string.
    """
    hex_color = hex_color.strip()
    if not hex_color.startswith("#") or len(hex_color) != 7:
        raise ValueError(
            f"hex_color must be in the form '#RRGGBB', got {hex_color!r}"
        )
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except ValueError as exc:
        raise ValueError(
            f"hex_color contains non-hex characters: {hex_color!r}"
        ) from exc
    return r, g, b


# ---------------------------------------------------------------------------
# Font sizing
# ---------------------------------------------------------------------------


def calculate_font_size(
    text: str,
    bbox_width_px: int,
    bbox_height_px: int,
    is_bold: bool,
    font_size_relative: str,
    min_size: int = 8,
    max_size: int = 72,
) -> int:
    """Estimate an integer font size that fits ``text`` inside a bounding box.

    The function starts from a seed size derived from ``font_size_relative``
    (``"large"`` → 36, ``"medium"`` → 24, ``"small"`` → 14), then iteratively
    reduces the size until the rendered text width fits within 95 % of
    ``bbox_width_px``. Because PIL's built-in default font is bitmap-only and
    not resizable, :func:`ImageFont.load_default` is used purely for the
    character-width heuristic; the returned integer is still meaningful as a
    point/pixel size hint for downstream rendering with a scalable font.

    Args:
        text: The string whose rendered width is being estimated.
        bbox_width_px: Available width in pixels.
        bbox_height_px: Available height in pixels (currently used only to
            bound the starting size).
        is_bold: Whether bold weight should influence the size estimate.
            Bold text is assumed to be ~10 % wider per character.
        font_size_relative: One of ``"large"``, ``"medium"``, or ``"small"``.
            Controls the starting point for the search.
        min_size: Minimum font size to return. Defaults to ``8``.
        max_size: Maximum font size to consider. Defaults to ``72``.

    Returns:
        The largest integer font size (≥ ``min_size``) at which the text is
        estimated to fit within ``bbox_width_px * 0.95`` pixels of width.
    """
    _SEED: dict[str, int] = {"large": 36, "medium": 24, "small": 14}
    seed = _SEED.get(font_size_relative, 24)

    # Clamp seed to [min_size, max_size] and also to bbox height
    size = min(seed, max_size, max(bbox_height_px, min_size))
    size = max(size, min_size)

    # Bold heuristic: assume glyphs are ~10 % wider
    bold_factor = 1.10 if is_bold else 1.0

    # Use the default PIL font to obtain an approximate character width.
    # load_default() accepts a 'size' kwarg in Pillow >= 10; fall back
    # gracefully for older versions.
    def _measure(candidate_size: int) -> float:
        try:
            font = ImageFont.load_default(size=candidate_size)
        except TypeError:
            font = ImageFont.load_default()

        # getbbox is available on FreeType fonts; for bitmap fonts fall back
        # to a fixed-width estimate.
        try:
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            # Bitmap default font: roughly 6 px per character
            text_width = len(text) * 6

        return text_width * bold_factor

    target_width = bbox_width_px * 0.95

    # Reduce size until it fits or we hit min_size
    while size > min_size and _measure(size) > target_width:
        size -= 1

    return size


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_image_size(image_bytes: bytes, max_mb: int = 10) -> None:
    """Assert that image data does not exceed a maximum file size.

    Args:
        image_bytes: Raw bytes of the image to check.
        max_mb: Maximum permitted size in megabytes. Defaults to ``10``.

    Raises:
        ValueError: If the size of ``image_bytes`` exceeds ``max_mb`` MiB,
            with a message stating the actual and maximum sizes.
    """
    max_bytes = max_mb * 1024 * 1024
    actual_bytes = len(image_bytes)
    if actual_bytes > max_bytes:
        actual_mb = actual_bytes / (1024 * 1024)
        raise ValueError(
            f"Image size {actual_mb:.2f} MB exceeds the maximum allowed "
            f"size of {max_mb} MB."
        )

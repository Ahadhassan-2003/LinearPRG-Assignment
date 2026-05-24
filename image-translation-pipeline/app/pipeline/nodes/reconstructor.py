"""Image reconstruction node for the image translation pipeline.

This node iterates over every validated TextBlock in the pipeline state,
covers the original text region with a solid filled rectangle, then draws the
translated text on top using PIL. The result is serialised back to bytes and
stored as ``output_image_bytes`` in the pipeline state.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.models import PipelineState, TextBlock
from app.utils.image_utils import (
    calculate_font_size,
    hex_to_rgb,
    image_to_bytes,
    load_image_from_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Font discovery — ordered candidate paths
# ---------------------------------------------------------------------------

_FONT_CANDIDATES_REGULAR: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      # Linux
    "/System/Library/Fonts/Helvetica.ttc",                   # macOS
    "C:/Windows/Fonts/arial.ttf",                            # Windows (uv/POSIX path)
    "/Windows/Fonts/arial.ttf",                              # Windows (Git-bash style)
]

_FONT_CANDIDATES_BOLD: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "/System/Library/Fonts/Helvetica.ttc",                   # macOS (no separate bold)
    "C:/Windows/Fonts/arialbd.ttf",                          # Windows bold
    "/Windows/Fonts/arialbd.ttf",
]


def _load_font(size: int, is_bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempt to load a scalable TrueType font at the requested point size.

    Searches a prioritised list of system font paths.  Falls back to PIL's
    built-in bitmap font if no TrueType file is found (size parameter is
    ignored in that case).

    Args:
        size: Desired font size in points.
        is_bold: Whether to prefer a bold-weight font file.

    Returns:
        A PIL font object — either a :class:`ImageFont.FreeTypeFont` when a
        TrueType file is found, or the built-in :class:`ImageFont.ImageFont`
        otherwise.
    """
    candidates = _FONT_CANDIDATES_BOLD if is_bold else _FONT_CANDIDATES_REGULAR
    # Also try the regular variants as a further fallback when bold is missing
    if is_bold:
        candidates = candidates + _FONT_CANDIDATES_REGULAR

    for path_str in candidates:
        path = Path(path_str)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue  # corrupt or wrong format — try next

    # Final fallback: PIL built-in bitmap font
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Private drawing helpers
# ---------------------------------------------------------------------------


def _cover_original_text(
    image: Image.Image,
    bbox_px: tuple[int, int, int, int],
    background_color: str,
) -> Image.Image:
    """Paint a filled rectangle over a text region to erase the original text.

    The operation is performed on the image object passed in (expected to be a
    working copy, not the original).

    Args:
        image: PIL ``Image`` to modify in-place.
        bbox_px: ``(x, y, w, h)`` bounding box in pixels.
        background_color: Fill colour as a ``"#RRGGBB"`` hex string.

    Returns:
        The same image object after modification.
    """
    x, y, w, h = bbox_px
    rgb = hex_to_rgb(background_color)

    fill: tuple[int, ...] = rgb
    if image.mode == "RGBA":
        fill = (*rgb, 255)

    draw = ImageDraw.Draw(image)
    draw.rectangle([x, y, x + w, y + h], fill=fill)  # type: ignore[arg-type]
    return image


def _draw_translated_text(
    image: Image.Image,
    text: str,
    bbox_px: tuple[int, int, int, int],
    text_color: str,
    is_bold: bool,
    font_size_relative: str,
    text_alignment: str,
) -> Image.Image:
    """Render translated text inside a bounding box with alignment and wrapping.

    The function:
    - Calculates an appropriate font size via :func:`~app.utils.image_utils.calculate_font_size`.
    - Loads the best available system font (falls back to PIL default).
    - Wraps text lines that would exceed the box width.
    - Honours ``text_alignment`` (``"left"``, ``"center"``, ``"right"``).

    Args:
        image: PIL ``Image`` to draw on.
        text: Translated text string to render.
        bbox_px: ``(x, y, w, h)`` bounding box in pixels.
        text_color: Glyph colour as ``"#RRGGBB"``.
        is_bold: Whether to prefer a bold font weight.
        font_size_relative: One of ``"large"``, ``"medium"``, ``"small"``.
        text_alignment: One of ``"left"``, ``"center"``, ``"right"``.

    Returns:
        The same image object after the text has been drawn.
    """
    x, y, w, h = bbox_px

    if w <= 0 or h <= 0 or not text.strip():
        return image

    # --- Font size --------------------------------------------------------
    font_size = calculate_font_size(
        text=text,
        bbox_width_px=w,
        bbox_height_px=h,
        is_bold=is_bold,
        font_size_relative=font_size_relative,
    )
    font = _load_font(size=font_size, is_bold=is_bold)

    # --- Text wrapping ----------------------------------------------------
    # Estimate average character width using a representative sample string
    try:
        sample_bbox = font.getbbox("W")
        avg_char_w = max(1, sample_bbox[2] - sample_bbox[0])
    except AttributeError:
        avg_char_w = font_size  # safe fallback for bitmap fonts

    max_chars = max(1, int(w * 0.95 / avg_char_w))
    wrapped = textwrap.fill(text, width=max_chars)
    lines = wrapped.splitlines() or [text]

    # --- Line height ------------------------------------------------------
    try:
        line_bbox = font.getbbox("Ag")
        line_height = line_bbox[3] - line_bbox[1] + 2
    except AttributeError:
        line_height = font_size + 2

    # --- Vertical centering: start y so the block sits in the middle ------
    total_text_height = line_height * len(lines)
    current_y = y + max(0, (h - total_text_height) // 2)

    # --- Colour -----------------------------------------------------------
    rgb = hex_to_rgb(text_color)
    fill: tuple[int, ...] = rgb
    if image.mode == "RGBA":
        fill = (*rgb, 255)

    draw = ImageDraw.Draw(image)
    pil_align = text_alignment  # PIL accepts "left", "center", "right" directly

    for line in lines:
        if current_y + line_height > y + h:
            break  # no more room vertically

        # Determine x anchor based on alignment
        try:
            line_w = font.getlength(line)
        except AttributeError:
            try:
                line_bbox_item = font.getbbox(line)
                line_w = line_bbox_item[2] - line_bbox_item[0]
            except AttributeError:
                line_w = len(line) * avg_char_w

        if pil_align == "center":
            line_x = x + (w - line_w) // 2
        elif pil_align == "right":
            line_x = x + w - line_w
        else:  # "left" or unknown
            line_x = x

        line_x = max(x, line_x)  # never draw outside left edge

        draw.text((line_x, current_y), line, font=font, fill=fill)  # type: ignore[arg-type]
        current_y += line_height

    return image


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def reconstruct_image_node(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: paint translated text onto the source image.

    For each :class:`~app.models.TextBlock` in the pipeline state the node:

    1. Converts the percentage-based bounding box to pixel coordinates.
    2. Covers the original text with a solid fill in ``background_color``.
    3. Renders the ``translated_text`` over the fill using the block's
       typographic metadata.

    Blocks are processed from **largest to smallest bounding-box area** so
    that smaller blocks are drawn last (on top) when regions overlap.

    If ``state["error"]`` is already set the function returns immediately
    without modifying anything, acting as a transparent pass-through.

    This function is named ``reconstruct_image`` so that it appears under
    that label in LangSmith traces.

    Args:
        state: Current :class:`~app.models.PipelineState`.

    Returns:
        A partial state-update dict containing:
          - ``"output_image_bytes"`` (bytes): PNG/JPEG bytes of the output.
          - ``"error"`` (str | None): ``None`` on success, error message on
            failure.
    """
    # ------------------------------------------------------------------
    # Pass-through: if a prior node already failed, do nothing
    # ------------------------------------------------------------------
    if state.get("error"):
        return {}

    try:
        image_bytes: bytes = state["image_bytes"]
        text_blocks: list[TextBlock] = state.get("text_blocks") or []
        image_format: str = state.get("image_format") or "PNG"

        # ---------------------------------------------------------------
        # 1. Load source image and convert to RGBA for safe compositing
        # ---------------------------------------------------------------
        pil_image, fmt, img_w, img_h = load_image_from_bytes(image_bytes)
        working: Image.Image = pil_image.convert("RGBA")

        # ---------------------------------------------------------------
        # 2. If no text was found, pass the original image through
        # ---------------------------------------------------------------
        if not text_blocks:
            logger.info("No text blocks — passing original image through.")
            out_format = "JPEG" if image_format.upper() == "JPEG" else "PNG"
            if out_format == "JPEG":
                working = working.convert("RGB")
            return {
                "output_image_bytes": image_to_bytes(working, format=out_format),
                "error": None,
            }

        # ---------------------------------------------------------------
        # 3. Sort blocks: largest bbox area first
        # ---------------------------------------------------------------
        def _area(block: TextBlock) -> float:
            return block.bbox.width * block.bbox.height

        sorted_blocks = sorted(text_blocks, key=_area, reverse=True)

        # ---------------------------------------------------------------
        # 4. Process each block
        # ---------------------------------------------------------------
        for block in sorted_blocks:
            bbox_px = block.bbox.to_pixels(img_w, img_h)
            x_px, y_px, w_px, h_px = bbox_px

            # Clamp to image bounds
            x_px = max(0, min(x_px, img_w - 1))
            y_px = max(0, min(y_px, img_h - 1))
            w_px = max(1, min(w_px, img_w - x_px))
            h_px = max(1, min(h_px, img_h - y_px))
            clamped_bbox = (x_px, y_px, w_px, h_px)

            # Cover original text
            working = _cover_original_text(
                working, clamped_bbox, block.background_color
            )

            # Draw translated text
            working = _draw_translated_text(
                image=working,
                text=block.translated_text,
                bbox_px=clamped_bbox,
                text_color=block.text_color,
                is_bold=block.is_bold,
                font_size_relative=block.font_size_relative,
                text_alignment=block.text_alignment,
            )

        # ---------------------------------------------------------------
        # 5. Convert back to original colour mode and serialise
        # ---------------------------------------------------------------
        out_format = "JPEG" if image_format.upper() == "JPEG" else "PNG"
        if out_format == "JPEG":
            working = working.convert("RGB")

        output_bytes = image_to_bytes(working, format=out_format)

        logger.info(
            "Reconstruction complete: %d blocks processed, output %d bytes.",
            len(sorted_blocks),
            len(output_bytes),
        )

        return {"output_image_bytes": output_bytes, "error": None}

    except Exception as exc:  # noqa: BLE001
        logger.exception("reconstruct_image_node failed: %s", exc)
        return {"error": str(exc), "output_image_bytes": b""}


# LangSmith trace label
reconstruct_image_node.__name__ = "reconstruct_image"
reconstruct_image_node.__qualname__ = "reconstruct_image"

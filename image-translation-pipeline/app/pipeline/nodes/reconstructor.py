"""Image reconstruction node for the image translation pipeline.

**What this module does**

Takes the list of validated ``TextBlock`` objects stored in
``PipelineState["text_blocks"]`` and paints the translated text onto a copy
of the source image. For each block it:

1. Draws a solid filled rectangle in ``background_color`` over the original
   text region, erasing it cleanly.
2. Renders ``translated_text`` over the rectangle using the block's
   typographic metadata (relative font size, bold, colour, alignment).

Blocks are processed from largest to smallest bounding-box area so that
smaller blocks are drawn last and always appear on top when regions overlap.

**Inputs (read from PipelineState)**

- ``image_bytes`` *(bytes, required)* — raw bytes of the source image.
- ``image_format`` *(str)* — Pillow format string (``"PNG"``, ``"JPEG"``).
- ``image_width`` / ``image_height`` *(int)* — pixel dimensions of the image.
- ``text_blocks`` *(list[TextBlock])* — validated blocks from the extractor node.
- ``error`` *(str | None)* — if non-``None``, the node returns immediately
  without modifying anything (transparent pass-through).

**Outputs (partial state update dict)**

- ``output_image_bytes`` *(bytes)* — serialised PNG (or JPEG) of the
  reconstructed image.
- ``error`` *(str | None)* — ``None`` on success; error message on failure.

**LangSmith span name**: ``"reconstruct_image"``

The function's ``__name__`` is overridden to this value so that it appears
under a clean label in LangSmith traces.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any
import io

from PIL import Image, ImageDraw, ImageFont

from app.models import PipelineState, TextBlock
from app.utils.image_utils import (
    sample_background_color,
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
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=fill)  # type: ignore[arg-type]
    return image


def _get_text_width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> float:
    """Helper to get the pixel width of a string rendered in the given font."""
    try:
        return float(font.getlength(text))
    except AttributeError:
        try:
            bbox = font.getbbox(text)
            return float(bbox[2] - bbox[0])
        except AttributeError:
            return len(text) * 6.0  # fallback for basic bitmap font


def _wrap_text(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str, max_width: float) -> list[str]:
    """Wrap text to ensure no line exceeds max_width."""
    words = text.split()
    if not words:
        return []
    lines = []
    current_line: list[str] = []
    for word in words:
        test_line = " ".join(current_line + [word])
        if current_line and _get_text_width(font, test_line) > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def _find_best_font_and_wrap(
    text: str, w: int, h: int, is_bold: bool, font_size_relative: str
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int]:
    """Find the largest font size where the wrapped text fits within the bbox dimensions."""
    _SEED = {"large": 36, "medium": 24, "small": 14}
    size = min(_SEED.get(font_size_relative, 24), 72, max(h, 8))
    size = max(size, 8)

    target_width = w * 0.95
    target_height = h * 0.95

    while size > 8:
        font = _load_font(size, is_bold)
        lines = _wrap_text(font, text, target_width)

        try:
            line_bbox = font.getbbox("Ag")
            line_height = line_bbox[3] - line_bbox[1] + 2
        except AttributeError:
            line_height = size + 2

        total_height = line_height * len(lines)
        max_line_width = max([_get_text_width(font, l) for l in lines]) if lines else 0

        if max_line_width <= target_width and total_height <= target_height:
            return font, lines, line_height

        size -= 1

    # Fallback to minimum size
    font = _load_font(8, is_bold)
    lines = _wrap_text(font, text, target_width)
    try:
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 2
    except AttributeError:
        line_height = 10
    return font, lines, line_height


def _refine_bbox_height(
    image: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    bg_color_hex: str,
) -> int:
    """Adjust the height of the bounding box to snap to background boundaries.
    
    Shrinks the box from the bottom if it overfills into a different colored
    region, and expands the box downwards to cover text descenders (like 'p')
    if the region below shares the same background color.
    """
    rgb_image = image.convert("RGB")
    br, bg, bb = hex_to_rgb(bg_color_hex)

    def row_matches_bg(row_y: int) -> bool:
        if row_y < 0 or row_y >= rgb_image.height:
            return False
        matches = 0
        total = 0
        # Check a sample of pixels across the row
        for px in range(x, x + w, max(1, w // 20)):
            if px >= rgb_image.width:
                break
            pr, pg, pb = rgb_image.getpixel((px, row_y))  # type: ignore[arg-type]
            # Manhattan distance in RGB space
            dist = abs(pr - br) + abs(pg - bg) + abs(pb - bb)
            if dist < 80:  # Tolerance for compression artifacts and noise
                matches += 1
            total += 1
        if total == 0:
            return False
        return matches / total >= 0.5

    # 1. Shrink from bottom if overfilling into a different background
    while h > 10 and not row_matches_bg(y + h - 1):
        h -= 1

    # 2. Expand bottom to cover descenders (e.g. 'p', 'y', 'g')
    expanded = 0
    max_expand = max(10, int(h * 0.4))
    while expanded < max_expand and row_matches_bg(y + h):
        h += 1
        expanded += 1

    return h


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

    # --- Font sizing and wrapping -----------------------------------------
    font, lines, line_height = _find_best_font_and_wrap(
        text=text,
        w=w,
        h=h,
        is_bold=is_bold,
        font_size_relative=font_size_relative,
    )

    # --- Vertical centering: start y so the block sits in the middle ------
    total_text_height = line_height * len(lines)
    # add a slight vertical offset adjustment to account for font ascenders
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

        line_w = _get_text_width(font, line)

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
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_w = state.get("image_width")
        img_h = state.get("image_height")
        if img_w is None or img_h is None:
            img_w, img_h = pil_image.size
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

            # Sample the actual background color from the edges
            sampled_bg = sample_background_color(pil_image, x_px, y_px, w_px, h_px)

            # Refine the bounding box height to fix overfilling and descender clipping
            h_px = _refine_bbox_height(pil_image, x_px, y_px, w_px, h_px, sampled_bg)
            h_px = max(1, min(h_px, img_h - y_px))
            clamped_bbox = (x_px, y_px, w_px, h_px)

            # Cover original text
            working = _cover_original_text(
                working, clamped_bbox, sampled_bg
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

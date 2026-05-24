"""Pydantic models and LangGraph state definitions for the image translation pipeline.

This module contains all data models used across the API layer and the
LangGraph pipeline. No business logic is implemented here — only model
definitions, validators, and pure data-conversion helpers.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import NotRequired, TypedDict


# ---------------------------------------------------------------------------
# Primitive geometry model
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box expressed as percentages of image dimensions.

    All coordinates are in the range [0, 100], where (0, 0) is the top-left
    corner of the image and (100, 100) is the bottom-right corner.

    Attributes:
        x: Left edge of the box as a percentage of the image width (0–100).
        y: Top edge of the box as a percentage of the image height (0–100).
        width: Box width as a percentage of the image width (0–100).
        height: Box height as a percentage of the image height (0–100).
    """

    x: float
    y: float
    width: float
    height: float

    @model_validator(mode="after")
    def _all_values_in_range(self) -> "BoundingBox":
        """Validate that every coordinate component is between 0 and 100.

        Returns:
            The validated BoundingBox instance.

        Raises:
            ValueError: If any field value falls outside the [0, 100] range.
        """
        for field_name, value in {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }.items():
            if not (0.0 <= value <= 100.0):
                raise ValueError(
                    f"BoundingBox.{field_name} must be between 0 and 100, got {value}"
                )
        return self

    def to_pixels(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Convert percentage-based coordinates to absolute pixel values.

        Args:
            img_width: The total width of the image in pixels.
            img_height: The total height of the image in pixels.

        Returns:
            A tuple ``(x_px, y_px, w_px, h_px)`` where each element is the
            corresponding dimension expressed in whole pixels.
        """
        x_px = int(self.x / 100.0 * img_width)
        y_px = int(self.y / 100.0 * img_height)
        w_px = int(self.width / 100.0 * img_width)
        h_px = int(self.height / 100.0 * img_height)
        return x_px, y_px, w_px, h_px


# ---------------------------------------------------------------------------
# Text block model
# ---------------------------------------------------------------------------

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


class TextBlock(BaseModel):
    """A detected region of text in an image together with its translation.

    Attributes:
        original_text: The text as it appears in the source image.
        translated_text: The text after translation.
        bbox: Bounding box that locates this block within the image.
        font_size_relative: Relative size classification of the text.
        is_bold: Whether the text appears to be rendered in bold weight.
        text_color: Foreground (glyph) colour expressed as a CSS hex string,
            e.g. ``"#FFFFFF"``.
        background_color: Background fill colour expressed as a CSS hex
            string, e.g. ``"#2D5016"``.
        text_alignment: Horizontal alignment of the text within its block.
    """

    original_text: str
    translated_text: str
    bbox: BoundingBox
    font_size_relative: Literal["large", "medium", "small"]
    is_bold: bool
    text_color: str
    background_color: str
    text_alignment: Literal["left", "center", "right"]

    @field_validator("text_color", "background_color", mode="before")
    @classmethod
    def _normalise_hex_color(cls, value: str) -> str:
        """Normalise and validate a CSS hex colour string.

        If the value does not start with ``#``, the character is prepended
        automatically. The resulting string must match the pattern
        ``#[0-9A-Fa-f]{6}``.

        Args:
            value: Raw colour string supplied by the caller.

        Returns:
            A normalised hex colour string starting with ``#``.

        Raises:
            ValueError: If the string does not form a valid 6-digit hex colour
                after optional ``#`` prepending.
        """
        if not value.startswith("#"):
            value = f"#{value}"
        if not _HEX_COLOR_RE.match(value):
            raise ValueError(
                f"Color must match the pattern #[0-9A-Fa-f]{{6}}, got {value!r}"
            )
        return value


# ---------------------------------------------------------------------------
# LangGraph pipeline state (TypedDict)
# ---------------------------------------------------------------------------


class _PipelineStateRequired(TypedDict):
    image_bytes: bytes

class PipelineState(_PipelineStateRequired, total=False):
    """Mutable state object threaded through every node of the LangGraph pipeline.

    ``image_bytes`` is the only mandatory key; all other keys default to
    ``None`` or their absent state until populated by a pipeline node.

    Attributes:
        image_bytes: Raw binary content of the input image. Required.
        image_width: Width of the decoded input image in pixels.
        image_height: Height of the decoded input image in pixels.
        image_format: Pillow format string of the input image (e.g.
            ``"PNG"``, ``"JPEG"``).
        detected_language: BCP-47 language tag or human-readable name of the
            language detected in the image (e.g. ``"Japanese"``).
        text_blocks: Ordered list of :class:`TextBlock` instances discovered
            by the extractor node and enriched by the translator node.
        output_image_bytes: Raw binary content of the reconstructed output
            image produced by the reconstructor node.
        error: Human-readable error message set by any node that encounters a
            non-recoverable failure; ``None`` when the pipeline is healthy.
        target_language: Requested target language.
        source_language: Hint for the source language, or "auto".
    """
    image_width: int | None
    image_height: int | None
    image_format: str | None
    detected_language: str | None
    text_blocks: list[TextBlock] | None
    output_image_bytes: bytes | None
    error: str | None
    target_language: str | None
    source_language: str | None


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------





class TranslationResponse(BaseModel):
    """API response returned after the pipeline completes.

    Attributes:
        success: ``True`` if the pipeline produced a translated image without
            errors; ``False`` otherwise.
        detected_language: The language identified in the source image, or
            ``None`` if detection failed or was not attempted.
        text_blocks_found: Total number of text regions detected in the image.
        output_image_b64: Base64-encoded bytes of the translated output image,
            suitable for embedding in JSON. ``None`` when ``success`` is
            ``False``.
        error: Human-readable description of the failure when ``success`` is
            ``False``; ``None`` on success.
    """

    success: bool
    detected_language: str | None
    text_blocks_found: int
    output_image_b64: str | None
    error: str | None = None

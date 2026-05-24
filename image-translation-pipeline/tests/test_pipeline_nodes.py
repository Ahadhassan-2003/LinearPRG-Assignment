"""Tests for the LangGraph pipeline node functions.

All external calls (LangChain / Anthropic API) are mocked so that no network
access or API keys are required.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.models import BoundingBox, PipelineState, TextBlock
from app.pipeline.nodes.extractor import extract_and_translate_node
from app.pipeline.nodes.reconstructor import reconstruct_image_node

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_RAW_BLOCK = {
    "original_text": "Aceite de Oliva",
    "translated_text": "Olive Oil",
    "bbox": {"x": 10, "y": 20, "width": 80, "height": 10},
    "font_size_relative": "large",
    "is_bold": True,
    "text_color": "#333333",
    "background_color": "#F5F0E8",
    "text_alignment": "center",
}

_CHAIN_SUCCESS_RESPONSE = {
    "detected_language": "Spanish",
    "text_blocks": [_RAW_BLOCK],
}


def _make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a small solid-white PNG and return its raw bytes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_text_block() -> TextBlock:
    """Return a minimal valid TextBlock for use in state fixtures."""
    return TextBlock(
        original_text="Hola",
        translated_text="Hello",
        bbox=BoundingBox(x=5, y=5, width=40, height=20),
        font_size_relative="medium",
        is_bold=False,
        text_color="#000000",
        background_color="#FFFFFF",
        text_alignment="left",
    )


# ---------------------------------------------------------------------------
# Extractor node tests
# ---------------------------------------------------------------------------


class TestExtractAndTranslateNode:
    """Tests for :func:`~app.pipeline.nodes.extractor.extract_and_translate_node`."""

    def test_returns_state_update_on_success(self) -> None:
        """Mock a successful chain response and verify the state update.

        The node should return a dict with ``detected_language``, a list of
        one validated ``TextBlock``, and ``error=None``.
        """
        state: PipelineState = {
            "image_bytes": _make_png_bytes(),
            "image_format": "PNG",
            "image_width": 100,
            "image_height": 100,
            "detected_language": None,
            "text_blocks": None,
            "output_image_bytes": None,
            "error": None,
        }

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _CHAIN_SUCCESS_RESPONSE

        with patch(
            "app.pipeline.nodes.extractor._chain",
            new=mock_chain,
        ):
            result = extract_and_translate_node(state)

        assert result["detected_language"] == "Spanish"
        assert result["error"] is None
        blocks = result["text_blocks"]
        assert isinstance(blocks, list)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].original_text == "Aceite de Oliva"
        assert blocks[0].translated_text == "Olive Oil"

    def test_returns_error_on_api_failure(self) -> None:
        """Mock the chain raising an exception and verify the error is captured.

        The node must catch the exception and return a state update with
        ``error`` set and ``text_blocks`` as an empty list.
        """
        state: PipelineState = {
            "image_bytes": _make_png_bytes(),
            "image_format": "PNG",
            "image_width": 100,
            "image_height": 100,
            "detected_language": None,
            "text_blocks": None,
            "output_image_bytes": None,
            "error": None,
        }

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API error")

        with patch(
            "app.pipeline.nodes.extractor._chain",
            new=mock_chain,
        ):
            result = extract_and_translate_node(state)

        assert "API error" in result["error"]
        assert result["text_blocks"] == []

    def test_skips_invalid_blocks_and_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Blocks that fail TextBlock validation are skipped with a warning.

        One valid block and one invalid block (bad color) are returned by the
        mock; only the valid one should appear in the result.
        """
        import logging

        bad_block = dict(_RAW_BLOCK)
        bad_block["text_color"] = "NOT_A_COLOR"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "detected_language": "Spanish",
            "text_blocks": [_RAW_BLOCK, bad_block],
        }

        state: PipelineState = {
            "image_bytes": _make_png_bytes(),
            "image_format": "PNG",
            "image_width": 100,
            "image_height": 100,
            "detected_language": None,
            "text_blocks": None,
            "output_image_bytes": None,
            "error": None,
        }

        with patch("app.pipeline.nodes.extractor._chain", new=mock_chain):
            with caplog.at_level(logging.WARNING, logger="app.pipeline.nodes.extractor"):
                result = extract_and_translate_node(state)

        assert len(result["text_blocks"]) == 1
        assert any("validation error" in m.lower() for m in caplog.messages)


# ---------------------------------------------------------------------------
# Reconstructor node tests
# ---------------------------------------------------------------------------


class TestReconstructImageNode:
    """Tests for :func:`~app.pipeline.nodes.reconstructor.reconstruct_image_node`."""

    def test_produces_valid_output_image(self) -> None:
        """A state with one text block should produce non-empty valid PNG bytes."""
        state: PipelineState = {
            "image_bytes": _make_png_bytes(100, 100),
            "image_format": "PNG",
            "image_width": 100,
            "image_height": 100,
            "detected_language": "Spanish",
            "text_blocks": [_make_text_block()],
            "output_image_bytes": None,
            "error": None,
        }

        result = reconstruct_image_node(state)

        assert result["error"] is None

        out_bytes: bytes = result["output_image_bytes"]
        assert isinstance(out_bytes, bytes)
        assert len(out_bytes) > 0

        # Must be a valid image
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.size == (100, 100)

    def test_passthrough_on_prior_error(self) -> None:
        """When a prior node has set an error, the reconstructor must be a no-op.

        The node should return an empty dict (or at minimum not overwrite the
        existing error with a new one).
        """
        state: PipelineState = {
            "image_bytes": _make_png_bytes(),
            "image_format": "PNG",
            "image_width": 100,
            "image_height": 100,
            "detected_language": None,
            "text_blocks": [],
            "output_image_bytes": None,
            "error": "previous error",
        }

        result = reconstruct_image_node(state)

        # The node must not produce output bytes when passing through
        assert result == {} or result.get("output_image_bytes") in (None, b"")
        # Must not overwrite the existing error with a different one
        assert "previous error" not in str(result.get("error", ""))

    def test_passthrough_on_empty_blocks(self) -> None:
        """No text blocks means the node returns the original image unchanged.

        The output image bytes must not be empty, must be a valid image, and
        must have the same dimensions as the input.
        """
        original_bytes = _make_png_bytes(80, 60)
        state: PipelineState = {
            "image_bytes": original_bytes,
            "image_format": "PNG",
            "image_width": 80,
            "image_height": 60,
            "detected_language": "Unknown",
            "text_blocks": [],
            "output_image_bytes": None,
            "error": None,
        }

        result = reconstruct_image_node(state)

        assert result["error"] is None
        out_bytes: bytes = result["output_image_bytes"]
        assert isinstance(out_bytes, bytes)
        assert len(out_bytes) > 0

        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.size == (80, 60)

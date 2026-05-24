"""Tests for the FastAPI HTTP endpoints.

All pipeline invocations are mocked so no API keys or real LangGraph
execution is required. Tests use FastAPI's synchronous ``TestClient``.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 50, height: int = 50) -> bytes:
    """Return raw bytes for a small solid-white PNG."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_pipeline_state(*, with_output: bool = True) -> dict:
    """Build a minimal fake final PipelineState for mock injection."""
    from app.models import BoundingBox, TextBlock

    block = TextBlock(
        original_text="Hola",
        translated_text="Hello",
        bbox=BoundingBox(x=5, y=5, width=40, height=20),
        font_size_relative="medium",
        is_bold=False,
        text_color="#000000",
        background_color="#FFFFFF",
        text_alignment="left",
    )
    return {
        "image_bytes": _make_png_bytes(),
        "image_format": "PNG",
        "image_width": 50,
        "image_height": 50,
        "detected_language": "Spanish",
        "text_blocks": [block],
        "output_image_bytes": _make_png_bytes() if with_output else b"",
        "error": None,
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the ``GET /health`` endpoint."""

    def test_returns_200_with_status_ok(self) -> None:
        """Health endpoint must return HTTP 200 with ``status == "ok"``."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_includes_version(self) -> None:
        """Health response must include a ``version`` field."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_includes_langsmith_tracing_flag(self) -> None:
        """Health response must include a boolean ``langsmith_tracing`` field."""
        response = client.get("/health")
        data = response.json()
        assert "langsmith_tracing" in data
        assert isinstance(data["langsmith_tracing"], bool)


# ---------------------------------------------------------------------------
# POST /translate-image — validation
# ---------------------------------------------------------------------------


class TestTranslateImageValidation:
    """Validation-layer tests for ``POST /translate-image``."""

    def test_missing_file_returns_422(self) -> None:
        """Omitting the required ``image`` field must return HTTP 422."""
        response = client.post("/translate-image", data={"target_language": "English"})
        assert response.status_code == 422

    def test_invalid_content_type_returns_400(self) -> None:
        """Uploading a non-image file must return HTTP 400."""
        txt_content = b"This is plain text, not an image."
        response = client.post(
            "/translate-image",
            files={"image": ("document.txt", txt_content, "text/plain")},
            data={"target_language": "English"},
        )
        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()

    def test_oversized_file_returns_413(self) -> None:
        """Uploading an image exceeding the size limit must return HTTP 413."""
        # Produce a fake 11 MB image payload — no real decoding needed since
        # the size check runs before load_image_from_bytes.
        huge_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)
        response = client.post(
            "/translate-image",
            files={"image": ("big.png", huge_bytes, "image/png")},
            data={"target_language": "English"},
        )
        assert response.status_code == 413
        assert "MB" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /translate-image — success path
# ---------------------------------------------------------------------------


class TestTranslateImageSuccess:
    """Success-path tests for ``POST /translate-image``."""

    def test_returns_200_png_with_headers(self, small_png_bytes: bytes) -> None:
        """A valid upload with a mocked pipeline should return image/png + headers."""
        fake_state = _make_pipeline_state(with_output=True)

        with patch("app.main.run_pipeline", return_value=fake_state):
            response = client.post(
                "/translate-image",
                files={"image": ("test.png", small_png_bytes, "image/png")},
                data={"target_language": "English"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "x-detected-language" in response.headers
        assert response.headers["x-detected-language"] == "Spanish"
        assert "x-text-blocks-found" in response.headers
        assert "x-pipeline-version" in response.headers

    def test_response_body_is_valid_image(self, small_png_bytes: bytes) -> None:
        """The binary body returned must be parseable as a valid PNG."""
        fake_state = _make_pipeline_state(with_output=True)

        with patch("app.main.run_pipeline", return_value=fake_state):
            response = client.post(
                "/translate-image",
                files={"image": ("test.png", small_png_bytes, "image/png")},
                data={"target_language": "English"},
            )

        assert response.status_code == 200
        img = Image.open(io.BytesIO(response.content))
        assert img.size == (50, 50)

    def test_pipeline_error_returns_500(self, small_png_bytes: bytes) -> None:
        """When the pipeline returns an error, the endpoint must return HTTP 500."""
        error_state = _make_pipeline_state(with_output=False)
        error_state["error"] = "Claude API timeout"
        error_state["output_image_bytes"] = b""

        with patch("app.main.run_pipeline", return_value=error_state):
            response = client.post(
                "/translate-image",
                files={"image": ("test.png", small_png_bytes, "image/png")},
                data={"target_language": "English"},
            )

        assert response.status_code == 500


# ---------------------------------------------------------------------------
# POST /translate-image/json
# ---------------------------------------------------------------------------


class TestTranslateImageJson:
    """Tests for the ``POST /translate-image/json`` endpoint."""

    def test_returns_success_json(self, small_png_bytes: bytes) -> None:
        """Successful pipeline run returns a TranslationResponse JSON object."""
        fake_state = _make_pipeline_state(with_output=True)

        with patch("app.main.run_pipeline", return_value=fake_state):
            response = client.post(
                "/translate-image/json",
                files={"image": ("test.png", small_png_bytes, "image/png")},
                data={"target_language": "English"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["detected_language"] == "Spanish"
        assert data["text_blocks_found"] == 1
        assert data["output_image_b64"] is not None
        assert data["error"] is None

    def test_returns_failure_json_on_pipeline_error(
        self, small_png_bytes: bytes
    ) -> None:
        """Pipeline error surfaces as ``success=false`` in the JSON body."""
        error_state = _make_pipeline_state(with_output=False)
        error_state["error"] = "Extraction failed"
        error_state["output_image_bytes"] = b""

        with patch("app.main.run_pipeline", return_value=error_state):
            response = client.post(
                "/translate-image/json",
                files={"image": ("test.png", small_png_bytes, "image/png")},
                data={"target_language": "English"},
            )

        assert response.status_code == 200  # JSON endpoint always returns 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "Extraction failed"
        assert data["output_image_b64"] is None

    def test_missing_file_returns_422(self) -> None:
        """Omitting the required ``image`` field must return HTTP 422."""
        response = client.post(
            "/translate-image/json", data={"target_language": "English"}
        )
        assert response.status_code == 422

    def test_invalid_content_type_returns_400(self) -> None:
        """Non-image Content-Type must be rejected with HTTP 400."""
        response = client.post(
            "/translate-image/json",
            files={"image": ("file.csv", b"a,b,c", "text/csv")},
            data={"target_language": "English"},
        )
        assert response.status_code == 400

    def test_b64_field_is_decodable(self, small_png_bytes: bytes) -> None:
        """The ``output_image_b64`` field must decode back to a valid PNG."""
        import base64

        fake_state = _make_pipeline_state(with_output=True)

        with patch("app.main.run_pipeline", return_value=fake_state):
            response = client.post(
                "/translate-image/json",
                files={"image": ("test.png", small_png_bytes, "image/png")},
                data={"target_language": "English"},
            )

        data = response.json()
        decoded = base64.b64decode(data["output_image_b64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (50, 50)

"""FastAPI application entry point for the Image Text Translation Pipeline.

This module defines the FastAPI app, all HTTP endpoints, request validation,
and exception handling. Pipeline logic lives entirely in
:mod:`app.pipeline.graph`; this layer is responsible only for HTTP
request/response concerns.
"""

from __future__ import annotations

import logging
import logging.config
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.models import TranslationResponse
from app.pipeline.graph import run_pipeline
from app.utils.image_utils import image_to_base64, load_image_from_bytes

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application between startup and shutdown.
    """
    logger.info(
        "Image Text Translation Pipeline starting up "
        "(LangSmith tracing: %s, project: %r).",
        settings.LANGCHAIN_TRACING_V2,
        settings.LANGCHAIN_PROJECT,
    )
    yield
    logger.info("Image Text Translation Pipeline shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Image Text Translation Pipeline",
    version=VERSION,
    description=(
        "Accepts an image containing text in any language, extracts every "
        "text region using Claude's vision capabilities via a LangGraph "
        "pipeline, and returns a new image with all text translated into the "
        "requested target language while preserving the original layout and "
        "visual style."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prototype setting — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler for any exception not handled by an endpoint.

    Logs the full traceback at ERROR level and returns a generic 500 response
    so the client always receives structured JSON rather than an HTML error
    page.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception.

    Returns:
        A 500 JSON response containing a generic error message and the
        exception detail.
    """
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_image_upload(image: UploadFile, image_bytes: bytes) -> None:
    """Run all pre-pipeline validation checks on an uploaded file.

    Raises :class:`fastapi.HTTPException` with an appropriate HTTP status code
    and a human-readable detail message if any check fails.

    Args:
        image: The ``UploadFile`` received from the multipart form.
        image_bytes: The already-read raw bytes of the upload.

    Raises:
        HTTPException: 400 if the Content-Type is not ``image/*``.
        HTTPException: 413 if the file exceeds :attr:`~app.config.Settings.MAX_IMAGE_SIZE_MB`.
    """
    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded file has Content-Type {content_type!r}. "
                "Only image/* files are accepted."
            ),
        )

    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {settings.MAX_IMAGE_SIZE_MB} MB limit.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health_check() -> dict[str, Any]:
    """Return the service health status.

    Returns:
        A JSON object with ``status``, ``version``, and whether LangSmith
        tracing is currently enabled.
    """
    return {
        "status": "ok",
        "version": VERSION,
        "langsmith_tracing": settings.LANGCHAIN_TRACING_V2.lower() == "true",
    }


@app.post(
    "/v1/translate-image",
    summary="Translate image text (returns image)",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        400: {"description": "Invalid file type"},
        413: {"description": "File too large"},
        500: {"description": "Pipeline error"},
    },
)
async def translate_image(
    image: UploadFile,
    target_language: str = Form(default="English"),
    source_language: str = Form(default="auto"),
) -> Response:
    """Translate all text in an uploaded image and return the modified image.

    Accepts a multipart/form-data POST containing the source image and
    optional language parameters. Returns the reconstructed image as raw PNG
    bytes with pipeline metadata in response headers.

    Args:
        image: Image file uploaded as multipart form data.
        target_language: Language to translate text into. Defaults to
            ``"English"``.
        source_language: Source language hint. Use ``"auto"`` (default) to
            let the model detect the language automatically.

    Returns:
        A ``200 image/png`` response containing the translated image, with
        the headers ``X-Detected-Language``, ``X-Text-Blocks-Found``, and
        ``X-Pipeline-Version`` populated from the pipeline result.

    Raises:
        HTTPException: 400 if the upload is not an image.
        HTTPException: 413 if the upload exceeds the size limit.
        HTTPException: 500 if the pipeline encounters an unrecoverable error.
    """
    logger.info(
        "POST /translate-image — file=%r target_language=%r source_language=%r",
        image.filename,
        target_language,
        source_language,
    )

    image_bytes = await image.read()
    _validate_image_upload(image, image_bytes)

    pil_image, fmt, width, height = load_image_from_bytes(image_bytes)

    logger.info(
        "Pipeline starting: format=%s size=%dx%d target=%r",
        fmt, width, height, target_language,
    )

    final_state = run_pipeline(
        image_bytes=image_bytes,
        image_width=width,
        image_height=height,
        image_format=fmt,
        target_language=target_language,
        source_language=source_language,
    )

    if final_state.get("error"):
        logger.error("Pipeline error: %s", final_state["error"])
        raise HTTPException(status_code=500, detail=final_state["error"])

    output_bytes: bytes = final_state.get("output_image_bytes") or b""
    if not output_bytes:
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed without producing output image bytes.",
        )

    text_blocks = final_state.get("text_blocks") or []
    detected_language = final_state.get("detected_language") or "Unknown"

    logger.info(
        "Pipeline completed: detected_language=%r blocks=%d output_bytes=%d",
        detected_language, len(text_blocks), len(output_bytes),
    )

    media_type = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    extension = "jpg" if fmt.upper() == "JPEG" else "png"
    return Response(
        content=output_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="translated_image.{extension}"',
            "X-Detected-Language": detected_language,
            "X-Text-Blocks-Found": str(len(text_blocks)),
            "X-Pipeline-Version": VERSION,
        },
    )


@app.post(
    "/v1/translate-image/json",
    summary="Translate image text (returns JSON)",
    response_model=TranslationResponse,
)
async def translate_image_json(
    image: UploadFile,
    target_language: str = Form(default="English"),
    source_language: str = Form(default="auto"),
) -> TranslationResponse:
    """Translate all text in an uploaded image and return a JSON response.

    Identical to ``POST /translate-image`` but returns a
    :class:`~app.models.TranslationResponse` JSON object whose
    ``output_image_b64`` field carries the translated image as a base64
    string. Useful for debugging, front-end integration, and the interview
    demo.

    Args:
        image: Image file uploaded as multipart form data.
        target_language: Language to translate text into. Defaults to
            ``"English"``.
        source_language: Source language hint. Defaults to ``"auto"``.

    Returns:
        A :class:`~app.models.TranslationResponse` with ``success=True`` and
        the base64-encoded output image on success, or ``success=False`` and
        an ``error`` message on failure.

    Raises:
        HTTPException: 400 if the upload is not an image.
        HTTPException: 413 if the upload exceeds the size limit.
    """
    logger.info(
        "POST /translate-image/json — file=%r target_language=%r source_language=%r",
        image.filename,
        target_language,
        source_language,
    )

    image_bytes = await image.read()
    _validate_image_upload(image, image_bytes)

    pil_image, fmt, width, height = load_image_from_bytes(image_bytes)

    logger.info(
        "Pipeline starting: format=%s size=%dx%d target=%r",
        fmt, width, height, target_language,
    )

    final_state = run_pipeline(
        image_bytes=image_bytes,
        image_width=width,
        image_height=height,
        image_format=fmt,
        target_language=target_language,
        source_language=source_language,
    )

    text_blocks = final_state.get("text_blocks") or []
    detected_language = final_state.get("detected_language") or None

    if final_state.get("error"):
        logger.error("Pipeline error: %s", final_state["error"])
        return TranslationResponse(
            success=False,
            detected_language=detected_language,
            text_blocks_found=len(text_blocks),
            output_image_b64=None,
            error=final_state["error"],
        )

    output_bytes: bytes = final_state.get("output_image_bytes") or b""
    if not output_bytes:
        return TranslationResponse(
            success=False,
            detected_language=detected_language,
            text_blocks_found=len(text_blocks),
            output_image_b64=None,
            error="Pipeline completed without producing output image bytes.",
        )

    b64 = image_to_base64(output_bytes)

    logger.info(
        "Pipeline completed: detected_language=%r blocks=%d",
        detected_language, len(text_blocks),
    )

    return TranslationResponse(
        success=True,
        detected_language=detected_language,
        text_blocks_found=len(text_blocks),
        output_image_b64=b64,
        error=None,
    )

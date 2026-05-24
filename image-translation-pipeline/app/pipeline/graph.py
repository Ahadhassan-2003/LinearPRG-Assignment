"""LangGraph pipeline graph for the image translation pipeline.

This module wires the extraction/translation node and the reconstruction node
into a compiled :class:`~langgraph.graph.StateGraph`, adds a conditional edge
that short-circuits reconstruction on error or when no text is found, and
exposes:

- ``pipeline`` — the compiled graph ready for invocation.
- ``run_pipeline`` — a :func:`~langsmith.traceable`-decorated convenience
  wrapper that builds the initial state and invokes the graph.

LangSmith environment variables are configured at import time from
:attr:`~app.config.settings` so that every run is traced automatically.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from langgraph.graph import END, StateGraph

from app.config import settings
from app.models import PipelineState
from app.pipeline.nodes.extractor import extract_and_translate_node
from app.pipeline.nodes.reconstructor import reconstruct_image_node

logger = logging.getLogger(__name__)

from functools import lru_cache

# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

_RECONSTRUCT = "reconstruct"
_END = "end"


def should_reconstruct(state: PipelineState) -> str:
    """Decide whether the reconstruction node should run.

    Called by LangGraph after the ``extract_and_translate`` node completes.
    Routes to the reconstruction node only when extraction succeeded and at
    least one text block was found; otherwise routes directly to the graph's
    terminal node.

    Args:
        state: The current :class:`~app.models.PipelineState`.

    Returns:
        ``"reconstruct"`` if reconstruction should proceed, or ``"end"`` if
        the pipeline should terminate early.
    """
    if state.get("error"):
        logger.info("Skipping reconstruction — pipeline error: %s", state["error"])
        return _END

    text_blocks = state.get("text_blocks") or []
    if not text_blocks:
        logger.info("Skipping reconstruction — no text blocks found.")
        return _END

    return _RECONSTRUCT


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

_builder: StateGraph = StateGraph(PipelineState)

# Nodes
_builder.add_node("extract_and_translate", extract_and_translate_node)
_builder.add_node("reconstruct", reconstruct_image_node)

# Entry point
_builder.set_entry_point("extract_and_translate")

# Conditional edge: extraction → (reconstruct | END)
_builder.add_conditional_edges(
    "extract_and_translate",
    should_reconstruct,
    {
        _RECONSTRUCT: "reconstruct",
        _END: END,
    },
)

# Terminal edge: reconstruct → END
_builder.add_edge("reconstruct", END)

@lru_cache(maxsize=1)
def get_pipeline():
    """Lazily compile the graph so startup isn't aborted by import errors."""
    return _builder.compile()

# ---------------------------------------------------------------------------
# Public run_pipeline helper
# ---------------------------------------------------------------------------


@traceable(name="image_translation_pipeline")
def run_pipeline(
    image_bytes: bytes,
    image_width: int,
    image_height: int,
    image_format: str,
    target_language: str = "English",
    source_language: str = "auto",
    filename: str | None = None,
) -> dict[str, Any]:
    """Run the full image translation pipeline as a single LangSmith trace.

    Constructs the initial :class:`~app.models.PipelineState`, invokes the
    compiled LangGraph pipeline, and returns the final state dict.

    The ``@traceable`` decorator causes the entire call — including every
    node invocation — to appear as one top-level trace in LangSmith, with
    each node appearing as a child span.

    Args:
        image_bytes: Raw bytes of the source image.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        image_format: Pillow format string of the source image (e.g.
            ``"PNG"``, ``"JPEG"``).
        target_language: Human-readable name of the language to translate
            text into. Defaults to ``"English"``.
        source_language: Human-readable name of the source language, or
            ``"auto"``. Defaults to ``"auto"``.

    Returns:
        The final :class:`~app.models.PipelineState` dict after all nodes
        have executed.
    """
    if filename:
        run = get_current_run_tree()
        if run:
            run.name = filename

    initial_state: PipelineState = {
        "image_bytes": image_bytes,
        "image_width": image_width,
        "image_height": image_height,
        "image_format": image_format,
        "detected_language": None,
        "text_blocks": None,
        "output_image_bytes": None,
        "error": None,
        "target_language": target_language,
        "source_language": source_language,
    }

    logger.info(
        "Starting pipeline: format=%s size=%dx%d target_language=%r source_language=%r",
        image_format,
        image_width,
        image_height,
        target_language,
        source_language,
    )

    pipeline = get_pipeline()
    config = {"run_name": filename} if filename else {}
    final_state: dict[str, Any] = pipeline.invoke(initial_state, config=config)

    if final_state.get("error"):
        logger.error("Pipeline finished with error: %s", final_state["error"])
    else:
        blocks = final_state.get("text_blocks") or []
        logger.info(
            "Pipeline finished successfully: %d block(s) translated.",
            len(blocks),
        )

    return final_state

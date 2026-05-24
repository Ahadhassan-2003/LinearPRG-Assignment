"""Extraction and translation node for the image translation pipeline.

**What this module does**

Sends the raw image bytes held in ``PipelineState["image_bytes"]`` to
Claude claude-sonnet-4-5 via a single LangChain vision call. The model analyses the
image, extracts every visible text region with spatial and typographic
metadata, and simultaneously translates the text into the requested target
language — all in one API round-trip.

**Inputs (read from PipelineState)**

- ``image_bytes`` *(bytes, required)* — raw bytes of the source image.
- ``target_language`` *(str, optional)* — human-readable translation target
  (e.g. ``"English"``). Falls back to ``"English"`` if absent.

**Outputs (partial state update dict)**

- ``detected_language`` *(str)* — primary language identified in the image.
- ``text_blocks`` *(list[TextBlock])* — validated text regions, each carrying
  original text, translated text, a percentage-based bounding box, and
  typographic hints (size, bold, colour, alignment).
- ``error`` *(str | None)* — ``None`` on success; error message string on
  any unrecoverable failure (API error, JSON parse failure, etc.).

**LangSmith span name**: ``"extract_and_translate"``

The function's ``__name__`` is overridden to this value so that every
invocation appears under a clean, human-readable label in LangSmith traces
rather than the full module-qualified name.
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

import io
import difflib
import pytesseract
from PIL import Image

from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from app.models import PipelineState, TextBlock
from app.utils.image_utils import image_bytes_to_base64_uri
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT_TEMPLATE: str = """You are an expert multilingual OCR and \
translation engine. Your role is to analyse images precisely, extract every \
visible text region with full spatial and typographic metadata, and translate \
the text naturally into the requested target language."""

_HUMAN_INSTRUCTION: str = """\
Analyse the image provided and extract ALL visible text regions.

For each text region return a JSON object with EXACTLY these fields:
  - original_text   : the exact text as it appears in the image (preserve \
brand names, measurements, symbols, and punctuation unchanged)
  - translated_text : a natural, idiomatic translation into {target_language} \
(not word-by-word). Consider that the original text is in {source_language}. \
If the text is already in {target_language} or is a brand \
name / number / symbol that should not be translated, set translated_text \
identical to original_text.
  - bbox            : object with keys x, y, width, height expressed as \
percentages (0–100) of the image's total width and height respectively, where \
(x=0, y=0) is the top-left corner.
  - font_size_relative : one of "large", "medium", or "small" relative to the \
overall content of the image.
  - is_bold         : boolean — true if the text appears in bold weight.
  - text_color      : the foreground (glyph) colour as a 6-digit CSS hex string, \
e.g. "#FFFFFF".
  - background_color: the background fill colour directly behind the text as a \
6-digit CSS hex string, e.g. "#2D5016".
  - text_alignment  : one of "left", "center", or "right".

Rules:
  - Extract EVERY text region, no matter how small.
  - Preserve brand names, model numbers, and measurements exactly.
  - If text is already in {target_language}, set translated_text == original_text.
  - Colours MUST be hex strings starting with "#" followed by exactly 6 hex digits.
  - Bounding box values MUST be numbers between 0 and 100.

Return ONLY a single JSON object — no markdown fences, no explanations — in \
this exact structure:
{{
  "detected_language": "<name of the primary language found in the image>",
  "text_blocks": [
    {{ ...block fields... }},
    ...
  ]
}}"""

# ---------------------------------------------------------------------------
# LangChain objects (module-level, constructed once)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_chain():
    llm = ChatAnthropic(
        model=settings.EXTRACTION_CLAUDE_MODEL,
        temperature=0,
        max_tokens=settings.EXTRACTION_MAX_TOKENS,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_PROMPT_TEMPLATE),
            (
                "human",
                [
                    # Vision content block — image is injected at runtime
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_uri}"},
                    },
                    # Text instruction with target_language variable
                    {
                        "type": "text",
                        "text": _HUMAN_INSTRUCTION,
                    },
                ],
            ),
        ]
    )

    return prompt | llm | JsonOutputParser()

# ---------------------------------------------------------------------------
# OCR Helpers
# ---------------------------------------------------------------------------

def _extract_tesseract_lines(image_bytes: bytes) -> list[dict]:
    """Extract line-level bounding boxes and text from an image using Tesseract."""
    image = Image.open(io.BytesIO(image_bytes))
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    # Group words by (block_num, par_num, line_num)
    lines = {}
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue
            
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        if key not in lines:
            lines[key] = {
                'text': text,
                'left': x,
                'top': y,
                'right': x + w,
                'bottom': y + h,
            }
        else:
            lines[key]['text'] += " " + text
            lines[key]['left'] = min(lines[key]['left'], x)
            lines[key]['top'] = min(lines[key]['top'], y)
            lines[key]['right'] = max(lines[key]['right'], x + w)
            lines[key]['bottom'] = max(lines[key]['bottom'], y + h)
            
    # Convert to a flat list with width/height
    result = []
    for val in lines.values():
        val['width'] = val['right'] - val['left']
        val['height'] = val['bottom'] - val['top']
        result.append(val)
    return result

# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def extract_and_translate_node(state: PipelineState) -> dict[str, Any]:
    """LangGraph node: extract text from the image and translate it.

    Sends the image stored in ``state["image_bytes"]`` to Claude claude-sonnet-4-5
    using a structured vision prompt. The model returns a JSON object
    containing the detected source language and a list of text blocks, each
    with original text, translated text, bounding box, and typographic
    metadata. Every block is validated against the :class:`~app.models.TextBlock`
    Pydantic model; blocks that fail validation are skipped with a warning.

    This function is named ``extract_and_translate`` so that it appears under
    that label in LangSmith traces.

    Args:
        state: The current :class:`~app.models.PipelineState` dict. Must
            contain at least ``"image_bytes"``. Optionally reads
            ``"target_language"`` if it exists in the state (falls back to
            ``"English"``).

    Returns:
        A partial state-update dict containing:
          - ``"detected_language"`` (str): language detected in the image.
          - ``"text_blocks"`` (list[TextBlock]): validated text blocks.
          - ``"error"`` (str | None): ``None`` on success, error message on failure.
    """
    try:
        # ------------------------------------------------------------------
        # 1. Prepare image URI
        # ------------------------------------------------------------------
        image_bytes: bytes = state["image_bytes"]
        fmt = state.get("image_format") or "PNG"
        mime_type = f"image/{fmt.lower()}"
        image_uri = image_bytes_to_base64_uri(image_bytes, mime_type=mime_type)

        # ------------------------------------------------------------------
        # 2. Resolve target and source languages from state
        # ------------------------------------------------------------------
        target_language: str = state.get("target_language") or "English"
        source_language: str = state.get("source_language") or "auto"

        # ------------------------------------------------------------------
        # 3. Invoke the LangChain chain
        # ------------------------------------------------------------------
        chain = _get_chain()
        raw_response: dict[str, Any] = chain.invoke(
            {
                "image_uri": image_uri,
                "target_language": target_language,
                "source_language": source_language,
            }
        )

        # ------------------------------------------------------------------
        # 4. Extract top-level fields from the response
        # ------------------------------------------------------------------
        detected_language: str = str(
            raw_response.get("detected_language", "Unknown")
        )
        raw_blocks: list[dict[str, Any]] = raw_response.get("text_blocks", [])

        # ------------------------------------------------------------------
        # 5. Extract precise bounding boxes using Tesseract OCR
        # ------------------------------------------------------------------
        tesseract_lines = _extract_tesseract_lines(image_bytes)
        img_w = state.get("image_width")
        img_h = state.get("image_height")
        if img_w is None or img_h is None:
            # Fallback if not available in state
            pil_img = Image.open(io.BytesIO(image_bytes))
            img_w, img_h = pil_img.size

        # ------------------------------------------------------------------
        # 6. Match Claude text to Tesseract and override bounding boxes
        # ------------------------------------------------------------------
        STOP_WORDS = {"de", "la", "el", "y", "en", "un", "una", "a", "los", "las", "con", "por", "para", "su", "al", "del", "lo", "o"}
        
        def is_fuzzy_match(t_text: str, claude_text: str) -> bool:
            import re
            import difflib
            t_clean = re.sub(r'[^\w\s]', '', t_text.lower()).strip()
            c_clean = re.sub(r'[^\w\s]', '', claude_text.lower())
            
            if difflib.SequenceMatcher(None, t_clean, c_clean).ratio() >= 0.65:
                return True
                
            if len(t_clean) >= 6 and t_clean in c_clean:
                return True
                
            t_words = [w for w in t_clean.split() if w not in STOP_WORDS]
            c_words = set(c_clean.split())
            
            if not t_words:
                return False
                
            common = set(t_words).intersection(c_words)
            ratio = len(common) / len(t_words)
            
            if len(t_words) <= 2:
                return ratio == 1.0
            return ratio >= 0.5

        for raw_block in raw_blocks:
            claude_text = raw_block.get("original_text", "").lower()
            if not claude_text:
                continue

            matched_lines = []

            for t_line in tesseract_lines:
                if is_fuzzy_match(t_line['text'], claude_text):
                    matched_lines.append(t_line)

            if matched_lines:
                # Merge the bounding boxes of all matched Tesseract lines
                min_left = min(line['left'] for line in matched_lines)
                min_top = min(line['top'] for line in matched_lines)
                max_right = max(line['right'] for line in matched_lines)
                max_bottom = max(line['bottom'] for line in matched_lines)
                
                raw_block["bbox"] = {
                    "x": (min_left / img_w) * 100.0,
                    "y": (min_top / img_h) * 100.0,
                    "width": ((max_right - min_left) / img_w) * 100.0,
                    "height": ((max_bottom - min_top) / img_h) * 100.0,
                }
                logger.info("Matched Claude text %r to %d Tesseract lines", raw_block["original_text"], len(matched_lines))
            else:
                logger.warning("No Tesseract match for %r (using LLM fallback bbox)", raw_block.get("original_text"))

        # ------------------------------------------------------------------
        # 7. Validate each block against the TextBlock model
        # ------------------------------------------------------------------
        validated_blocks: list[TextBlock] = []
        for idx, raw_block in enumerate(raw_blocks):
            try:
                block = TextBlock.model_validate(raw_block)
                validated_blocks.append(block)
            except ValidationError as ve:
                logger.warning(
                    "Skipping text block %d due to validation error: %s",
                    idx,
                    ve,
                )

        logger.info(
            "Extraction complete: detected_language=%r, blocks=%d/%d valid",
            detected_language,
            len(validated_blocks),
            len(raw_blocks),
        )

        return {
            "detected_language": detected_language,
            "text_blocks": validated_blocks,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("extract_and_translate_node failed: %s", exc)
        return {
            "error": str(exc),
            "text_blocks": [],
        }


# Rename so LangSmith traces display a clean label instead of the full
# module-qualified name.
extract_and_translate_node.__name__ = "extract_and_translate"
extract_and_translate_node.__qualname__ = "extract_and_translate"

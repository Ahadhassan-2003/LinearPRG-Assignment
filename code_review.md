# Code Review — Image Text Translation Pipeline

> Full review across all layers: config, models, pipeline nodes, API, utils, and tests.
> Issues are grouped by severity then category.

---

## 🔴 Critical (would cause bugs in production)

### 1. `extractor.py` — LLM objects constructed at **import time**, ignoring `settings`

**File**: `app/pipeline/nodes/extractor.py`, lines 101–128

```python
_llm = ChatAnthropic(
    model="claude-sonnet-4-5",   # hardcoded — ignores EXTRACTION_CLAUDE_MODEL
    temperature=0,
    max_tokens=4096,              # hardcoded — ignores EXTRACTION_MAX_TOKENS
)
```

`Settings` defined `EXTRACTION_CLAUDE_MODEL` and `EXTRACTION_MAX_TOKENS` precisely so
these could be changed via `.env` without touching code — but the extractor
ignores both. The values are permanently baked in at import time.

**Fix**: wrap construction in a factory that reads from `settings`:

```python
from app.config import settings

_llm = ChatAnthropic(
    model=settings.EXTRACTION_CLAUDE_MODEL,
    temperature=0,
    max_tokens=settings.EXTRACTION_MAX_TOKENS,
)
```

---

### 2. `extractor.py` — MIME type hardcoded to `image/png` regardless of actual format

**File**: `app/pipeline/nodes/extractor.py`, line 168

```python
image_uri = image_bytes_to_base64_uri(image_bytes, mime_type="image/png")
```

If the user uploads a JPEG, the data URI claims it is PNG. Claude receives
incorrect metadata and may misparse the payload or return degraded results.

**Fix**: derive the MIME type from the actual image format held in state:

```python
fmt = state.get("image_format") or "PNG"
mime_type = f"image/{fmt.lower()}"
image_uri = image_bytes_to_base64_uri(image_bytes, mime_type=mime_type)
```

---

### 3. `models.py` — `PipelineState` declares `image_bytes` as required but `total=False` overrides that

**File**: `app/models.py`, line 145

```python
class PipelineState(TypedDict, total=False):
    image_bytes: bytes  # comment says "Required"
```

`total=False` makes **every** key optional at the TypedDict level, including
`image_bytes`. Nothing prevents the dict from being constructed without it,
and the nodes access `state["image_bytes"]` directly (no `.get()`), which
would raise a `KeyError` at runtime.

**Fix**: split into a two-class pattern (required + optional):

```python
class _PipelineStateRequired(TypedDict):
    image_bytes: bytes

class PipelineState(_PipelineStateRequired, total=False):
    image_width: int | None
    image_height: int | None
    ...
```

---

### 4. `graph.py` — `target_language` injected as an **undeclared state key**

**File**: `app/pipeline/graph.py`, line 156

```python
"target_language": target_language,  # type: ignore[typeddict-unknown-key]
```

`target_language` is not a field of `PipelineState`. The `# type: ignore`
suppresses the type error rather than fixing it. If LangGraph's state
reducer ever becomes strict about unknown keys (already the case in some
versions), this will raise at runtime. It is also invisible to anyone reading
`PipelineState` — there is no indication that this key exists.

**Fix**: add `target_language: NotRequired[str]` to `PipelineState`, then
remove the `# type: ignore` comment.

---

### 5. `main.py` — `pil_image` result from `load_image_from_bytes` is discarded; JPEG images always returned as PNG

**File**: `app/main.py`, lines 229 and 265

```python
pil_image, fmt, width, height = load_image_from_bytes(image_bytes)  # pil_image unused
...
return Response(content=output_bytes, media_type="image/png")  # always PNG
```

The endpoint calls `load_image_from_bytes` to get `fmt`, then passes `fmt`
into `run_pipeline` so the reconstructor can save JPEG as JPEG — but the
`media_type` in the response is unconditionally `"image/png"`. A JPEG input
produces JPEG output bytes but a `Content-Type: image/png` header, confusing
any client that parses the content type.

**Fix**:
```python
media_type = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
return Response(content=output_bytes, media_type=media_type, ...)
```

---

## 🟠 High (incorrect behaviour under common conditions)

### 6. `config.py` — `ANTHROPIC_API_KEY` and `LANGCHAIN_API_KEY` required at import time; breaks all tests that import `app.main`

**File**: `app/config.py`, lines 29 and 32

Both keys have no default and `Settings()` is called at module level on
line 58. Any import of `app.config` — including transitive imports via
`app.main` or `app.pipeline.graph` — will raise `pydantic_settings.
ValidationError` if the `.env` file is absent or the variables are unset.

The tests work today only because a real `.env` is present in the workspace.
In a clean CI environment without secrets, **all 28 tests would fail on
import**, not on assertion.

**Fix**: make both keys optional with `None` as default, and guard usages:
```python
ANTHROPIC_API_KEY: str | None = None
LANGCHAIN_API_KEY: str | None = None
```
Validate at request time (in `run_pipeline`) rather than at startup, or use
`model_config = SettingsConfigDict(..., env_ignore_empty=True)`.

---

### 7. `graph.py` — LangSmith env-vars written **after** `_llm` is constructed in `extractor.py`

**File**: `app/pipeline/graph.py`, lines 36–43

Python evaluates imports in order. `extractor.py` is imported at line 27 of
`graph.py`, which constructs `_llm = ChatAnthropic(...)` — **before** the
`os.environ` assignments on lines 36–43 run. LangChain reads `LANGCHAIN_API_KEY`
from the environment when the client is instantiated, so tracing will not be
set up for the LLM client even though the env vars are written later.

**Fix**: move the `os.environ` block to `app/config.py` (or a `setup_langsmith()`
call at app startup in `lifespan`) so it runs before any import of the
extractor module.

---

### 8. `reconstructor.py` — re-decodes `image_bytes` from scratch on every call; ignores `image_width`/`image_height` from state

**File**: `app/pipeline/nodes/reconstructor.py`, lines 299–300

```python
pil_image, fmt, img_w, img_h = load_image_from_bytes(image_bytes)
```

The main endpoint already decoded the image in `load_image_from_bytes` to
get `width` and `height`, which it stored in `state["image_width"]` and
`state["image_height"]`. The reconstructor ignores those values and decodes
the image a second time. This is redundant work and means `image_format`
state key and the `fmt` returned by re-decode can theoretically disagree
(e.g. the state says `"JPEG"` but the PIL re-decode returns `None` because
`Image.open` + `.load()` on a BytesIO loses the original format after a
second open).

Note that `image.format` is `None` after a re-open from bytes that didn't
come directly from a file with an extension — the code handles this with
`or "PNG"` but it silently ignores the state's `image_format` value.

---

### 9. `main.py` — `source_language` accepted from caller but never passed to the pipeline

**File**: `app/main.py`, lines 193–194

```python
source_language: str = Form(default="auto"),
```

`source_language` is received, logged, and then silently dropped. The
extractor prompt has no concept of a source language hint. Either:
- Remove the parameter from the API (to avoid misleading callers), or
- Pass it into the state and include it in the prompt instruction.

---

### 10. `reconstructor.py` — `_cover_original_text` draws `x + w` and `y + h` which is **one pixel too wide**

**File**: `app/pipeline/nodes/reconstructor.py`, line 143

```python
draw.rectangle([x, y, x + w, y + h], fill=fill)
```

PIL's `rectangle` treats coordinates as **inclusive** on both ends, so
`rectangle([0, 0, 10, 10])` draws an 11×11 rectangle. The correct right/bottom
coordinates should be `x + w - 1` and `y + h - 1`. This causes a 1-pixel
overflow on each edge of every text block, potentially overwriting adjacent
content.

**Fix**:
```python
draw.rectangle([x, y, x + w - 1, y + h - 1], fill=fill)
```

---

## 🟡 Medium (design / maintainability concerns)

### 11. `extractor.py` — `warnings.filterwarnings("ignore", ...)` called inside the node function on every invocation

**File**: `app/pipeline/nodes/extractor.py`, line 161

```python
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
```

This is a **process-global** mutation called on every single API request.
It permanently suppresses warnings for the whole process after the first
call, which hides legitimate issues. It also has no corresponding
`warnings.resetwarnings()`.

**Fix**: call it once at module level, or use `warnings.catch_warnings()` as
a context manager if truly per-call scoping is needed.

---

### 12. `image_utils.py` — `sample_background_color` is never called anywhere

**File**: `app/utils/image_utils.py`, lines 107–174

The function is implemented and tested in isolation, but the reconstructor
uses the `background_color` field from the LLM response directly instead of
sampling the actual image. This is fine as a design choice (trust the model),
but the function is dead code — it adds surface area with no callers.

**Fix**: either wire it in as a fallback when the model returns white/black
as background colour, or remove it and its test.

---

### 13. `image_utils.py` — `calculate_font_size` uses PIL's default font to measure, then `reconstructor.py` loads a different font to render

The size is estimated with `ImageFont.load_default()` but the actual
rendering uses the system TrueType font (Arial/DejaVu). These have very
different metrics, so the size estimate can be significantly off — text may
still overflow even after the shrink loop converges.

**Fix**: pass the actual font object into `calculate_font_size` (or measure
with the real font in `_draw_translated_text` directly), so measurement and
rendering use the same metrics.

---

### 14. `graph.py` — `pipeline` compiled at **module level**; any import error crashes the whole app

**File**: `app/pipeline/graph.py`, lines 84–107

`_builder.compile()` runs at import time. If LangGraph's internals change or
a node has a bad signature, the FastAPI server will fail to start with an
unrelated-looking import error.

**Fix**: wrap compilation in a function and call it lazily, or at minimum
add a try/except with a clear error message.

---

### 15. `models.py` — `TranslationRequest` is defined but never used by any endpoint

**File**: `app/models.py`, lines 182–194

Both endpoints accept `target_language` and `source_language` as `Form`
fields directly rather than using `TranslationRequest`. The model is dead
code and creates a false impression that there is a JSON request body.

**Fix**: either use `TranslationRequest` in the endpoints, or remove it.

---

### 16. `config.py` — extra blank lines between imports and class body (minor style)

**File**: `app/config.py`, lines 3–7

```python
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
```

Three blank lines instead of two (PEP 8 mandates two between top-level
definitions). Minor, but tools like `ruff` would flag this.

---

## 🟢 Low / Observations

### 17. No request ID / correlation ID in logs

Each log line is independent. In production, multiple concurrent requests
interleave their log lines with no way to correlate them. A middleware that
injects a UUID request ID into every log record (using `logging.LoggerAdapter`
or `contextvars`) would make debugging significantly easier.

### 18. `output_images/` directory not in `.gitignore` itself — only `*.png` files inside it

**File**: `.gitignore`
```
output_images/*.png
```
The directory itself is tracked but empty (no `.gitkeep`). If a non-PNG
output format is added later (e.g. JPEG), those files will be committed.
Consider `output_images/*` or add a `.gitkeep`.

### 19. No API versioning

The API is at `/translate-image` with no version prefix (`/v1/`). Once
deployed, breaking changes to the request/response schema will require a
coordinated cut-over. A `/v1/` prefix costs nothing now and prevents pain later.

### 20. `test_api.py` — `TestClient` is constructed as a **module-level singleton**

**File**: `tests/test_api.py`, line 13

```python
client = TestClient(app, raise_server_exceptions=False)
```

A module-level client means every test class shares the same application
state. This is fine today but can cause subtle ordering-dependent failures
once the app accumulates real startup side-effects (DB connections, caches,
etc.). The recommended pattern is to create the client inside a `pytest`
fixture with appropriate scope.

### 21. `ImageDraw` imported in `image_utils.py` but never used

**File**: `app/utils/image_utils.py`, line 13

```python
from PIL import Image, ImageDraw, ImageFont
```

`ImageDraw` is unused in this module (it is used in `reconstructor.py`).
Ruff/flake8 would flag this as `F401`.

---

## Summary table

| # | Severity | File | Issue |
|---|---|---|---|
| 1 | 🔴 Critical | `extractor.py` | `settings` values ignored for model/max_tokens |
| 2 | 🔴 Critical | `extractor.py` | MIME type hardcoded to `image/png` for all formats |
| 3 | 🔴 Critical | `models.py` | `image_bytes` falsely required in `total=False` TypedDict |
| 4 | 🔴 Critical | `graph.py` | `target_language` is an undeclared state key |
| 5 | 🔴 Critical | `main.py` | JPEG inputs always returned with `Content-Type: image/png` |
| 6 | 🟠 High | `config.py` | Required API keys crash import in CI (no `.env`) |
| 7 | 🟠 High | `graph.py` | LangSmith env vars set after `_llm` is constructed |
| 8 | 🟠 High | `reconstructor.py` | Image decoded twice; `image_format` from state ignored |
| 9 | 🟠 High | `main.py` | `source_language` received but silently dropped |
| 10 | 🟠 High | `reconstructor.py` | Off-by-one in `rectangle()` — 1-px overflow on each edge |
| 11 | 🟡 Medium | `extractor.py` | `warnings.filterwarnings` global mutation per request |
| 12 | 🟡 Medium | `image_utils.py` | `sample_background_color` is dead code |
| 13 | 🟡 Medium | `image_utils.py` | Font metric mismatch between size-estimation and rendering |
| 14 | 🟡 Medium | `graph.py` | `pipeline.compile()` at module level hides startup errors |
| 15 | 🟡 Medium | `models.py` | `TranslationRequest` defined but unused |
| 16 | 🟡 Medium | `config.py` | PEP 8: three blank lines instead of two |
| 17 | 🟢 Low | `main.py` | No request correlation ID in logs |
| 18 | 🟢 Low | `.gitignore` | Non-PNG outputs in `output_images/` would be committed |
| 19 | 🟢 Low | `main.py` | No API version prefix (`/v1/`) |
| 20 | 🟢 Low | `test_api.py` | Module-level `TestClient` singleton |
| 21 | 🟢 Low | `image_utils.py` | `ImageDraw` imported but unused |

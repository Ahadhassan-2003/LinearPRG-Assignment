"""Application settings loaded from environment variables via pydantic-settings."""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for the image translation pipeline.

    All fields are populated from environment variables (or a ``.env`` file).
    Fields with defaults are optional; fields without defaults are required.

    Attributes:
        ANTHROPIC_API_KEY: Secret key for authenticating with the Anthropic API.
            Optional at import time; validated at request time.
        LANGCHAIN_API_KEY: Secret key for LangSmith tracing and observability.
            Optional at import time; only needed when tracing is enabled.
        LANGCHAIN_TRACING_V2: Enables LangSmith trace upload when ``\"true\"``.
        LANGCHAIN_PROJECT: Project name shown in the LangSmith dashboard.
        MAX_IMAGE_SIZE_MB: Maximum permitted upload size in megabytes.
        DEFAULT_TARGET_LANGUAGE: Fallback translation target when the caller
            does not specify one.
        EXTRACTION_CLAUDE_MODEL: Claude model identifier used by the extraction
            and translation node.
        EXTRACTION_MAX_TOKENS: Maximum number of tokens the extraction model
            may generate in a single response.
    """

    # Made optional so the app can start and tests can run without a real .env.
    # The extractor validates the key is present at request time.
    ANTHROPIC_API_KEY: str | None = None

    # Optional — only needed when LANGCHAIN_TRACING_V2 is "true".
    LANGCHAIN_API_KEY: str | None = None

    # Enable LangSmith tracing
    LANGCHAIN_TRACING_V2: str = "true"

    # LangSmith project name
    LANGCHAIN_PROJECT: str = "image-translation-pipeline"

    # Maximum allowed image upload size in megabytes
    MAX_IMAGE_SIZE_MB: int = 10

    # Default language to translate text into
    DEFAULT_TARGET_LANGUAGE: str = "English"

    # Claude model used by the extraction / translation node
    EXTRACTION_CLAUDE_MODEL: str = "claude-sonnet-4-5"

    # Token budget for the extraction model response
    EXTRACTION_MAX_TOKENS: int = 4096

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()

# ---------------------------------------------------------------------------
# LangSmith / LangChain env-var bootstrap  (fix #7)
# ---------------------------------------------------------------------------
# These MUST be written to os.environ BEFORE any LangChain object
# (ChatAnthropic, etc.) is constructed, because LangChain reads them at
# instantiation time.  config.py is always the first app module imported,
# so setting them here guarantees they are present when extractor.py's
# module-level ``_llm`` is built.

if settings.LANGCHAIN_TRACING_V2:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.LANGCHAIN_TRACING_V2)
if settings.LANGCHAIN_PROJECT:
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGCHAIN_PROJECT)
if settings.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGCHAIN_API_KEY)
if settings.ANTHROPIC_API_KEY:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.ANTHROPIC_API_KEY)

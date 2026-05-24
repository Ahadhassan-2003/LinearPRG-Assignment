"""Application settings loaded from environment variables via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration object for the image translation pipeline.

    All fields are populated from environment variables (or a ``.env`` file).
    Fields with defaults are optional; fields without defaults are required.

    Attributes:
        ANTHROPIC_API_KEY: Secret key for authenticating with the Anthropic API.
        LANGCHAIN_API_KEY: Secret key for LangSmith tracing and observability.
        LANGCHAIN_TRACING_V2: Enables LangSmith trace upload when ``"true"``.
        LANGCHAIN_PROJECT: Project name shown in the LangSmith dashboard.
        MAX_IMAGE_SIZE_MB: Maximum permitted upload size in megabytes.
        DEFAULT_TARGET_LANGUAGE: Fallback translation target when the caller
            does not specify one.
        EXTRACTION_CLAUDE_MODEL: Claude model identifier used by the extraction
            and translation node.
        EXTRACTION_MAX_TOKENS: Maximum number of tokens the extraction model
            may generate in a single response.
    """

    # Anthropic API key for Claude models
    ANTHROPIC_API_KEY: str

    # LangSmith API key for tracing and observability
    LANGCHAIN_API_KEY: str

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

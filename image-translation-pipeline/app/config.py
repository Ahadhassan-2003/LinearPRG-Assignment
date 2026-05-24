from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

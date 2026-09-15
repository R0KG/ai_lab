"""Application configuration loaded from environment variables and ``.env``."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for local, Docker, and AWS deployments."""

    app_name: str = "Enterprise AI Lab"
    environment: str = "development"
    debug: bool = False

    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/enterprise_ai_lab"
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = "embeddinggemma"
    embedding_dimension: int = 768

    # boto3 uses the standard AWS credential chain: environment variables,
    # AWS profile, container/task role, or instance role. No keys belong here.
    aws_region: str = "eu-central-1"
    aws_profile: str | None = None
    aws_bedrock_model_id: str = "amazon.nova-lite-v1:0"

    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = Field(default=5, ge=1, le=50)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the current process."""

    return Settings()

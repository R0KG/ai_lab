"""Tests for chat provider selection."""

from app.api.chat import build_llm_provider
from app.core.config import Settings
from app.llm.bedrock import BedrockLLMProvider
from app.llm.ollama import OllamaLLMProvider


def test_build_llm_provider_selects_ollama() -> None:
    settings = Settings(llm_backend="ollama")

    provider = build_llm_provider(settings)

    assert isinstance(provider, OllamaLLMProvider)


def test_build_llm_provider_selects_bedrock() -> None:
    settings = Settings(llm_backend="bedrock")

    provider = build_llm_provider(settings)

    assert isinstance(provider, BedrockLLMProvider)

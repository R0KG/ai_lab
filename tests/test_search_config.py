"""Tests for search configuration behavior."""

from app.api import search as search_api
from app.core.config import Settings


def test_search_uses_configured_top_k_when_limit_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        search_api,
        "get_settings",
        lambda: Settings(top_k=7),
    )

    assert search_api.resolve_search_limit(None) == 7


def test_explicit_search_limit_overrides_configured_top_k(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        search_api,
        "get_settings",
        lambda: Settings(top_k=7),
    )

    assert search_api.resolve_search_limit(3) == 3

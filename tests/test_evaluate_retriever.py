"""Tests for retrieval evaluation helpers."""

import json
from pathlib import Path

from scripts.evalueate_retriever import (
    calculate_document_recall,
    calculate_page_recall,
    evaluate_case,
    resolve_document_id,
)

DATASET_PATH = Path("datasets/rag_eval.json")


def test_evaluation_dataset_has_phase_five_cases() -> None:
    cases = json.loads(DATASET_PATH.read_text())

    assert len(cases) >= 20
    assert {
        "simple_lookup",
        "multi_document",
        "multi_page",
        "summarization",
        "structured_extraction",
        "german_question",
        "ambiguous",
        "unanswerable",
    } <= {case["category"] for case in cases}

    for case in cases:
        assert case["id"]
        assert case["question"]
        assert case["relevant_documents"]
        assert isinstance(case["relevant_pages"], list)
        assert isinstance(case["answerable"], bool)

        if case["category"] == "multi_document":
            assert "document_id" not in case
            assert len(case["relevant_document_checksums"]) >= 2
        else:
            assert "document_id" not in case
            assert len(case["document_checksum"]) == 64


def test_calculate_page_recall_counts_unique_expected_pages() -> None:
    score = calculate_page_recall(
        expected_pages={1, 2},
        retrieved_pages={1, 1, 2, 3},
    )

    assert score == 1.0


def test_calculate_page_recall_returns_partial_score() -> None:
    score = calculate_page_recall(
        expected_pages={1, 2},
        retrieved_pages={1, 3},
    )

    assert score == 0.5


def test_calculate_page_recall_skips_unanswerable_case() -> None:
    score = calculate_page_recall(
        expected_pages=set(),
        retrieved_pages={1, 2},
    )

    assert score is None


def test_calculate_document_recall_counts_expected_documents() -> None:
    score = calculate_document_recall(
        expected_document_ids={"document-a", "document-b"},
        retrieved_document_ids={"document-a", "document-c"},
    )

    assert score == 0.5


async def test_resolve_document_id_looks_up_checksum_through_api() -> None:
    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == (
            "/v1/documents/by-checksum/" + "a" * 64
        )
        return httpx.Response(200, json={"id": "current-database-id"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="http://test",
        transport=transport,
    ) as client:
        document_id = await resolve_document_id(client, "a" * 64, {})

    assert document_id == "current-database-id"


async def test_evaluate_case_resolves_checksum_for_search_and_chat() -> None:
    import httpx

    checksum = "a" * 64
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == f"/v1/documents/by-checksum/{checksum}":
            return httpx.Response(200, json={"id": "resolved-document-id"})
        if request.url.path == "/v1/search":
            assert request.url.params["document_id"] == "resolved-document-id"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"page_number": 1, "document_id": "resolved-document-id"}
                    ]
                },
            )
        if request.url.path == "/v1/chat":
            assert json.loads(request.read())["document_id"] == (
                "resolved-document-id"
            )
            return httpx.Response(200, json={"answer": "answer", "sources": []})
        raise AssertionError(f"Unexpected request: {request.url}")

    case = {
        "id": "q-test",
        "question": "A test question?",
        "category": "simple_lookup",
        "answerable": True,
        "expected_answer": "A test answer.",
        "relevant_pages": [1],
        "document_checksum": checksum,
    }

    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await evaluate_case(client, case, {})

    assert result["retrieved_document_ids"] == ["resolved-document-id"]
    assert result["page_recall_at_k"] == 1.0
    assert [request.url.path for request in requests] == [
        f"/v1/documents/by-checksum/{checksum}",
        "/v1/search",
        "/v1/chat",
    ]


async def test_evaluate_multi_document_case_resolves_each_checksum() -> None:
    import httpx

    checksums = ["a" * 64, "b" * 64]
    resolved_ids = ["document-one", "document-two"]

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/documents/by-checksum/"):
            checksum = request.url.path.rsplit("/", maxsplit=1)[-1]
            return httpx.Response(
                200,
                json={"id": resolved_ids[checksums.index(checksum)]},
            )
        if request.url.path == "/v1/search":
            assert "document_id" not in request.url.params
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"page_number": 1, "document_id": document_id}
                        for document_id in resolved_ids
                    ]
                },
            )
        if request.url.path == "/v1/chat":
            assert json.loads(request.read())["document_id"] is None
            return httpx.Response(200, json={"answer": "answer", "sources": []})
        raise AssertionError(f"Unexpected request: {request.url}")

    case = {
        "id": "q-multi-test",
        "question": "Compare two documents.",
        "category": "multi_document",
        "answerable": True,
        "expected_answer": "A comparison.",
        "relevant_pages": [],
        "relevant_document_checksums": checksums,
    }

    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await evaluate_case(client, case, {})

    assert result["expected_document_ids"] == resolved_ids
    assert result["document_recall_at_k"] == 1.0

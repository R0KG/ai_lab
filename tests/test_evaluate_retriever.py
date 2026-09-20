"""Tests for retrieval evaluation helpers."""

import json
from pathlib import Path

from scripts.evalueate_retriever import (
    calculate_document_recall,
    calculate_page_recall,
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
            assert case["document_id"] is None
            assert len(case["relevant_document_ids"]) >= 2
        else:
            assert case["document_id"]


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

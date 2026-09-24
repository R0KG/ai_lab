"""Evaluate retrieval quality against datasets/rag_eval.json."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

API_URL = "http://localhost:8000"
DATASET_PATH = Path("datasets/rag_eval.json")
RESULTS_PATH = Path("artifacts/rag_eval_results.json")
TOP_K = 5


def load_cases() -> list[dict[str, Any]]:
    """Load evaluation cases from JSON."""
    data = json.loads(DATASET_PATH.read_text())

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must contain a JSON array")

    return data


def calculate_page_recall(
    expected_pages: set[int],
    retrieved_pages: set[int],
) -> float | None:
    """Calculate page-level recall for one evaluation case."""

    if not expected_pages:
        return None

    matched_pages = expected_pages & retrieved_pages
    return len(matched_pages) / len(expected_pages)


def calculate_document_recall(
    expected_document_ids: set[str],
    retrieved_document_ids: set[str],
) -> float | None:
    """Calculate document-level recall for a multi-document case."""

    if not expected_document_ids:
        return None

    matched_documents = expected_document_ids & retrieved_document_ids
    return len(matched_documents) / len(expected_document_ids)


async def resolve_document_id(
    client: httpx.AsyncClient,
    checksum: str,
    cache: dict[str, str],
) -> str:
    """Resolve a stable dataset checksum to its current database UUID."""

    if checksum not in cache:
        response = await client.get(f"/v1/documents/by-checksum/{checksum}")
        response.raise_for_status()
        cache[checksum] = str(response.json()["id"])

    return cache[checksum]


async def evaluate_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    document_ids_by_checksum: dict[str, str],
) -> dict[str, Any]:
    """Evaluate retrieval and answer generation for one case."""

    expected_pages = {int(page) for page in case["relevant_pages"]}
    if case.get("document_checksum"):
        document_id = await resolve_document_id(
            client,
            case["document_checksum"],
            document_ids_by_checksum,
        )
        expected_document_ids = {document_id}
        explicit_document_ids: set[str] = set()
    else:
        document_ids = {
            await resolve_document_id(
                client,
                checksum,
                document_ids_by_checksum,
            )
            for checksum in case["relevant_document_checksums"]
        }
        document_id = None
        expected_document_ids = document_ids
        explicit_document_ids = document_ids

    search_params: dict[str, Any] = {
        "q": case["question"],
        "limit": TOP_K,
    }
    if document_id is not None:
        search_params["document_id"] = document_id

    search_started_at = time.perf_counter()
    search_response = await client.get(
        "/v1/search",
        params=search_params,
    )
    search_response.raise_for_status()
    search_payload = search_response.json()
    search_latency_ms = (time.perf_counter() - search_started_at) * 1000

    retrieved_pages = {
        int(result["page_number"])
        for result in search_payload["results"]
    }
    retrieved_document_ids = {
        str(result["document_id"])
        for result in search_payload["results"]
    }
    matched_pages = expected_pages & retrieved_pages
    matched_document_ids = expected_document_ids & retrieved_document_ids

    page_recall = calculate_page_recall(expected_pages, retrieved_pages)
    document_recall = calculate_document_recall(
        explicit_document_ids,
        retrieved_document_ids,
    )
    recall_at_k = (
        page_recall
        if page_recall is not None
        else document_recall
    )

    chat_started_at = time.perf_counter()
    chat_response = await client.post(
        "/v1/chat",
        json={
            "query": case["question"],
            "limit": TOP_K,
            "document_id": document_id,
        },
    )
    chat_response.raise_for_status()
    chat_payload = chat_response.json()
    chat_latency_ms = (time.perf_counter() - chat_started_at) * 1000

    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "answerable": case["answerable"],
        "expected_answer": case["expected_answer"],
        "generated_answer": chat_payload["answer"],
        "expected_pages": sorted(expected_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "matched_pages": sorted(matched_pages),
        "expected_document_ids": sorted(expected_document_ids),
        "retrieved_document_ids": sorted(retrieved_document_ids),
        "matched_document_ids": sorted(matched_document_ids),
        "page_recall_at_k": page_recall,
        "document_recall_at_k": document_recall,
        "recall_at_k": recall_at_k,
        "search_latency_ms": round(search_latency_ms, 2),
        "chat_latency_ms": round(chat_latency_ms, 2),
        "sources": chat_payload["sources"],
    }


async def main() -> None:
    """Evaluate all retrieval and generation cases."""

    cases = load_cases()
    results: list[dict[str, Any]] = []
    scores: list[float] = []

    async with httpx.AsyncClient(
        base_url=API_URL,
        timeout=60.0,
    ) as client:
        document_ids_by_checksum: dict[str, str] = {}
        for case in cases:
            result = await evaluate_case(
                client,
                case,
                document_ids_by_checksum,
            )
            results.append(result)

            recall = result["recall_at_k"]
            if isinstance(recall, float):
                scores.append(recall)
                recall_text = f"{recall:.2f}"
            else:
                recall_text = "n/a"

            print(
                f"{result['id']}: "
                f"Recall@{TOP_K}={recall_text}, "
                f"search={result['search_latency_ms']:.2f}ms, "
                f"chat={result['chat_latency_ms']:.2f}ms"
            )

    if not scores:
        raise ValueError("No answerable cases were evaluated")

    average_recall = sum(scores) / len(scores)
    report = {
        "dataset": str(DATASET_PATH),
        "top_k": TOP_K,
        "average_recall_at_k": round(average_recall, 4),
        "cases": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )

    print(f"Average Recall@{TOP_K}: {average_recall:.2f}")
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

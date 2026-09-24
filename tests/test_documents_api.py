"""Tests for document upload and checksum lookup endpoints."""

from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import documents as documents_api
from app.models.document import Document


def test_upload_stores_checksum_and_duplicate_upload_skips_ingestion(
    tmp_path,
    monkeypatch,
) -> None:
    checksum = "07dbe48c5da487890a8eeb3ec25e5aa88d1a490d6a696a77eca5c3719fdc4170"
    document = Document(
        id=uuid4(),
        name="resume.pdf",
        source_uri="uploads/resume.pdf",
        content_type="application/pdf",
        checksum=checksum,
        document_metadata={},
    )

    class FakeDocumentService:
        def __init__(self) -> None:
            self.stored: Document | None = None

        async def get_by_checksum(self, requested_checksum: str):
            assert requested_checksum == checksum
            return self.stored

        async def get_or_create_document(self, payload, requested_checksum):
            assert requested_checksum == checksum
            self.stored = document
            return document, True

    class FakeIngestionService:
        calls = 0

        async def ingest_pdf(self, document_id, file_path):
            self.calls += 1
            assert file_path.read_bytes() == b"same PDF bytes"
            return []

    service = FakeDocumentService()
    ingestion = FakeIngestionService()
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path)

    app = FastAPI()
    app.include_router(documents_api.router)
    app.dependency_overrides[documents_api.get_document_service] = lambda: service
    app.dependency_overrides[documents_api.get_ingestion_service] = lambda: ingestion

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post(
            "/v1/documents/upload",
            files={"file": ("resume.pdf", b"same PDF bytes", "application/pdf")},
        )
        second = client.post(
            "/v1/documents/upload",
            files={"file": ("renamed.pdf", b"same PDF bytes", "application/pdf")},
        )

    assert first.status_code == 201
    assert first.json()["checksum"] == checksum
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert ingestion.calls == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_document_can_be_resolved_by_checksum() -> None:
    checksum = "b" * 64
    document = SimpleNamespace(
        id=uuid4(),
        name="resume.pdf",
        source_uri="uploads/resume.pdf",
        content_type="application/pdf",
        checksum=checksum,
        document_metadata={},
    )

    class FakeDocumentService:
        async def get_by_checksum(self, requested_checksum: str):
            assert requested_checksum == checksum
            return document

    app = FastAPI()
    app.include_router(documents_api.router)
    app.dependency_overrides[documents_api.get_document_service] = (
        lambda: FakeDocumentService()
    )

    with TestClient(app) as client:
        response = client.get(f"/v1/documents/by-checksum/{checksum}")

    assert response.status_code == 200
    assert response.json()["checksum"] == checksum

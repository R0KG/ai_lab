"""Tests for document deduplication behavior."""

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.document import Document
from app.schemas.documents import DocumentCreate
from app.services.document_service import DocumentService


class FakeDocumentRepository:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.create_calls = 0

    async def get_by_checksum(self, checksum: str) -> Document | None:
        return self.documents.get(checksum)

    async def create(self, document: Document) -> Document:
        self.create_calls += 1
        assert document.checksum is not None
        self.documents[document.checksum] = document
        return document


@pytest.mark.asyncio
async def test_get_or_create_reuses_document_with_same_checksum() -> None:
    repository = FakeDocumentRepository()
    service = DocumentService(repository)  # type: ignore[arg-type]
    checksum = "a" * 64
    payload = DocumentCreate(name="resume.pdf")

    first, first_created = await service.get_or_create_document(
        payload,
        checksum,
    )
    second, second_created = await service.get_or_create_document(
        payload,
        checksum,
    )

    assert first.id == second.id
    assert first_created is True
    assert second_created is False
    assert repository.create_calls == 1
    assert first.checksum == checksum


@pytest.mark.asyncio
async def test_get_or_create_uses_unique_constraint_for_upload_race() -> None:
    repository = FakeDocumentRepository()
    checksum = "c" * 64
    winning_document = Document(
        id=uuid4(),
        name="resume.pdf",
        checksum=checksum,
    )

    async def create_with_concurrent_winner(document: Document) -> Document:
        repository.documents[checksum] = winning_document
        raise IntegrityError("insert", {}, Exception("unique constraint"))

    repository.create = create_with_concurrent_winner  # type: ignore[method-assign]
    service = DocumentService(repository)  # type: ignore[arg-type]

    document, created = await service.get_or_create_document(
        DocumentCreate(name="resume.pdf"),
        checksum,
    )

    assert document.id == winning_document.id
    assert created is False

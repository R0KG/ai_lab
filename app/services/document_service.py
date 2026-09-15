"""Document ingestion service."""

from uuid import UUID

from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.schemas.documents import DocumentCreate


class DocumentService:
    def __init__(self, repository: DocumentRepository) -> None:
        self.repository = repository

    async def create_document(self, data: DocumentCreate) -> Document:
        document = Document(
            name=data.name,
            source_uri=data.source_uri,
            content_type=data.content_type,
            document_metadata=data.document_metadata,
        )
        return await self.repository.create(document)

    async def get_document(
        self,
        document_id: UUID,
    ) -> Document | None:
        return await self.repository.get_by_id(document_id)

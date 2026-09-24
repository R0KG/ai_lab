"""Document ingestion service."""

from uuid import UUID

from sqlalchemy.exc import IntegrityError

from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.schemas.documents import DocumentCreate


class DocumentService:
    def __init__(self, repository: DocumentRepository) -> None:
        self.repository = repository

    async def create_document(
        self,
        data: DocumentCreate,
        checksum: str | None = None,
    ) -> Document:
        document = Document(
            name=data.name,
            source_uri=data.source_uri,
            content_type=data.content_type,
            checksum=checksum,
            document_metadata=data.document_metadata,
        )
        return await self.repository.create(document)

    async def get_by_checksum(self, checksum: str) -> Document | None:
        return await self.repository.get_by_checksum(checksum)

    async def get_or_create_document(
        self,
        data: DocumentCreate,
        checksum: str,
    ) -> tuple[Document, bool]:
        """Reuse an existing content record or create it exactly once."""

        existing = await self.get_by_checksum(checksum)
        if existing is not None:
            return existing, False

        try:
            return await self.create_document(data, checksum), True
        except IntegrityError:
            # The unique constraint resolves simultaneous uploads of the same
            # bytes. After rollback, return the winning request's row.
            existing = await self.get_by_checksum(checksum)
            if existing is None:
                raise
            return existing, False

    async def get_document(
        self,
        document_id: UUID,
    ) -> Document | None:
        return await self.repository.get_by_id(document_id)

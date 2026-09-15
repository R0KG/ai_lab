from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document


@dataclass
class DocumentRepository:
    session: AsyncSession

    async def create(self, document: Document) -> Document:

        self.session.add(document)

        try:
            await self.session.commit()
        except IntegrityError:
            await self.session.rollback()
            raise

        await self.session.refresh(document)
        return document

    async def get_by_id(self, document_id: UUID) -> Document | None:
        statement = select(Document).where(Document.id == document_id)

        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_checksum(
        self,
        checksum: str,
    ) -> Document | None:
        statement = select(Document).where(Document.checksum == checksum)

        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        offset: int = 0,
        limit: int = 50,
    ) -> list[Document]:
        statement = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(statement)

        return list(result.scalars().all())

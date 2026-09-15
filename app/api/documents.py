"""Document endpoints."""

from pathlib import Path
from shutil import copyfileobj
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.database import get_db
from app.embeddings.local import OllamaEmbeddingProvider
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.documents import DocumentCreate, DocumentResponse
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService

UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


router = APIRouter(
    prefix="/v1/documents",
    tags=["documents"],
)


def get_document_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentService:
    repository = DocumentRepository(db)
    return DocumentService(repository)


def get_ingestion_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> IngestionService:
    settings = get_settings()
    repository = ChunkRepository(db)
    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )

    return IngestionService(
        chunk_repository=repository,
        embedding_provider=embedding_provider,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_dimension=settings.embedding_dimension,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    payload: DocumentCreate,
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
) -> DocumentResponse:
    document = await service.create_document(payload)
    return DocumentResponse.model_validate(document)


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
    ingestion_service: Annotated[
        IngestionService,
        Depends(get_ingestion_service),
    ],
    file: UploadFile = File(...),
) -> DocumentResponse:
    original_name = file.filename or "uploaded-file"

    file_extension = Path(original_name).suffix.lower()
    stored_name = f"{uuid4()}{file_extension}"

    destination = UPLOAD_DIR / stored_name

    with destination.open("wb") as output_file:
        copyfileobj(file.file, output_file)

    payload = DocumentCreate(
        name=original_name,
        content_type=file.content_type,
        source_uri=f"uploads/{stored_name}",
    )

    document = await service.create_document(payload)

    await ingestion_service.ingest_pdf(
        document_id=document.id,
        file_path=destination,
    )

    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
async def get_document(
    document_id: UUID,
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
) -> DocumentResponse:
    document = await service.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return DocumentResponse.model_validate(document)

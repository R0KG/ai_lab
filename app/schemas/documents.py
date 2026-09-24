"""Document request and response schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    name: str
    source_uri: str | None = None
    content_type: str | None = None
    document_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    source_uri: str | None
    content_type: str | None
    checksum: str | None

    document_metadata: dict[str, Any]

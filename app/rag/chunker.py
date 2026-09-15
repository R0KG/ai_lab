"""Document text normalization and chunking."""

from dataclasses import dataclass

from app.rag.extractor import ExtractedText


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A text fragment with its source page and document-local index."""

    page_number: int
    chunk_index: int
    text: str


def chunk_text(
    pages: list[ExtractedText],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Split extracted page text into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    chunks: list[TextChunk] = []
    chunk_index = 0
    step = chunk_size - chunk_overlap

    for page in pages:
        words = page.text.split()

        for start in range(0, len(words), step):
            current_words = words[start : start + chunk_size]

            if not current_words:
                break

            chunks.append(
                TextChunk(
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=" ".join(current_words),
                )
            )
            chunk_index += 1

            if start + chunk_size >= len(words):
                break

    return chunks

"""Tests for document text chunking."""

import pytest

from app.rag.chunker import chunk_text
from app.rag.extractor import ExtractedText


def test_chunk_text_preserves_overlap_and_page_metadata() -> None:
    pages = [
        ExtractedText(
            page_number=1,
            text="one two three four five six seven eight",
        ),
        ExtractedText(page_number=2, text="nine ten"),
    ]

    chunks = chunk_text(pages, chunk_size=4, chunk_overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight",
        "nine ten",
    ]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (4, -1), (4, 4), (4, 5)],
)
def test_chunk_text_rejects_invalid_parameters(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    pages = [ExtractedText(page_number=1, text="one two three")]

    with pytest.raises(ValueError):
        chunk_text(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

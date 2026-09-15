from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True, slots=True)
class ExtractedText:
    page_number: int
    text: str


def extract_pdf(path: Path) -> list[ExtractedText]:
    pages: list[ExtractedText] = []

    with pymupdf.open(path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1
            raw_text = page.get_text("text", sort=True)

            if not isinstance(raw_text, str):
                raise TypeError("Expected plain text from PDF page")

            text = raw_text.strip()

            if not text:
                continue

            pages.append(
                ExtractedText(
                    page_number=page_number,
                    text=text,
                )
            )
    return pages

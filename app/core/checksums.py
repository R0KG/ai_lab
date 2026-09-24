"""Helpers for content-based document identity."""

from hashlib import sha256
from typing import BinaryIO


def sha256_file(file_obj: BinaryIO) -> str:
    """Hash a file-like object without changing its current position."""

    original_position = file_obj.tell()
    digest = sha256()

    try:
        file_obj.seek(0)
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    finally:
        file_obj.seek(original_position)

    return digest.hexdigest()

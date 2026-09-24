"""Tests for file checksum helpers."""

from hashlib import sha256
from io import BytesIO

from app.core.checksums import sha256_file


def test_sha256_file_hashes_content_and_preserves_stream_position() -> None:
    file_obj = BytesIO(b"sample PDF bytes")
    file_obj.seek(7)

    checksum = sha256_file(file_obj)

    assert checksum == sha256(b"sample PDF bytes").hexdigest()
    assert file_obj.tell() == 7

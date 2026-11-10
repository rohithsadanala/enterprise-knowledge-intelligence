"""Document loaders: PDF, DOCX, Markdown and plain text.

Each loader returns a :class:`~eki.models.Document` with the full text
plus metadata (source path, file type, page count where available).
Only the file types listed in :data:`SUPPORTED_EXTENSIONS` are accepted;
anything else raises :class:`UnsupportedFileTypeError`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from eki.models import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file's extension is not in SUPPORTED_EXTENSIONS."""


def _doc_id(source: str, content: str) -> str:
    """Stable id from the source path + a content hash."""
    digest = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    stem = Path(source).stem or "document"
    return f"{stem}-{digest}"


def load_text(path: str | Path) -> Document:
    """Load a plain-text or Markdown file."""
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        id=_doc_id(str(path), content),
        title=path.stem,
        source=str(path),
        content=content,
        metadata={"file_type": path.suffix.lower(), "size_bytes": path.stat().st_size},
    )


def load_pdf(path: str | Path) -> Document:
    """Load a PDF, concatenating page text and recording page numbers.

    Page markers are embedded as ``\\n\\n[Page N]\\n`` lines so chunk
    metadata can later recover which page a chunk came from.
    """
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n\n[Page {i}]\n{text}")
    content = "".join(pages).strip()
    return Document(
        id=_doc_id(str(path), content),
        title=path.stem,
        source=str(path),
        content=content,
        metadata={
            "file_type": ".pdf",
            "pages": len(reader.pages),
            "size_bytes": path.stat().st_size,
        },
    )


def load_docx(path: str | Path) -> Document:
    """Load a .docx file, joining paragraphs (and table cells)."""
    from docx import Document as DocxDocument

    path = Path(path)
    doc = DocxDocument(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells if cell.text.strip())
    content = "\n".join(parts)
    return Document(
        id=_doc_id(str(path), content),
        title=path.stem,
        source=str(path),
        content=content,
        metadata={"file_type": ".docx", "size_bytes": path.stat().st_size},
    )


def load_document(path: str | Path) -> Document:
    """Dispatch to the right loader based on file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    return load_text(path)


def load_directory(
    directory: str | Path, recursive: bool = True
) -> list[Document]:
    """Load every supported document under a directory.

    Files that fail to load are skipped (with a warning on stderr)
    rather than aborting the whole ingest run.
    """
    import sys

    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    documents: list[Document] = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            documents.append(load_document(path))
        except Exception as exc:  # noqa: BLE001 - ingest must be resilient
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
    return documents


def file_title(path: str | Path) -> str:
    """Human-friendly title for a file path."""
    return Path(path).stem.replace("_", " ").replace("-", " ").title()


def ensure_data_dir(data_dir: str | Path) -> Path:
    """Create the data directory (and an ``uploads`` subdir) if needed."""
    path = Path(data_dir)
    (path / "uploads").mkdir(parents=True, exist_ok=True)
    return path

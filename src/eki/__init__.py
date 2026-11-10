"""Enterprise Knowledge Intelligence — public package API."""

from eki.config import Settings
from eki.models import Chunk, Citation, Document, QueryResult
from eki.pipeline import KnowledgePipeline

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "Citation",
    "Document",
    "KnowledgePipeline",
    "QueryResult",
    "Settings",
    "__version__",
]

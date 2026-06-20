"""Load raw text from PDF, Markdown, plain-text, and HTML sources."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawDocument:
    name: str
    content: str
    source_uri: str | None = None
    mime_type: str = "text/plain"
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Returns a RawDocument from a file path or URL string."""

    def load(self, source: str | Path) -> RawDocument:
        path = Path(source)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix in (".md", ".markdown"):
            return self._load_text(path, mime_type="text/markdown")
        elif suffix == ".html":
            return self._load_html(path)
        else:
            return self._load_text(path)

    def load_text(self, text: str, name: str = "inline") -> RawDocument:
        return RawDocument(name=name, content=text)

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    def _load_text(self, path: Path, mime_type: str = "text/plain") -> RawDocument:
        content = path.read_text(encoding="utf-8", errors="replace")
        return RawDocument(
            name=path.name,
            content=content,
            source_uri=str(path),
            mime_type=mime_type,
        )

    def _load_pdf(self, path: Path) -> RawDocument:
        try:
            import pypdf  # optional dependency

            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            content = "\n\n".join(pages)
        except ImportError:
            raise RuntimeError(
                "pypdf is required for PDF loading: pip install pypdf"
            )
        return RawDocument(
            name=path.name,
            content=content,
            source_uri=str(path),
            mime_type="application/pdf",
            metadata={"page_count": len(pages) if "pages" in dir() else 0},
        )

    def _load_html(self, path: Path) -> RawDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Strip tags; keep text nodes separated by newlines
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s{2,}", "\n", text).strip()
        return RawDocument(
            name=path.name,
            content=text,
            source_uri=str(path),
            mime_type="text/html",
        )

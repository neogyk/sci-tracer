from docling.document_converter import DocumentConverter


def doc_reader(source: str) -> str:
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document.export_to_markdown()


def doc_reader_pages(source: str) -> list[str]:
    """Convert a document and return one Markdown string per page.

    Uses docling's native page structure so each element is emitted on the
    page it physically belongs to.  Falls back to blank-line splitting of the
    full markdown when page data is unavailable (e.g. plain-text inputs).
    """
    converter = DocumentConverter()
    result    = converter.convert(source)
    doc       = result.document

    # docling >= 2.x exposes pages via doc.pages (dict page_no -> PageItem)
    pages: list[str] = []
    if hasattr(doc, "pages") and doc.pages:
        for page_no in sorted(doc.pages.keys()):
            page_md = doc.export_to_markdown(page_no=page_no) if _supports_page_arg(doc) \
                      else _extract_page_text(doc, page_no)
            if page_md.strip():
                pages.append(page_md.strip())
    else:
        # Fallback: split the full markdown on page-break markers inserted by docling
        full_md = doc.export_to_markdown()
        for marker in ("<!-- PageBreak -->", "\f", "\n---\n"):
            if marker in full_md:
                pages = [p.strip() for p in full_md.split(marker) if p.strip()]
                break
        else:
            # Last resort: treat each paragraph as its own page chunk
            pages = [p.strip() for p in full_md.split("\n\n") if p.strip()]

    return pages


def _supports_page_arg(doc) -> bool:
    """Return True if export_to_markdown accepts a page_no keyword argument."""
    import inspect
    try:
        sig = inspect.signature(doc.export_to_markdown)
        return "page_no" in sig.parameters
    except (TypeError, ValueError):
        return False


def _extract_page_text(doc, page_no: int) -> str:
    """Extract text for a single page by filtering doc elements by page number."""
    lines: list[str] = []
    for item, _ in doc.iterate_items():
        prov = getattr(item, "prov", None) or []
        for p in prov:
            if getattr(p, "page_no", None) == page_no:
                text = getattr(item, "text", None) or getattr(item, "orig", "")
                if text:
                    lines.append(text)
                break
    return "\n\n".join(lines)

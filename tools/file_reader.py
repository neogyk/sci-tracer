from docling.document_converter import DocumentConverter


def doc_reader(source: str):
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document.export_to_markdown()

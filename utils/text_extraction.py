"""Extract text from PDF, DOCX, and TXT CV files."""

from pathlib import Path
from typing import BinaryIO

import pdfplumber
from docx import Document


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """Extract text from a PDF file-like object."""
    text_parts = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_obj: BinaryIO) -> str:
    """Extract text from a DOCX file-like object."""
    document = Document(file_obj)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs).strip()


def extract_text_from_txt(file_obj: BinaryIO) -> str:
    """Extract text from a TXT file-like object."""
    raw = file_obj.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore").strip()
    return str(raw).strip()


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Extract text from a Streamlit uploaded file."""
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)

    if suffix == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    if suffix == ".docx":
        return extract_text_from_docx(uploaded_file)
    if suffix == ".txt":
        return extract_text_from_txt(uploaded_file)

    raise ValueError(f"Unsupported file type: {suffix}. Please upload PDF, DOCX, or TXT.")


"""
Document processing module:
- PDF parsing
- Text chunking for embeddings
"""

import pdfplumber
from io import BytesIO
from typing import List
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def parse_pdf(file_path=None, file_bytes=None) -> str:
    """
    Extract text from a PDF.

    Args:
        file_path: Path to a PDF on disk.
        file_bytes: BytesIO for in‑memory PDF (e.g. Streamlit upload).

    Returns:
        Extracted text from the PDF.

    Raises:
        ValueError: If no input is provided or no text can be extracted.
        Exception: For underlying parsing errors.
    """
    if file_path is None and file_bytes is None:
        raise ValueError("Either file_path or file_bytes must be provided")

    full_text = ""

    try:
        pdf_file = (
            pdfplumber.open(file_bytes)
            if file_bytes is not None
            else pdfplumber.open(file_path)
        )

        for page in pdf_file.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"

        pdf_file.close()

        if not full_text.strip():
            raise ValueError(
                "No text could be extracted from the PDF. "
                "It might be image-based or corrupted."
            )

        return full_text.strip()

    except Exception as e:
        raise Exception(f"Error parsing PDF: {str(e)}")


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split raw text into overlapping chunks suitable for embeddings.

    Args:
        text: Full document text.
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters of overlap between chunks.

    Returns:
        List of LangChain Document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return text_splitter.create_documents([text])


def parse_pdf_from_streamlit(uploaded_file) -> str:
    """
    Extract text from a Streamlit UploadedFile (PDF).

    Args:
        uploaded_file: Object returned by st.file_uploader.

    Returns:
        Extracted text from the PDF.
    """
    file_bytes = BytesIO(uploaded_file.read())
    file_bytes.seek(0)
    return parse_pdf(file_bytes=file_bytes)


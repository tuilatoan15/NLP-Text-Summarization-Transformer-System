"""Document loaders for TXT, PDF, DOCX, and OCR fallbacks."""

from loaders.docx_loader import DOCXLoader
from loaders.pdf_loader import PDFLoader
from loaders.txt_loader import TXTLoader

__all__ = ["DOCXLoader", "PDFLoader", "TXTLoader"]

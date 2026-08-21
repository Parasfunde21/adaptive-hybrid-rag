from pathlib import Path

from pypdf import PdfReader
from docx import Document


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


class DocumentExtractor:
    """
    Extract plain text from user-uploaded documents.

    Supported formats:
    - PDF
    - DOCX
    - TXT
    """

    def extract(self, file_path: str) -> str:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if extension == ".pdf":
            return self._extract_pdf(path)

        if extension == ".docx":
            return self._extract_docx(path)

        if extension == ".txt":
            return self._extract_txt(path)

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    def _extract_pdf(self, path: Path) -> str:

        reader = PdfReader(str(path))

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            text = text.strip()

            if text:
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        return "\n\n".join(pages)

    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    def _extract_docx(self, path: Path) -> str:

        document = Document(str(path))

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    # -----------------------------------------------------
    # TXT
    # -----------------------------------------------------

    def _extract_txt(self, path: Path) -> str:

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()


document_extractor = DocumentExtractor()
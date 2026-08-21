class DocumentChunker:
    """
    Split extracted document text into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100
    ):

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str):

        if not text or not text.strip():
            return []

        text = text.strip()

        chunks = []

        start = 0

        text_length = len(text)

        while start < text_length:

            end = min(
                start + self.chunk_size,
                text_length
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - self.overlap

        return chunks


document_chunker = DocumentChunker(
    chunk_size=500,
    overlap=100
)
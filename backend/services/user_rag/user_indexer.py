from pathlib import Path
import hashlib
import re

import chromadb

from config.settings import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
)

from services.ingestion.document_extractor import (
    document_extractor,
)

from services.ingestion.chunker import (
    document_chunker,
)

from sentence_transformers import SentenceTransformer


# ============================================================
# User Document Indexer
# ============================================================

class UserDocumentIndexer:

    def __init__(self):

        print(
            f"Loading user-RAG embedding model: "
            f"{EMBEDDING_MODEL}"
        )

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        # The embedding model is already available locally.
        #
        # local_files_only=True prevents SentenceTransformer
        # from contacting Hugging Face during every startup.
        # ----------------------------------------------------

        try:

            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                local_files_only=True
            )

        except Exception as exc:

            raise RuntimeError(
                "Could not load the local embedding model "
                f"'{EMBEDDING_MODEL}'. "
                "Make sure the model is already downloaded "
                "and available in the local Hugging Face cache."
            ) from exc

        print(
            "User-RAG embedding model loaded successfully."
        )

        # ----------------------------------------------------
        # ChromaDB
        # ----------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

    # ========================================================
    # Collection
    # ========================================================

    def _get_collection(
        self,
        user_id: str
    ):

        safe_user_id = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            str(user_id)
        )

        collection_name = (
            f"user_rag_{safe_user_id}"
        )

        return self.client.get_or_create_collection(
            name=collection_name
        )

    # ========================================================
    # File ID
    # ========================================================

    def _file_id(
        self,
        file_path: str
    ):

        path = Path(file_path)

        content = path.read_bytes()

        return hashlib.sha256(
            content
        ).hexdigest()[:16]

    # ========================================================
    # Index Document
    # ========================================================

    def index_document(
        self,
        user_id: str,
        file_path: str,
        original_filename: str | None = None,
    ):

        path = Path(file_path)
        
        display_filename = (
            Path(original_filename).name
            if original_filename
            else path.name
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not path.exists():

            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Path is not a file: {path}"
            )

        if path.stat().st_size == 0:

            raise ValueError(
                f"File is empty: {path.name}"
            )

        user_id = str(
            user_id
        ).strip()

        if not user_id:

            raise ValueError(
                "user_id cannot be empty."
            )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("USER DOCUMENT INDEXING")
        print("=" * 70)

        print(
            f"User      : {user_id}"
        )

        print(
            f"File      : {path.name}"
        )

        print(
            f"Type      : "
            f"{path.suffix.lower() or 'unknown'}"
        )

        print(
            f"Size      : "
            f"{path.stat().st_size} bytes"
        )

        # ----------------------------------------------------
        # File ID
        # ----------------------------------------------------

        file_id = self._file_id(
            str(path)
        )

        print(
            f"File ID   : {file_id}"
        )

        # ----------------------------------------------------
        # Extract Text
        # ----------------------------------------------------

        print(
            "Extracting text..."
        )

        text = document_extractor.extract(
            str(path)
        )

        if text is None:

            raise ValueError(
                "Document extractor returned None."
            )

        text = str(
            text
        ).strip()

        if not text:

            raise ValueError(
                "No text could be extracted "
                f"from {path.name}"
            )

        print(
            f"Extracted characters: "
            f"{len(text)}"
        )

        # ----------------------------------------------------
        # Chunk
        # ----------------------------------------------------

        print(
            "Creating chunks..."
        )

        chunks = document_chunker.chunk(
            text
        )

        if not chunks:

            raise ValueError(
                "Document produced zero chunks."
            )

        # Remove invalid chunks.

        chunks = [
            str(chunk).strip()
            for chunk in chunks
            if chunk is not None
            and str(chunk).strip()
        ]

        if not chunks:

            raise ValueError(
                "All generated chunks are empty."
            )

        print(
            f"Chunks created: "
            f"{len(chunks)}"
        )

        # ----------------------------------------------------
        # Collection
        # ----------------------------------------------------

        collection = self._get_collection(
            user_id
        )

        # ----------------------------------------------------
        # Existing File
        # ----------------------------------------------------
        #
        # If this exact file already exists, remove its
        # previous chunks before indexing it again.
        #
        # This prevents duplicate chunks.
        # ----------------------------------------------------

        try:

            existing = collection.get(
                where={
                    "file_id": file_id
                }
            )

            existing_ids = (
                existing.get(
                    "ids",
                    []
                )
                if existing
                else []
            )

            if existing_ids:

                print(
                    "Existing file detected."
                )

                print(
                    f"Removing "
                    f"{len(existing_ids)} "
                    f"existing chunks..."
                )

                collection.delete(
                    ids=existing_ids
                )

        except Exception as exc:

            print(
                "Warning: existing-file check "
                f"failed: {exc}"
            )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        print(
            "Generating embeddings..."
        )

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=False,
            show_progress_bar=True
        )

        # ----------------------------------------------------
        # IDs + Metadata
        # ----------------------------------------------------

        ids = []

        metadatas = []

        for index, chunk in enumerate(
            chunks
        ):

            chunk_id = (
                f"{file_id}_{index}"
            )

            ids.append(
                chunk_id
            )

            metadatas.append(
                {
                    "user_id":
                        user_id,

                    "file_id":
                        file_id,

                    "file_name":
                        display_filename,

                    "chunk_index":
                        index,

                    "document_type":
                        path.suffix.lower(),

                    "source":
                        "user_upload",

                    "document_id":
                        file_id,

                    "chunk_id":
                        chunk_id,
                }
            )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        print(
            "Storing chunks in ChromaDB..."
        )

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        # ----------------------------------------------------
        # Verification
        # ----------------------------------------------------

        collection_count = (
            collection.count()
        )

        print()
        print(
            f"Indexed {len(chunks)} chunks."
        )

        print(
            f"Collection: "
            f"{collection.name}"
        )

        print(
            f"Collection count: "
            f"{collection_count}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "success":
                True,

            "user_id":
                user_id,

            "file_id":
                file_id,

            "file_name":
                display_filename,

            "document_type":
                path.suffix.lower(),

            "characters":
                len(text),

            "chunks":
                len(chunks),

            "collection":
                collection.name,

            "collection_count":
                collection_count,
        }


# ============================================================
# Singleton
# ============================================================

user_document_indexer = (
    UserDocumentIndexer()
)
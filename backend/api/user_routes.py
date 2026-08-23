import shutil
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from api.auth import get_current_user

from services.user_rag.user_indexer import (
    user_document_indexer,
)


router = APIRouter(
    prefix="/user",
    tags=["User RAG"],
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


# ============================================================
# Upload Document
# ============================================================

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A file must be provided.",
        )

    # --------------------------------------------------------
    # Preserve the ORIGINAL filename.
    # --------------------------------------------------------

    original_filename = Path(
        file.filename
    ).name

    extension = (
        Path(original_filename)
        .suffix
        .lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{extension}'. "
                f"Supported types: "
                f"{sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Create temporary file for processing.
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temp_path = Path(
                temp_file.name
            )

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

        # ----------------------------------------------------
        # Index document.
        #
        # IMPORTANT:
        # file_path       = temporary physical file
        # original_filename = actual uploaded filename
        # ----------------------------------------------------

        result = (
            user_document_indexer.index_document(
                user_id=user["id"],
                file_path=str(temp_path),
                original_filename=original_filename,
            )
        )

        # ----------------------------------------------------
        # Return original filename to frontend.
        # ----------------------------------------------------

        result["uploaded_file_name"] = (
            original_filename
        )

        result["file_name"] = (
            original_filename
        )

        return {
            "success": True,
            "user_id": user["id"],
            **result,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary processing file.
        # ----------------------------------------------------

        if temp_path is not None:

            try:

                temp_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass


# ============================================================
# List User Documents
# ============================================================

@router.get("/documents")
def list_documents(
    user=Depends(get_current_user),
):

    try:

        collection = (
            user_document_indexer._get_collection(
                user["id"]
            )
        )

        results = collection.get(
            include=["metadatas"]
        )

        metadatas = (
            results.get(
                "metadatas",
                []
            )
            or []
        )

        documents = {}

        # ----------------------------------------------------
        # Build unique document list.
        # ----------------------------------------------------

        for metadata in metadatas:

            metadata = metadata or {}

            file_id = metadata.get(
                "file_id"
            )

            if not file_id:
                continue

            documents[file_id] = {
                "file_id":
                    file_id,

                "file_name":
                    metadata.get(
                        "file_name"
                    ),

                "document_type":
                    metadata.get(
                        "document_type"
                    ),

                "chunks":
                    0,
            }

        # ----------------------------------------------------
        # Count chunks.
        # ----------------------------------------------------

        for metadata in metadatas:

            metadata = metadata or {}

            file_id = metadata.get(
                "file_id"
            )

            if file_id in documents:

                documents[file_id]["chunks"] += 1

        return {
            "success": True,
            "user_id": user["id"],
            "documents": list(
                documents.values()
            ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import get_current_user

from services.conversation_service import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    list_conversations,
    update_title,
)

from services.user_rag.user_pipeline import (
    user_rag_pipeline,
)


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


# ============================================================
# Schemas
# ============================================================

class CreateConversationRequest(BaseModel):

    title: str = Field(
        default="New Conversation",
        max_length=100,
    )


class SendMessageRequest(BaseModel):

    query: str = Field(
        min_length=1,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=20,
    )


class RenameConversationRequest(BaseModel):

    title: str = Field(
        min_length=1,
        max_length=100,
    )


# ============================================================
# Create
# ============================================================

@router.post("")
def create_new_conversation(
    request: CreateConversationRequest,
    user=Depends(get_current_user),
):

    conversation = create_conversation(
        user_id=user["id"],
        title=request.title,
    )

    return {
        "success": True,
        "conversation": conversation,
    }


# ============================================================
# List
# ============================================================

@router.get("")
def get_user_conversations(
    user=Depends(get_current_user),
):

    return {
        "success": True,
        "conversations": list_conversations(
            user["id"]
        ),
    }


# ============================================================
# Get Conversation
# ============================================================

@router.get("/{conversation_id}")
def get_user_conversation(
    conversation_id: str,
    user=Depends(get_current_user),
):

    conversation = get_conversation(
        user["id"],
        conversation_id,
    )

    if conversation is None:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    messages = get_messages(
        user["id"],
        conversation_id,
    )

    return {
        "success": True,
        "conversation": conversation,
        "messages": messages,
    }


# ============================================================
# Send Message
# ============================================================

@router.post("/{conversation_id}/messages")
def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    user=Depends(get_current_user),
):

    conversation = get_conversation(
        user["id"],
        conversation_id,
    )

    if conversation is None:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    query = request.query.strip()

    # Save user message first.
    add_message(
        user_id=user["id"],
        conversation_id=conversation_id,
        role="user",
        content=query,
    )

    try:

        result = user_rag_pipeline.answer(
            user_id=user["id"],
            query=query,
            retrieval_candidate_k=request.top_k,
            rerank_top_k=5,
        )

        answer = result.get(
            "answer",
            "",
        )

        add_message(
            user_id=user["id"],
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )

        # Automatically name an untouched conversation.
        if (
            conversation["title"]
            == "New Conversation"
        ):

            title = query[:60]

            if len(query) > 60:
                title += "..."

            update_title(
                user_id=user["id"],
                conversation_id=conversation_id,
                title=title,
            )

        return {
            "success": True,
            "conversation_id": conversation_id,
            "answer": answer,
            "rag": result,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# Rename
# ============================================================

@router.patch("/{conversation_id}")
def rename_conversation(
    conversation_id: str,
    request: RenameConversationRequest,
    user=Depends(get_current_user),
):

    conversation = get_conversation(
        user["id"],
        conversation_id,
    )

    if conversation is None:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    update_title(
        user_id=user["id"],
        conversation_id=conversation_id,
        title=request.title,
    )

    return {
        "success": True,
    }


# ============================================================
# Delete
# ============================================================

@router.delete("/{conversation_id}")
def remove_conversation(
    conversation_id: str,
    user=Depends(get_current_user),
):

    deleted = delete_conversation(
        user_id=user["id"],
        conversation_id=conversation_id,
    )

    if not deleted:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {
        "success": True,
    }
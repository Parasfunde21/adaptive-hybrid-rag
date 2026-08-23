import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


# ============================================================
# Database
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_DIR = BASE_DIR / "data"

DB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DB_PATH = DB_DIR / "rag_app.db"


# ============================================================
# Connection
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# Initialize Database
# ============================================================

def _initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (

            id TEXT PRIMARY KEY,

            user_id TEXT NOT NULL,

            title TEXT NOT NULL DEFAULT 'New Conversation',

            created_at TEXT NOT NULL,

            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (

            id TEXT PRIMARY KEY,

            user_id TEXT NOT NULL,

            conversation_id TEXT NOT NULL,

            role TEXT NOT NULL,

            content TEXT NOT NULL,

            created_at TEXT NOT NULL,

            FOREIGN KEY (
                conversation_id
            )
            REFERENCES conversations(id)
            ON DELETE CASCADE
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_conversations_user
        ON conversations(user_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_messages_conversation
        ON messages(conversation_id)
        """
    )

    connection.commit()

    connection.close()


_initialize_database()


# ============================================================
# Helpers
# ============================================================

def _now():

    return datetime.utcnow().isoformat()


def _row_to_dict(row):

    if row is None:
        return None

    return dict(row)


# ============================================================
# Create Conversation
# ============================================================

def create_conversation(
    user_id,
    title="New Conversation",
):

    conversation_id = uuid.uuid4().hex

    now = _now()

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO conversations (
            id,
            user_id,
            title,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            user_id,
            title.strip()
            or "New Conversation",
            now,
            now,
        ),
    )

    connection.commit()

    row = connection.execute(
        """
        SELECT
            id,
            user_id,
            title,
            created_at,
            updated_at
        FROM conversations
        WHERE id = ?
        AND user_id = ?
        """,
        (
            conversation_id,
            user_id,
        ),
    ).fetchone()

    connection.close()

    return _row_to_dict(row)


# ============================================================
# List Conversations
# ============================================================

def list_conversations(
    user_id,
):

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            user_id,
            title,
            created_at,
            updated_at
        FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (
            user_id,
        ),
    ).fetchall()

    connection.close()

    return [
        _row_to_dict(row)
        for row in rows
    ]


# ============================================================
# Get Conversation
# ============================================================

def get_conversation(
    user_id,
    conversation_id,
):

    connection = get_connection()

    row = connection.execute(
        """
        SELECT
            id,
            user_id,
            title,
            created_at,
            updated_at
        FROM conversations
        WHERE id = ?
        AND user_id = ?
        """,
        (
            conversation_id,
            user_id,
        ),
    ).fetchone()

    connection.close()

    return _row_to_dict(row)


# ============================================================
# Add Message
# ============================================================

def add_message(
    user_id,
    conversation_id,
    role,
    content,
):

    # --------------------------------------------------------
    # Verify conversation belongs to user
    # --------------------------------------------------------

    connection = get_connection()

    conversation = connection.execute(
        """
        SELECT id
        FROM conversations
        WHERE id = ?
        AND user_id = ?
        """,
        (
            conversation_id,
            user_id,
        ),
    ).fetchone()

    if conversation is None:

        connection.close()

        raise ValueError(
            "Conversation not found."
        )

    # --------------------------------------------------------
    # Insert message
    # --------------------------------------------------------

    message_id = uuid.uuid4().hex

    now = _now()

    connection.execute(
        """
        INSERT INTO messages (
            id,
            user_id,
            conversation_id,
            role,
            content,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            user_id,
            conversation_id,
            role,
            content,
            now,
        ),
    )

    # --------------------------------------------------------
    # Update conversation timestamp
    # --------------------------------------------------------

    connection.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE id = ?
        AND user_id = ?
        """,
        (
            now,
            conversation_id,
            user_id,
        ),
    )

    connection.commit()

    row = connection.execute(
        """
        SELECT
            id,
            user_id,
            conversation_id,
            role,
            content,
            created_at
        FROM messages
        WHERE id = ?
        """,
        (
            message_id,
        ),
    ).fetchone()

    connection.close()

    return _row_to_dict(row)


# ============================================================
# Get Messages
# ============================================================

def get_messages(
    user_id,
    conversation_id,
):

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            user_id,
            conversation_id,
            role,
            content,
            created_at
        FROM messages
        WHERE conversation_id = ?
        AND user_id = ?
        ORDER BY created_at ASC
        """,
        (
            conversation_id,
            user_id,
        ),
    ).fetchall()

    connection.close()

    return [
        _row_to_dict(row)
        for row in rows
    ]


# ============================================================
# Update Title
# ============================================================

def update_title(
    user_id,
    conversation_id,
    title,
):

    connection = get_connection()

    cursor = connection.execute(
        """
        UPDATE conversations
        SET
            title = ?,
            updated_at = ?
        WHERE id = ?
        AND user_id = ?
        """,
        (
            title.strip()
            or "New Conversation",
            _now(),
            conversation_id,
            user_id,
        ),
    )

    connection.commit()

    updated = (
        cursor.rowcount > 0
    )

    connection.close()

    return updated


# ============================================================
# Delete Conversation
# ============================================================

def delete_conversation(
    user_id,
    conversation_id,
):

    connection = get_connection()

    # --------------------------------------------------------
    # Delete messages first
    # --------------------------------------------------------

    connection.execute(
        """
        DELETE FROM messages
        WHERE conversation_id = ?
        AND user_id = ?
        """,
        (
            conversation_id,
            user_id,
        ),
    )

    # --------------------------------------------------------
    # Delete conversation
    # --------------------------------------------------------

    cursor = connection.execute(
        """
        DELETE FROM conversations
        WHERE id = ?
        AND user_id = ?
        """,
        (
            conversation_id,
            user_id,
        ),
    )

    connection.commit()

    deleted = (
        cursor.rowcount > 0
    )

    connection.close()

    return deleted
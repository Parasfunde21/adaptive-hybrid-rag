import sqlite3
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

APP_DB = DATA_DIR / "app.db"
RAG_DB = DATA_DIR / "rag_app.db"


# ============================================================
# Main migration
# ============================================================

def migrate():

    print("=" * 70)
    print("DATABASE MIGRATION")
    print("=" * 70)

    if not APP_DB.exists():
        raise FileNotFoundError(
            f"Authentication database not found: {APP_DB}"
        )

    if not RAG_DB.exists():
        raise FileNotFoundError(
            f"RAG database not found: {RAG_DB}"
        )

    connection = sqlite3.connect(APP_DB)

    try:

        connection.row_factory = sqlite3.Row

        # ----------------------------------------------------
        # Enable foreign keys
        # ----------------------------------------------------

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        # ----------------------------------------------------
        # Add email verification fields
        # ----------------------------------------------------

        user_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

        if "email_verified" not in user_columns:

            print("Adding email_verified column...")

            connection.execute(
                """
                ALTER TABLE users
                ADD COLUMN email_verified
                INTEGER NOT NULL DEFAULT 0
                """
            )

        if "last_login" not in user_columns:

            print("Adding last_login column...")

            connection.execute(
                """
                ALTER TABLE users
                ADD COLUMN last_login
                TIMESTAMP
                """
            )

        # ----------------------------------------------------
        # Email verification tokens
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            email_verification_tokens (

                token TEXT PRIMARY KEY,

                user_id TEXT NOT NULL,

                expires_at TIMESTAMP NOT NULL,

                used INTEGER NOT NULL DEFAULT 0,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Password reset tokens
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            password_reset_tokens (

                token TEXT PRIMARY KEY,

                user_id TEXT NOT NULL,

                expires_at TIMESTAMP NOT NULL,

                used INTEGER NOT NULL DEFAULT 0,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Conversations
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (

                id TEXT PRIMARY KEY,

                user_id TEXT NOT NULL,

                title TEXT NOT NULL,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                updated_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Messages
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (

                id TEXT PRIMARY KEY,

                user_id TEXT NOT NULL,

                conversation_id TEXT NOT NULL,

                role TEXT NOT NULL,

                content TEXT NOT NULL,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE,

                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Documents
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (

                id TEXT PRIMARY KEY,

                user_id TEXT NOT NULL,

                file_name TEXT NOT NULL,

                document_type TEXT,

                chunks INTEGER DEFAULT 0,

                collection_name TEXT,

                uploaded_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )

        connection.commit()

        print()
        print("Authentication tables ready.")
        print("Conversation tables ready.")
        print("Message tables ready.")
        print("Document table ready.")

        # ====================================================
        # Copy old conversation data
        # ====================================================

        old_db = sqlite3.connect(RAG_DB)
        old_db.row_factory = sqlite3.Row

        old_conversations = old_db.execute(
            """
            SELECT
                id,
                user_id,
                title,
                created_at,
                updated_at
            FROM conversations
            """
        ).fetchall()

        old_messages = old_db.execute(
            """
            SELECT
                id,
                user_id,
                conversation_id,
                role,
                content,
                created_at
            FROM messages
            """
        ).fetchall()

        print()
        print(
            f"Found {len(old_conversations)} "
            f"old conversations."
        )

        print(
            f"Found {len(old_messages)} "
            f"old messages."
        )

        # ----------------------------------------------------
        # Copy conversations
        # ----------------------------------------------------

        for row in old_conversations:

            # Make sure user still exists.
            user_exists = connection.execute(
                """
                SELECT id
                FROM users
                WHERE id = ?
                """,
                (row["user_id"],)
            ).fetchone()

            if user_exists is None:

                print(
                    f"Skipping conversation "
                    f"{row['id']} - user missing."
                )

                continue

            connection.execute(
                """
                INSERT OR IGNORE INTO conversations (
                    id,
                    user_id,
                    title,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["user_id"],
                    row["title"],
                    row["created_at"],
                    row["updated_at"],
                )
            )

        # ----------------------------------------------------
        # Copy messages
        # ----------------------------------------------------

        for row in old_messages:

            conversation_exists = connection.execute(
                """
                SELECT id
                FROM conversations
                WHERE id = ?
                """,
                (row["conversation_id"],)
            ).fetchone()

            if conversation_exists is None:

                print(
                    f"Skipping message "
                    f"{row['id']} - conversation missing."
                )

                continue

            connection.execute(
                """
                INSERT OR IGNORE INTO messages (
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
                    row["id"],
                    row["user_id"],
                    row["conversation_id"],
                    row["role"],
                    row["content"],
                    row["created_at"],
                )
            )

        connection.commit()

        old_db.close()

        print()
        print("=" * 70)
        print("MIGRATION COMPLETE")
        print("=" * 70)

        # ====================================================
        # Verification
        # ====================================================

        users = connection.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

        conversations = connection.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        messages = connection.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        print(f"Users:          {users}")
        print(f"Conversations:  {conversations}")
        print(f"Messages:       {messages}")

    finally:

        connection.close()


if __name__ == "__main__":
    migrate()
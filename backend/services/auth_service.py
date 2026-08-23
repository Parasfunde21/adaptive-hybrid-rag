import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# Database
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "app.db"


def get_connection():
    connection = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# Password Hashing
# ============================================================

def hash_password(
    password: str,
    salt: bytes | None = None,
):

    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120000,
    )

    return (
        salt.hex(),
        password_hash.hex(),
    )


def verify_password(
    password: str,
    salt_hex: str,
    password_hash_hex: str,
):

    salt = bytes.fromhex(salt_hex)

    calculated = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120000,
    ).hex()

    return hmac.compare_digest(
        calculated,
        password_hash_hex,
    )


# ============================================================
# Database Initialization
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Users
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            email_verified INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # Sessions
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id)
                REFERENCES users(id)
        )
        """
    )

    # --------------------------------------------------------
    # Email Verification
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS email_verification_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id)
                REFERENCES users(id)
        )
        """
    )

    # --------------------------------------------------------
    # Password Reset
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id)
                REFERENCES users(id)
        )
        """
    )

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# User Creation
# ============================================================

def create_user(
    username: str,
    email: str,
    password: str,
):

    username = username.strip()
    email = email.strip().lower()

    if not username:
        raise ValueError(
            "Username is required."
        )

    if not email:
        raise ValueError(
            "Email is required."
        )

    if len(password) < 6:
        raise ValueError(
            "Password must contain at least 6 characters."
        )

    connection = get_connection()

    try:

        existing = connection.execute(
            """
            SELECT id
            FROM users
            WHERE username = ?
               OR email = ?
            """,
            (
                username,
                email,
            ),
        ).fetchone()

        if existing:
            raise ValueError(
                "Username or email already exists."
            )

        user_id = secrets.token_hex(12)

        salt, password_hash = hash_password(
            password
        )

        connection.execute(
            """
            INSERT INTO users (
                id,
                username,
                email,
                password_hash,
                password_salt,
                email_verified
            )
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (
                user_id,
                username,
                email,
                password_hash,
                salt,
            ),
        )

        connection.commit()

        return {
            "id": user_id,
            "username": username,
            "email": email,
            "email_verified": False,
        }

    finally:

        connection.close()


# ============================================================
# Authentication
# ============================================================

def authenticate_user(
    username_or_email: str,
    password: str,
):

    value = username_or_email.strip()

    connection = get_connection()

    try:

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
               OR email = ?
            """,
            (
                value,
                value.lower(),
            ),
        ).fetchone()

        if user is None:
            return None

        if not verify_password(
            password,
            user["password_salt"],
            user["password_hash"],
        ):
            return None

        # Update login time.
        connection.execute(
            """
            UPDATE users
            SET last_login = ?
            WHERE id = ?
            """,
            (
                datetime.utcnow().isoformat(),
                user["id"],
            ),
        )

        connection.commit()

        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "email_verified": bool(
                user["email_verified"]
            ),
        }

    finally:

        connection.close()


# ============================================================
# Sessions
# ============================================================

def create_session(user_id: str):

    token = secrets.token_urlsafe(48)

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO sessions (
                token,
                user_id
            )
            VALUES (?, ?)
            """,
            (
                token,
                user_id,
            ),
        )

        connection.commit()

    finally:

        connection.close()

    return token


def get_user_from_token(token: str):

    if not token:
        return None

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT
                users.id,
                users.username,
                users.email,
                users.email_verified
            FROM sessions
            JOIN users
                ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "email_verified": bool(
                row["email_verified"]
            ),
        }

    finally:

        connection.close()


def delete_session(token: str):

    connection = get_connection()

    try:

        connection.execute(
            """
            DELETE FROM sessions
            WHERE token = ?
            """,
            (token,),
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# Email Verification
# ============================================================

def create_verification_token(
    user_id: str,
):

    token = secrets.token_urlsafe(48)

    expires_at = (
        datetime.utcnow()
        + timedelta(hours=24)
    ).isoformat()

    connection = get_connection()

    try:

        # Remove previous tokens.
        connection.execute(
            """
            DELETE FROM email_verification_tokens
            WHERE user_id = ?
            """,
            (user_id,),
        )

        connection.execute(
            """
            INSERT INTO email_verification_tokens (
                token,
                user_id,
                expires_at
            )
            VALUES (?, ?, ?)
            """,
            (
                token,
                user_id,
                expires_at,
            ),
        )

        connection.commit()

    finally:

        connection.close()

    return token


def verify_email(
    token: str,
):

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT
                token,
                user_id,
                expires_at
            FROM email_verification_tokens
            WHERE token = ?
            """,
            (token,),
        ).fetchone()

        if row is None:
            return False

        if datetime.fromisoformat(
            row["expires_at"]
        ) < datetime.utcnow():

            connection.execute(
                """
                DELETE FROM email_verification_tokens
                WHERE token = ?
                """,
                (token,),
            )

            connection.commit()

            return False

        connection.execute(
            """
            UPDATE users
            SET email_verified = 1
            WHERE id = ?
            """,
            (row["user_id"],),
        )

        connection.execute(
            """
            DELETE FROM email_verification_tokens
            WHERE token = ?
            """,
            (token,),
        )

        connection.commit()

        return True

    finally:

        connection.close()


# ============================================================
# Password Reset
# ============================================================

def create_password_reset_token(
    email: str,
):

    email = email.strip().lower()

    connection = get_connection()

    try:

        user = connection.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

        if user is None:
            return None

        token = secrets.token_urlsafe(48)

        expires_at = (
            datetime.utcnow()
            + timedelta(minutes=30)
        ).isoformat()

        # Remove old reset tokens.
        connection.execute(
            """
            DELETE FROM password_reset_tokens
            WHERE user_id = ?
            """,
            (user["id"],),
        )

        connection.execute(
            """
            INSERT INTO password_reset_tokens (
                token,
                user_id,
                expires_at,
                used
            )
            VALUES (?, ?, ?, 0)
            """,
            (
                token,
                user["id"],
                expires_at,
            ),
        )

        connection.commit()

        return token

    finally:

        connection.close()


def reset_password(
    token: str,
    new_password: str,
):

    if len(new_password) < 6:
        raise ValueError(
            "Password must contain at least 6 characters."
        )

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT
                token,
                user_id,
                expires_at,
                used
            FROM password_reset_tokens
            WHERE token = ?
            """,
            (token,),
        ).fetchone()

        if row is None:
            return False

        if row["used"]:
            return False

        if datetime.fromisoformat(
            row["expires_at"]
        ) < datetime.utcnow():

            return False

        salt, password_hash = hash_password(
            new_password
        )

        connection.execute(
            """
            UPDATE users
            SET
                password_hash = ?,
                password_salt = ?
            WHERE id = ?
            """,
            (
                password_hash,
                salt,
                row["user_id"],
            ),
        )

        # Mark token as used.
        connection.execute(
            """
            UPDATE password_reset_tokens
            SET used = 1
            WHERE token = ?
            """,
            (token,),
        )

        # Security: invalidate existing sessions.
        connection.execute(
            """
            DELETE FROM sessions
            WHERE user_id = ?
            """,
            (row["user_id"],),
        )

        connection.commit()

        return True

    finally:

        connection.close()
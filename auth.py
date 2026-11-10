import sqlite3
import bcrypt
from pathlib import Path


# --------------------------------------------------
# DATABASE LOCATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"


# --------------------------------------------------
# CREATE DATABASE AND USERS TABLE
# --------------------------------------------------

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# --------------------------------------------------
# REGISTER USER
# --------------------------------------------------

def register_user(username, email, password):

    # Hash the password using bcrypt
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (username, email, password_hash)
        )

        conn.commit()

        return True, "Registration successful."

    except sqlite3.IntegrityError:

        return False, "Username or email already exists."

    finally:
        conn.close()


# --------------------------------------------------
# LOGIN USER
# --------------------------------------------------

def login_user(username, password):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT password_hash
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    result = cursor.fetchone()

    conn.close()

    if result:
        stored_hash = result[0]

        # Check entered password against stored bcrypt hash
        if bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash.encode("utf-8")
        ):
            return True

    return False


# --------------------------------------------------
# CREATE DATABASE WHEN FILE IS LOADED
# --------------------------------------------------

create_database()
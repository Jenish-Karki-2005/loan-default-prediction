import sqlite3
import bcrypt
from pathlib import Path


# Get the project folder
BASE_DIR = Path(__file__).resolve().parent

# Database location
DB_PATH = BASE_DIR / "users.db"


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


# Create the database
create_database()


# Test registration
print(
    register_user(
        "testuser",
        "test@example.com",
        "Test@123"
    )
)
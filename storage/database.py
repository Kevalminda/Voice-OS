import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path

DATABASE_PATH = Path(__file__).parent / "voiceos.db"


def get_connection():
    path = Path(os.getenv("VOICEOS_DATABASE_PATH", str(DATABASE_PATH)))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with closing(get_connection()) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    logging.getLogger(__name__).debug("Database initialized.")

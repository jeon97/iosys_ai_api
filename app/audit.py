from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core import AuditRecord


class SqliteAuditRepository:
    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    input_length INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    attempts INTEGER NOT NULL,
                    error_type TEXT
                )
                """
            )

    def save(self, record: AuditRecord) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO ai_audit_log (
                    model, status, input_sha256, input_length,
                    duration_ms, attempts, error_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.model, record.status, record.input_sha256,
                    record.input_length, record.duration_ms,
                    record.attempts, record.error_type,
                ),
            )


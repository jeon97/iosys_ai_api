import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.audit import SqliteAuditRepository
from app.core import AuditRecord


class SqliteAuditRepositoryTest(unittest.TestCase):
    def test_stores_metadata_without_input_text(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "audit.db"
            repository = SqliteAuditRepository(database)
            repository.save(AuditRecord(
                model="llama3", status="SUCCESS", input_sha256="a" * 64,
                input_length=12, duration_ms=30, attempts=1,
            ))

            with sqlite3.connect(database) as connection:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(ai_audit_log)")]
                row = connection.execute(
                    "SELECT model, status, input_sha256, input_length FROM ai_audit_log"
                ).fetchone()

            self.assertNotIn("input_text", columns)
            self.assertEqual(("llama3", "SUCCESS", "a" * 64, 12), row)


if __name__ == "__main__":
    unittest.main()

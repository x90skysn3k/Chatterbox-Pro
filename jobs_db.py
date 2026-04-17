"""
SQLite durability layer for the in-memory _jobs dict in server.py.

The dict stays the hot-path source of truth — all lock-free reads and
progress writes continue to hit it. This module snapshots relevant
lifecycle events (create, terminal status, delete) to disk so:

 - The dashboard in /studio can list finished jobs across server restarts.
 - Interrupted jobs (server crash / restart mid-generation) are marked as
   'interrupted' on startup rather than vanishing silently.
 - The phishing audit trail has a durable record of every run.

Nothing here blocks the TTS hot path. Writes are synchronous but tiny
(one row per event). If sqlite is unhappy we log and move on rather than
propagating the error into the generation thread.
"""
import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger("extended-server")

DB_PATH = os.environ.get("CHATTERBOX_JOBS_DB", os.path.join("output", "jobs.db"))
_db_lock = threading.Lock()
_initialized = False


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> dict[str, str]:
    """
    Create schema if needed. On startup, mark any 'processing' rows as
    'interrupted' — they can't still be running because we just booted.
    Returns a dict of {job_id: new_status} for any jobs that were adjusted.
    """
    global _initialized
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with _db_lock, _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                request_json TEXT,
                output_path TEXT,
                error TEXT,
                elapsed_seconds REAL,
                engagement_id TEXT,
                kind TEXT DEFAULT 'tts'
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_jobs_created ON jobs(created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_jobs_engagement ON jobs(engagement_id)")

        rows = c.execute(
            "SELECT job_id FROM jobs WHERE status = 'processing'"
        ).fetchall()
        now = time.time()
        fixed = {}
        for (jid,) in rows:
            c.execute(
                "UPDATE jobs SET status='interrupted', updated_at=? WHERE job_id=?",
                (now, jid),
            )
            fixed[jid] = "interrupted"
    _initialized = True
    if fixed:
        logger.info(f"[jobs_db] Marked {len(fixed)} interrupted jobs on startup")
    return fixed


def _safe_write(fn):
    """Swallow sqlite errors — persistence must never break the TTS path."""
    def wrapper(*args, **kwargs):
        if not _initialized:
            return None
        try:
            with _db_lock, _conn() as c:
                return fn(c, *args, **kwargs)
        except Exception as e:
            logger.warning(f"[jobs_db] write failed ({fn.__name__}): {e}")
            return None
    return wrapper


@_safe_write
def record_job_created(c, job_id: str, request_json: str, kind: str = "tts",
                       engagement_id: str | None = None):
    now = time.time()
    c.execute(
        """INSERT OR REPLACE INTO jobs
           (job_id, status, created_at, updated_at, request_json, kind, engagement_id)
           VALUES (?, 'processing', ?, ?, ?, ?, ?)""",
        (job_id, now, now, request_json, kind, engagement_id),
    )


@_safe_write
def record_job_done(c, job_id: str, output_path: str | None, elapsed: float):
    c.execute(
        """UPDATE jobs SET status='done', updated_at=?, output_path=?, elapsed_seconds=?
           WHERE job_id=?""",
        (time.time(), output_path, elapsed, job_id),
    )


@_safe_write
def record_job_failed(c, job_id: str, error: str, elapsed: float | None = None):
    c.execute(
        """UPDATE jobs SET status='failed', updated_at=?, error=?, elapsed_seconds=?
           WHERE job_id=?""",
        (time.time(), error, elapsed, job_id),
    )


@_safe_write
def record_job_cancelled(c, job_id: str):
    c.execute(
        "UPDATE jobs SET status='cancelled', updated_at=? WHERE job_id=?",
        (time.time(), job_id),
    )


@_safe_write
def record_job_deleted(c, job_id: str):
    # Soft-delete so the audit trail survives; /jobs can filter these out.
    c.execute(
        "UPDATE jobs SET status='deleted', updated_at=? WHERE job_id=?",
        (time.time(), job_id),
    )


def list_jobs(limit: int = 100, include_deleted: bool = False,
              engagement_id: str | None = None) -> list[dict[str, Any]]:
    if not _initialized:
        return []
    try:
        with _db_lock, _conn() as c:
            c.row_factory = sqlite3.Row
            query = "SELECT * FROM jobs"
            conds, args = [], []
            if not include_deleted:
                conds.append("status != 'deleted'")
            if engagement_id:
                conds.append("engagement_id = ?")
                args.append(engagement_id)
            if conds:
                query += " WHERE " + " AND ".join(conds)
            query += " ORDER BY created_at DESC LIMIT ?"
            args.append(limit)
            rows = c.execute(query, args).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"[jobs_db] list_jobs failed: {e}")
        return []


def get_job(job_id: str) -> dict[str, Any] | None:
    if not _initialized:
        return None
    try:
        with _db_lock, _conn() as c:
            c.row_factory = sqlite3.Row
            row = c.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.warning(f"[jobs_db] get_job failed: {e}")
        return None

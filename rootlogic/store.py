"""SQLite persistence: sessions, conversation, action log, tasks, sources, token usage, memory.

One file, zero servers. Every table is keyed by ``session_id`` so a session can be
replayed, audited, or deleted as a unit. ``memory_fts`` (FTS5) powers long-term memory:
prior sessions are retrieved by keyword overlap with a new topic.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    status TEXT NOT NULL,              -- running | done | aborted | failed | interrupted
    created_at TEXT NOT NULL,
    finished_at TEXT,
    plan_json TEXT,
    summary TEXT,
    related_topics TEXT,               -- JSON list
    report_path TEXT
);
CREATE TABLE IF NOT EXISTS messages (  -- the human <-> agent conversation
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    role TEXT NOT NULL,                -- user | agent
    kind TEXT NOT NULL,                -- topic | clarifying_question | answer | note | report
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (    -- append-only action log (transparency)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    origin TEXT NOT NULL,
    finding_json TEXT,
    PRIMARY KEY (session_id, task_id)
);
CREATE TABLE IF NOT EXISTS sources (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    published TEXT,
    credibility TEXT,
    kept INTEGER NOT NULL,             -- 1 kept, 0 dropped by filters
    reason TEXT,                       -- drop reason
    data TEXT,
    PRIMARY KEY (session_id, url)
);
CREATE TABLE IF NOT EXISTS llm_calls ( -- token + cost accounting, one row per API request
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    ts TEXT NOT NULL,
    purpose TEXT NOT NULL,             -- clarify | plan | research:t1 | reflect | analyze | report
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    web_searches INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    stop_reason TEXT,
    request_id TEXT
);
CREATE TABLE IF NOT EXISTS preferences ( -- the user's standing preferences (long-term profile)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    text TEXT NOT NULL UNIQUE COLLATE NOCASE,
    session_id TEXT,                   -- session it was learned from; NULL if added by hand
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_rules ( -- user's site rules: block | allow | trust | distrust
    domain TEXT PRIMARY KEY,
    rule TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    session_id UNINDEXED, topic, summary, takeaways
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, id);
CREATE INDEX IF NOT EXISTS idx_calls_session ON llm_calls(session_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Worker threads write concurrently; one connection guarded by a lock is plenty here.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Additive migrations for databases created by older versions."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(sessions)")}
        if "parent_id" not in cols:  # follow-up sessions link to the session they continue
            self._conn.execute("ALTER TABLE sessions ADD COLUMN parent_id TEXT")
        self._conn.commit()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------ sessions
    def create_session(self, topic: str, parent_id: str | None = None) -> str:
        sid = uuid.uuid4().hex[:8]
        self._exec("INSERT INTO sessions (id, topic, status, created_at, parent_id) "
                   "VALUES (?,?,?,?,?)", (sid, topic, "running", now(), parent_id))
        return sid

    def update_session(self, sid: str, **fields: Any) -> None:
        if "related_topics" in fields:
            fields["related_topics"] = json.dumps(fields["related_topics"])
        if fields.get("status") in ("done", "aborted", "failed"):
            fields["finished_at"] = now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE sessions SET {cols} WHERE id = ?", (*fields.values(), sid))

    def session(self, sid: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM sessions WHERE id = ?", (sid,))
        return rows[0] if rows else None

    def sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,))

    def mark_interrupted(self) -> int:
        """Sessions left 'running' by a process that died. Returns how many were marked."""
        cur = self._exec("UPDATE sessions SET status = 'interrupted' WHERE status = 'running'")
        return cur.rowcount

    def delete_session(self, sid: str) -> None:
        self._exec("DELETE FROM memory_fts WHERE session_id = ?", (sid,))
        self._exec("DELETE FROM llm_calls WHERE session_id = ?", (sid,))
        self._exec("DELETE FROM sessions WHERE id = ?", (sid,))

    # ------------------------------------------------------------------ conversation + log
    def add_message(self, sid: str, role: str, kind: str, content: str) -> None:
        self._exec("INSERT INTO messages (session_id, ts, role, kind, content) VALUES (?,?,?,?,?)",
                   (sid, now(), role, kind, content))

    def messages(self, sid: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM messages WHERE session_id = ? ORDER BY id", (sid,))

    def add_event(self, sid: str, type_: str, message: str, data: dict | None = None) -> None:
        self._exec("INSERT INTO events (session_id, ts, type, message, data) VALUES (?,?,?,?,?)",
                   (sid, now(), type_, message, json.dumps(data) if data else None))

    def events(self, sid: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM events WHERE session_id = ? ORDER BY id", (sid,))

    # ------------------------------------------------------------------ tasks + sources
    def tasks(self, sid: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM tasks WHERE session_id = ? ORDER BY rowid", (sid,))

    def upsert_task(self, sid: str, task_id: str, question: str, status: str, origin: str,
                    finding_json: str | None = None) -> None:
        self._exec(
            """INSERT INTO tasks (session_id, task_id, question, status, origin, finding_json)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(session_id, task_id) DO UPDATE SET
                 status = excluded.status,
                 finding_json = COALESCE(excluded.finding_json, tasks.finding_json)""",
            (sid, task_id, question, status, origin, finding_json),
        )

    def update_finding(self, sid: str, task_id: str, finding_json: str) -> None:
        self._exec("UPDATE tasks SET finding_json = ? WHERE session_id = ? AND task_id = ?",
                   (finding_json, sid, task_id))

    def add_source(self, sid: str, task_id: str, url: str, *, title: str = "", published: str = "",
                   credibility: str = "", kept: bool, reason: str = "", data: str = "") -> None:
        self._exec(
            """INSERT OR IGNORE INTO sources
               (session_id, task_id, url, title, published, credibility, kept, reason, data)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (sid, task_id, url, title, published, credibility, int(kept), reason, data),
        )

    def sources(self, sid: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM sources WHERE session_id = ? ORDER BY rowid", (sid,))

    # ------------------------------------------------------------------ usage
    def record_call(self, *, session_id: str | None, purpose: str, model: str, input_tokens: int,
                    output_tokens: int, cache_read_tokens: int, cache_write_tokens: int,
                    web_searches: int, cost_usd: float, stop_reason: str | None,
                    request_id: str | None) -> None:
        self._exec(
            """INSERT INTO llm_calls (session_id, ts, purpose, model, input_tokens, output_tokens,
               cache_read_tokens, cache_write_tokens, web_searches, cost_usd, stop_reason, request_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, now(), purpose, model, input_tokens, output_tokens, cache_read_tokens,
             cache_write_tokens, web_searches, cost_usd, stop_reason, request_id),
        )

    def usage(self, sid: str | None = None) -> dict[str, Any]:
        where, params = ("WHERE session_id = ?", (sid,)) if sid else ("", ())
        rows = self._all(
            f"""SELECT COUNT(*) AS calls,
                   COALESCE(SUM(input_tokens),0) AS input_tokens,
                   COALESCE(SUM(output_tokens),0) AS output_tokens,
                   COALESCE(SUM(cache_read_tokens),0) AS cache_read_tokens,
                   COALESCE(SUM(cache_write_tokens),0) AS cache_write_tokens,
                   COALESCE(SUM(web_searches),0) AS web_searches,
                   COALESCE(SUM(cost_usd),0) AS cost_usd
                FROM llm_calls {where}""", params)
        return rows[0]

    def usage_by_purpose(self, sid: str) -> list[dict[str, Any]]:
        return self._all(
            """SELECT purpose, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens,
                      SUM(output_tokens) AS output_tokens, SUM(web_searches) AS web_searches,
                      SUM(cost_usd) AS cost_usd
               FROM llm_calls WHERE session_id = ? GROUP BY purpose ORDER BY MIN(id)""", (sid,))

    # ------------------------------------------------------------------ user profile
    def preferences(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM preferences ORDER BY category, id")

    def add_preference(self, category: str, text: str, session_id: str | None = None) -> bool:
        """Returns False if an identical preference (case-insensitive) already exists."""
        cur = self._exec("INSERT OR IGNORE INTO preferences (category, text, session_id, "
                         "created_at) VALUES (?,?,?,?)", (category, text.strip(), session_id, now()))
        return cur.rowcount > 0

    def remove_preference(self, pref_id: int) -> bool:
        return self._exec("DELETE FROM preferences WHERE id = ?", (pref_id,)).rowcount > 0

    def clear_preferences(self) -> int:
        return self._exec("DELETE FROM preferences").rowcount

    # ------------------------------------------------------------------ source rules
    def source_rules(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM source_rules ORDER BY rule, domain")

    def set_source_rule(self, domain: str, rule: str) -> None:
        """One rule per domain; setting a new rule replaces the old one."""
        self._exec("INSERT INTO source_rules (domain, rule, created_at) VALUES (?,?,?) "
                   "ON CONFLICT(domain) DO UPDATE SET rule = excluded.rule", (domain, rule, now()))

    def remove_source_rule(self, domain: str) -> bool:
        return self._exec("DELETE FROM source_rules WHERE domain = ?", (domain,)).rowcount > 0

    # ------------------------------------------------------------------ long-term memory
    def remember(self, sid: str, topic: str, summary: str, takeaways: list[str]) -> None:
        self._exec("DELETE FROM memory_fts WHERE session_id = ?", (sid,))
        self._exec("INSERT INTO memory_fts (session_id, topic, summary, takeaways) VALUES (?,?,?,?)",
                   (sid, topic, summary, "\n".join(takeaways)))

    def recall(self, query: str, limit: int = 3, exclude: str | None = None) -> list[dict[str, Any]]:
        """Prior sessions ranked by BM25 relevance to ``query``."""
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        return self._all(
            """SELECT m.session_id, m.topic, m.summary, s.related_topics, s.created_at
               FROM memory_fts m JOIN sessions s ON s.id = m.session_id
               WHERE memory_fts MATCH ? AND m.session_id != COALESCE(?, '')
               ORDER BY bm25(memory_fts) LIMIT ?""",
            (match, exclude, limit),
        )

    def suggestions(self, limit: int = 8) -> list[str]:
        """Related topics from recent sessions that the user hasn't researched yet."""
        done = {s["topic"].lower() for s in self.sessions(100)}
        out: list[str] = []
        for s in self.sessions(20):
            for t in json.loads(s["related_topics"] or "[]"):
                if t.lower() not in done and t not in out:
                    out.append(t)
        return out[:limit]

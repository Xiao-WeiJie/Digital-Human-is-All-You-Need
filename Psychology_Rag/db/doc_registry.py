# -*- coding: utf-8 -*-
"""
文档注册表 — 追踪已入库的数据源文件。

解决的问题:
  当用户新增或更新了某个数据文件（如新版 PsyQA），
  系统需要知道哪些文件已入库、哪些是新增的，实现增量更新。

存储在 SQLite 的 doc_registry 表中（与 SessionStore 共用同一个数据库文件）。
"""

import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import sqlite3

from db.session_store import DEFAULT_DB_PATH


class DocRegistry:
    """
    文档注册表。

    记录每个已入库数据文件的路径、SHA-256 哈希和入库时间，
    支持判断文件是否已入库以及是否发生了变更。
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._init_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_table(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_registry (
                file_path   TEXT PRIMARY KEY,
                file_hash   TEXT NOT NULL,
                collection  TEXT NOT NULL,
                doc_count   INTEGER DEFAULT 0,
                ingested_at TEXT NOT NULL
            )
        """)
        conn.commit()

    @staticmethod
    def _file_hash(filepath: str) -> str:
        """计算文件的 SHA-256 哈希。"""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_ingested(self, filepath: str) -> bool:
        """检查文件是否已入库（且内容未变更）。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT file_hash FROM doc_registry WHERE file_path = ?",
            (str(filepath),),
        ).fetchone()

        if not row:
            return False  # 从未入库

        # 检查内容是否发生变更
        current_hash = self._file_hash(filepath)
        return row["file_hash"] == current_hash

    def register(
        self,
        filepath: str,
        collection: str,
        doc_count: int,
    ) -> None:
        """
        注册一个已入库的文件。

        Args:
            filepath: 数据文件路径
            collection: 入库到哪个 Collection（factual_kb / guardrail_kb）
            doc_count: 该文件产生的文档块数量
        """
        conn = self._get_conn()
        now = datetime.now().isoformat()
        file_hash = self._file_hash(filepath)

        conn.execute(
            "INSERT OR REPLACE INTO doc_registry "
            "(file_path, file_hash, collection, doc_count, ingested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(filepath), file_hash, collection, doc_count, now),
        )
        conn.commit()

    def list_registered(self) -> list:
        """列出所有已注册的文件。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT file_path, collection, doc_count, ingested_at "
            "FROM doc_registry ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

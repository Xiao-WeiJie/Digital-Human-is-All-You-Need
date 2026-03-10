# -*- coding: utf-8 -*-
"""
会话持久化存储 — 基于 SQLite。

解决的核心问题:
  旧方案: 对话历史存在 Python dict 中 → 进程关闭即丢失
  新方案: 对话历史写入 SQLite → 重启后可恢复对话上下文

存储内容:
  ┌───────────────────────────────────────────┐
  │  sessions 表        会话元信息             │
  │  ├─ session_id      唯一标识               │
  │  ├─ user_name       用户名                 │
  │  ├─ summary         当前对话摘要           │
  │  ├─ created_at      创建时间               │
  │  └─ updated_at      最后活跃时间           │
  │                                           │
  │  messages 表        消息记录               │
  │  ├─ id              自增主键               │
  │  ├─ session_id      所属会话 FK            │
  │  ├─ role            "human" / "ai"        │
  │  ├─ content         消息正文               │
  │  └─ timestamp       消息时间               │
  │                                           │
  │  empathy_examples 表  共情策略缓存         │
  │  ├─ id              自增主键               │
  │  ├─ user_input      来访者话语             │
  │  └─ response        咨询师共情回复         │
  └───────────────────────────────────────────┘

依赖: 仅 Python 标准库 sqlite3，无需额外安装。
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ── 默认数据库路径 ──
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "psymind.db"


class SessionStore:
    """
    基于 SQLite 的会话持久化存储。

    管理对话历史、摘要、用户名、共情示例等结构化数据。
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: SQLite 数据库文件路径，默认为 db/psymind.db
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（懒初始化）。"""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,  # 允许多线程访问
            )
            self._conn.row_factory = sqlite3.Row  # 支持按列名访问
            self._conn.execute("PRAGMA journal_mode=WAL")  # 写入性能优化
        return self._conn

    def _init_db(self):
        """初始化数据库表结构。"""
        conn = self._get_conn()

        conn.executescript("""
            -- 会话元信息表
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                user_name   TEXT DEFAULT '朋友',
                summary     TEXT DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            -- 消息记录表
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('human', 'ai')),
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- 消息表索引：按会话+时间快速查询
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp);

            -- 共情策略缓存表
            CREATE TABLE IF NOT EXISTS empathy_examples (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input      TEXT NOT NULL,
                response        TEXT NOT NULL,
                source          TEXT DEFAULT 'CPsyCounD'
            );
        """)
        conn.commit()

        print(f"  [数据库] SQLite: {self.db_path}")

    # ════════════════════════════════════════════
    # 会话管理
    # ════════════════════════════════════════════

    def create_session(self, session_id: str) -> None:
        """创建新会话。"""
        now = datetime.now().isoformat()
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) "
            "VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        conn.commit()

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def list_sessions(self, limit: int = 20) -> List[dict]:
        """列出最近的会话。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT session_id, user_name, summary, created_at, updated_at "
            "FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ════════════════════════════════════════════
    # 消息历史
    # ════════════════════════════════════════════

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        添加一条消息记录。

        Args:
            session_id: 会话 ID
            role: "human" 或 "ai"
            content: 消息正文
        """
        now = datetime.now().isoformat()
        conn = self._get_conn()

        # 确保会话存在
        self.create_session(session_id)

        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        # 更新会话活跃时间
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        conn.commit()

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[BaseMessage]:
        """
        获取某个会话的消息历史，返回 LangChain Message 对象列表。

        Args:
            session_id: 会话 ID
            limit: 最多返回几条消息（从最近的开始），None 为全部

        Returns:
            LangChain BaseMessage 列表（HumanMessage / AIMessage）
        """
        conn = self._get_conn()

        if limit:
            # 取最近 N 条，但要按时间正序返回
            rows = conn.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, timestamp FROM messages "
                "  WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?"
                ") ORDER BY timestamp ASC",
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()

        messages = []
        for row in rows:
            if row["role"] == "human":
                messages.append(HumanMessage(content=row["content"]))
            else:
                messages.append(AIMessage(content=row["content"]))
        return messages

    def get_message_count(self, session_id: str) -> int:
        """获取某个会话的消息总数。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def delete_old_messages(self, session_id: str, keep_recent: int) -> int:
        """
        删除旧消息，只保留最近 N 条（用于摘要后的裁剪）。

        Args:
            session_id: 会话 ID
            keep_recent: 保留最近几条消息

        Returns:
            删除的消息数量
        """
        conn = self._get_conn()

        # 找出要保留的最小 ID
        row = conn.execute(
            "SELECT id FROM messages WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT 1 OFFSET ?",
            (session_id, keep_recent - 1),
        ).fetchone()

        if not row:
            return 0  # 消息不够多，无需删除

        cutoff_id = row["id"]
        cursor = conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND id < ?",
            (session_id, cutoff_id),
        )
        conn.commit()
        return cursor.rowcount

    # ════════════════════════════════════════════
    # 摘要 & 用户名
    # ════════════════════════════════════════════

    def get_summary(self, session_id: str) -> str:
        """获取某个会话的对话摘要。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT summary FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["summary"] if row and row["summary"] else ""

    def set_summary(self, session_id: str, summary: str) -> None:
        """更新某个会话的对话摘要。"""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET summary = ? WHERE session_id = ?",
            (summary, session_id),
        )
        conn.commit()

    def get_user_name(self, session_id: str) -> str:
        """获取某个会话的用户名。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT user_name FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["user_name"] if row and row["user_name"] else "朋友"

    def set_user_name(self, session_id: str, name: str) -> None:
        """设置某个会话的用户名。"""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET user_name = ? WHERE session_id = ?",
            (name, session_id),
        )
        conn.commit()

    # ════════════════════════════════════════════
    # 共情策略缓存
    # ════════════════════════════════════════════

    def save_empathy_examples(self, examples: List[Dict[str, str]]) -> int:
        """
        批量保存共情策略示例（清空旧数据后重新写入）。

        Args:
            examples: [{"user_input": "...", "empathetic_response": "..."}]

        Returns:
            写入的条数
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM empathy_examples")
        conn.executemany(
            "INSERT INTO empathy_examples (user_input, response) VALUES (?, ?)",
            [(ex["user_input"], ex["empathetic_response"]) for ex in examples],
        )
        conn.commit()
        return len(examples)

    def get_empathy_examples(self, limit: int = 200) -> List[Dict[str, str]]:
        """获取共情策略示例。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT user_input, response FROM empathy_examples LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"user_input": r["user_input"], "empathetic_response": r["response"]}
            for r in rows
        ]

    def get_empathy_count(self) -> int:
        """获取共情示例总数。"""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM empathy_examples").fetchone()
        return row["cnt"] if row else 0

    # ════════════════════════════════════════════
    # 统计 & 清理
    # ════════════════════════════════════════════

    def get_stats(self) -> dict:
        """返回数据库统计信息。"""
        conn = self._get_conn()
        sessions = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()
        messages = conn.execute("SELECT COUNT(*) as cnt FROM messages").fetchone()
        empathy = conn.execute("SELECT COUNT(*) as cnt FROM empathy_examples").fetchone()
        return {
            "db_path": str(self.db_path),
            "sessions": sessions["cnt"],
            "messages": messages["cnt"],
            "empathy_examples": empathy["cnt"],
        }

    def close(self):
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

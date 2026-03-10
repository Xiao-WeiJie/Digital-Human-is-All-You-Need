# -*- coding: utf-8 -*-
"""
对话记忆管理器 — SQLite 持久化版。

旧方案: 对话历史存在 Python dict → 进程关闭即丢失
新方案: 对话历史写入 SQLite → 重启后可恢复、可追溯

功能:
  - 多 Session 管理（持久化到 SQLite）
  - 自动对话摘要（超过阈值时触发 LLM 压缩）
  - 10+ 轮长程对话支持
  - 用户名自动识别与记忆
"""

import re
from typing import List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from db.session_store import SessionStore
from config.settings import MAX_RECENT_TURNS, SUMMARY_TRIGGER_TURNS
from prompts.templates import SUMMARY_SYSTEM_PROMPT


class ConversationMemoryManager:
    """
    对话记忆管理器 — 基于 SQLite 持久化。

    所有读写操作通过 SessionStore 落库，
    即使进程重启也能恢复完整的对话上下文。
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        session_store: SessionStore,
        max_recent_turns: int = MAX_RECENT_TURNS,
        summary_trigger: int = SUMMARY_TRIGGER_TURNS,
    ):
        """
        Args:
            llm: 用于生成摘要的语言模型（建议低温度）
            session_store: SQLite 会话存储实例
            max_recent_turns: 保留最近几轮完整对话
            summary_trigger: 超过几轮时触发摘要压缩
        """
        self.llm = llm
        self.store = session_store
        self.max_recent_turns = max_recent_turns
        self.summary_trigger = summary_trigger

    # ── 会话 & 消息 ──

    def ensure_session(self, session_id: str):
        """确保会话存在（幂等操作）。"""
        self.store.create_session(session_id)

    def add_user_message(self, session_id: str, content: str):
        """记录一条用户消息。"""
        self.store.add_message(session_id, "human", content)

    def add_ai_message(self, session_id: str, content: str):
        """记录一条 AI 回复。"""
        self.store.add_message(session_id, "ai", content)

    def get_history_messages(self, session_id: str) -> List[BaseMessage]:
        """
        获取用于注入 Prompt 的历史消息列表。

        返回结构: [SystemMessage(摘要)] + [最近 N 轮的 Human/AI 消息]
        """
        messages = []

        # 摘要前缀
        summary = self.store.get_summary(session_id)
        if summary:
            messages.append(SystemMessage(content=f"[对话历史摘要] {summary}"))

        # 最近 N 轮完整消息
        recent_limit = self.max_recent_turns * 2  # 每轮 2 条
        recent = self.store.get_messages(session_id, limit=recent_limit)
        messages.extend(recent)

        return messages

    # ── 摘要 & 用户名 ──

    def get_summary(self, session_id: str) -> str:
        return self.store.get_summary(session_id)

    def get_user_name(self, session_id: str) -> str:
        return self.store.get_user_name(session_id)

    def try_extract_name(self, session_id: str, user_input: str):
        """尝试从用户输入中提取名字。"""
        # 只匹配中文字符和英文字母，避免将标点或助词错误纳入名字
        patterns = [
            r"我叫([\u4e00-\u9fa5a-zA-Z]{1,4})",
            r"我是([\u4e00-\u9fa5a-zA-Z]{1,4})",
            r"叫我([\u4e00-\u9fa5a-zA-Z]{1,4})",
            r"我的名字是([\u4e00-\u9fa5a-zA-Z]{1,4})",
        ]
        noise_words = {"一个", "什么", "怎么", "这个", "那个", "这样", "那样"}
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                name = match.group(1)
                # 检查名字本身或其子串是否为噪声词
                if len(name) <= 4 and not any(nw in name for nw in noise_words):
                    self.store.set_user_name(session_id, name)
                    return

    async def maybe_summarize(self, session_id: str):
        """
        检查是否需要对话摘要。

        当消息总数超过阈值时，用 LLM 将旧消息压缩为摘要，
        然后从 SQLite 中删除旧消息，只保留最近 N 轮。
        """
        total = self.store.get_message_count(session_id)
        num_turns = total // 2

        if num_turns <= self.summary_trigger:
            return

        keep_count = self.max_recent_turns * 2

        # 获取需要摘要的旧消息（全部消息减去最近保留的）
        all_messages = self.store.get_messages(session_id)
        msgs_to_summarize = all_messages[:-keep_count] if keep_count < len(all_messages) else []

        if not msgs_to_summarize:
            return

        # 拼接对话文本
        old_summary = self.store.get_summary(session_id)
        conversation_text = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '小暖'}: {m.content}"
            for m in msgs_to_summarize
        )

        # 调用 LLM 生成摘要
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("human",
             "已有摘要：{old_summary}\n\n"
             "需要整合的新对话：\n{conversation}\n\n"
             "请生成更新后的摘要："),
        ])
        summary_chain = summary_prompt | self.llm | StrOutputParser()
        new_summary = await summary_chain.ainvoke({
            "old_summary": old_summary or "（这是对话的开始）",
            "conversation": conversation_text,
        })

        # 持久化摘要 & 裁剪旧消息
        self.store.set_summary(session_id, new_summary)
        deleted = self.store.delete_old_messages(session_id, keep_count)

        print(f"  [记忆管理] Session {session_id[:8]}... "
              f"摘要压缩完成，删除 {deleted} 条旧消息")

    # ── 调试 ──

    def get_debug_info(self, session_id: str) -> dict:
        """返回调试信息。"""
        return {
            "message_count": self.store.get_message_count(session_id),
            "summary": self.store.get_summary(session_id) or "（无）",
            "user_name": self.store.get_user_name(session_id),
        }
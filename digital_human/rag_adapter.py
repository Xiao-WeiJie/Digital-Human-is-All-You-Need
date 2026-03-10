# -*- coding: utf-8 -*-
"""
Digital Human Phase 1 - RAG Adapter

封装 Psychology_Rag 对话系统。
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Optional

# 添加 RAG 目录到路径
RAG_DIR = Path(__file__).parent.parent / "Psychology_Rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

# 设置环境变量
from .config import RAGConfig


class RAGAdapter:
    """RAG 对话系统适配器"""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.system = None
        self._initialized = False
        self._session_id = None

    def initialize(self):
        """初始化 RAG 系统（延迟加载）"""
        if self._initialized:
            return

        # 设置 API Key
        if self.config.dashscope_api_key:
            os.environ["DASHSCOPE_API_KEY"] = self.config.dashscope_api_key

        print("[RAG] 正在初始化 PsyMind 系统...")
        try:
            from system import PsyMindSystem
            self.system = PsyMindSystem()
            print("[RAG] PsyMind 系统初始化完成")
        except Exception as e:
            print(f"[RAG] 初始化失败: {e}")
            raise

        self._initialized = True

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """获取或创建会话 ID"""
        if session_id:
            self._session_id = session_id
        elif not self._session_id:
            self._session_id = str(uuid.uuid4())
        return self._session_id

    async def chat(
        self,
        user_input: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        对话接口

        Args:
            user_input: 用户输入文本
            session_id: 会话 ID（可选，用于保持对话上下文）

        Returns:
            dict: {
                "response": str,  # AI 回复
                "session_id": str,  # 会话 ID
                "intent": str,  # 意图分类（可选）
            }
        """
        if not self._initialized:
            self.initialize()

        sid = self.get_or_create_session(session_id)

        try:
            # 调用 RAG 系统处理消息
            response = await self.system.process_message(user_input, sid)

            return {
                "response": response,
                "session_id": sid,
            }

        except Exception as e:
            print(f"[RAG] 处理消息失败: {e}")
            return {
                "response": f"抱歉，我遇到了一些问题。请稍后再试。",
                "session_id": sid,
                "error": str(e),
            }

    def chat_sync(self, user_input: str, session_id: Optional[str] = None) -> str:
        """
        同步对话接口（简单版本）

        Args:
            user_input: 用户输入文本
            session_id: 会话 ID

        Returns:
            str: AI 回复
        """
        import asyncio

        result = asyncio.run(self.chat(user_input, session_id))
        return result["response"]

    def reset_session(self) -> str:
        """重置会话，创建新会话"""
        self._session_id = str(uuid.uuid4())
        return self._session_id

    def __call__(self, user_input: str) -> str:
        """使适配器可调用"""
        return self.chat_sync(user_input)

# -*- coding: utf-8 -*-
"""
PsyMind 系统主类 — 全局编排器（数据库持久化版）。

组件依赖关系:
  ┌──────────────────────────────────────────────┐
  │              PsyMindSystem                    │
  │                                              │
  │  ┌─────────┐   ┌──────────────────────────┐  │
  │  │ LLM     │   │ DB Layer                 │  │
  │  │ (Qwen)  │   │ ┌──────────────────────┐ │  │
  │  └────┬────┘   │ │ Chroma (向量库)       │ │  │
  │       │        │ │ • factual_kb          │ │  │
  │       │        │ │ • guardrail_kb        │ │  │
  │       │        │ └──────────────────────┘ │  │
  │       │        │ ┌──────────────────────┐ │  │
  │       │        │ │ SQLite (结构化)       │ │  │
  │       │        │ │ • sessions + messages │ │  │
  │       │        │ │ • empathy_examples    │ │  │
  │       │        │ │ • doc_registry        │ │  │
  │       │        │ └──────────────────────┘ │  │
  │       │        └──────────────────────────┘  │
  │       ▼                    │                 │
  │  ┌─────────┐  ┌───────────▼──────────┐      │
  │  │ Router  │  │ MultiWayRetriever    │      │
  │  └────┬────┘  └───────────┬──────────┘      │
  │       │                   │                 │
  │       ▼                   ▼                 │
  │  ┌──────────────────────────────────────┐   │
  │  │ LCEL Chains + ConversationMemory     │   │
  │  └──────────────────────────────────────┘   │
  └──────────────────────────────────────────────┘
"""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_COLD_TEMPERATURE,
    LLM_MAX_TOKENS,
    CRISIS_HOTLINE_INFO,
    DISCLAIMER,
)
from db.vector_store import VectorStoreManager
from db.session_store import SessionStore
from db.doc_registry import DocRegistry
from retrieval.multi_way_retriever import MultiWayRetriever
from routing.intent_classifier import IntentClassifier
from memory.conversation_memory import ConversationMemoryManager
from chains.dialogue_chains import ChainFactory


def _init_llm(temperature: float = LLM_TEMPERATURE) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=DASHSCOPE_BASE_URL,
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
    )


def _init_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=DASHSCOPE_BASE_URL,
        check_embedding_ctx_length=False
    )


class PsyMindSystem:
    """PsyMind 对话系统全局编排器 — 数据库持久化版。"""

    def __init__(self):
        print("=" * 60)
        print("  🧠 PsyMind — 心理健康科普对话系统 初始化中...")
        print("=" * 60)

        # ── 模型层 ──
        self.llm = _init_llm(temperature=LLM_TEMPERATURE)
        self.llm_cold = _init_llm(temperature=LLM_COLD_TEMPERATURE)
        self.embeddings = _init_embeddings()

        # ── 数据库层 ──
        self.session_store = SessionStore()             # SQLite: 会话 + 共情
        self.vector_manager = VectorStoreManager(self.embeddings)  # Chroma: 向量
        self.doc_registry = DocRegistry()               # SQLite: 文档注册

        # ── 知识检索层（接入数据库） ──
        self.retriever = MultiWayRetriever(
            embeddings=self.embeddings,
            vector_manager=self.vector_manager,
            session_store=self.session_store,
        )
        self.retriever.initialize()

        # ── 路由层 ──
        self.classifier = IntentClassifier(self.llm_cold)

        # ── 记忆层（接入 SQLite） ──
        self.memory = ConversationMemoryManager(
            llm=self.llm_cold,
            session_store=self.session_store,
        )

        # ── 对话链层 ──
        self.chains = ChainFactory.build_all(self.llm)

        # ── 打印存储统计 ──
        self._print_stats()

        print("=" * 60)
        print("  ✅ 系统初始化完成！")
        print("=" * 60)

    def _print_stats(self):
        """打印各存储层的统计信息。"""
        stats = self.retriever.get_stats()
        print(f"\n  📊 存储统计:")
        print(f"     Chroma 科普核心库: {stats.get('factual_kb_count', 0)} 个文档块")
        print(f"     Chroma 安全护栏库: {stats.get('guardrail_kb_count', 0)} 个文档块")
        print(f"     SQLite 共情示例:   {stats.get('empathy_examples', 0)} 条")
        print(f"     SQLite 历史会话:   {stats.get('sessions', 0)} 个")
        print(f"     SQLite 历史消息:   {stats.get('messages', 0)} 条")

    # ────────────────────────────────────────────
    # 核心处理流程
    # ────────────────────────────────────────────

    async def process_message(self, user_input: str, session_id: str) -> str:
        """处理一条用户消息的完整流程。"""

        # 确保会话存在
        self.memory.ensure_session(session_id)

        # ── Step 1: 意图分类 ──
        summary = self.memory.get_summary(session_id)
        intent = await self.classifier.classify(user_input, summary)
        print(f"  [路由] 意图分类: {intent}")

        # ── Step 2: 构建上下文历史（从 SQLite 读取） ──
        history_messages = self.memory.get_history_messages(session_id)
        user_name = self.memory.get_user_name(session_id)

        # ── Step 3: 路由 → 检索 + 生成 ──
        response = await self._route_and_generate(
            intent, user_input, history_messages, user_name
        )

        # ── Step 4: 持久化消息（写入 SQLite） ──
        self.memory.add_user_message(session_id, user_input)
        self.memory.add_ai_message(session_id, response)

        # ── Step 5: 自动摘要 ──
        await self.memory.maybe_summarize(session_id)

        # ── Step 6: 用户名识别 ──
        self.memory.try_extract_name(session_id, user_input)

        return response

    async def _route_and_generate(
        self,
        intent: str,
        user_input: str,
        history_messages: list,
        user_name: str,
    ) -> str:
        """根据意图路由到对应的链路，执行检索与生成。"""
        base_params = {
            "input": user_input,
            "history": history_messages,
            "user_name": user_name,
        }

        if intent == "emotional":
            response = await self.chains.emotional.ainvoke({
                **base_params,
                "intent_mode": "情绪关怀",
                "empathy_examples": self.retriever.format_empathy_examples(),
                "factual_context": self.retriever.retrieve_factual(user_input, k=2),
            })

        elif intent == "knowledge":
            response = await self.chains.knowledge.ainvoke({
                **base_params,
                "intent_mode": "知识科普",
                "factual_context": self.retriever.retrieve_factual(user_input, k=3),
            })

        elif intent == "guardrail":
            response = await self.chains.guardrail.ainvoke({
                **base_params,
                "intent_mode": "安全护栏",
                "guardrail_context": self.retriever.retrieve_guardrail(user_input),
                "crisis_info": CRISIS_HOTLINE_INFO,
            })
            response += DISCLAIMER

        else:
            response = await self.chains.chitchat.ainvoke({
                **base_params,
                "intent_mode": "日常交流",
            })

        return response
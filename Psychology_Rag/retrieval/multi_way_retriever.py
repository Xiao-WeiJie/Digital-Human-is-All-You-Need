# -*- coding: utf-8 -*-
"""
多路检索器 (Multi-way Retriever) — 数据库持久化版。

架构对比:
  旧方案 (内存)                  新方案 (持久化)
  启动 → 加载文档 → 向量化       启动 → 连接 Chroma → 直接可用
       → 建 FAISS 内存索引             → 毫秒级启动
  关闭 → 索引丢失               关闭 → 数据仍在磁盘
  新增数据 → 全量重建           新增数据 → 增量入库

组件:
  Chroma factual_kb   ← 科普核心库（向量检索）
  Chroma guardrail_kb ← 安全护栏库（向量检索）
  SQLite empathy_examples ← 共情策略库（Few-shot 注入）
"""

from typing import List, Dict
from langchain_openai import OpenAIEmbeddings

from db.vector_store import VectorStoreManager
from db.session_store import SessionStore
from data.guardrail_docs import build_guardrail_docs
from data.empathy_examples import build_empathy_examples
from data.factual_docs import build_factual_knowledge_docs
from config.settings import (
    FACTUAL_RETRIEVAL_K,
    GUARDRAIL_RETRIEVAL_K,
    MAX_EMPATHY_EXAMPLES,
)


class MultiWayRetriever:
    """
    多路检索器 — 数据库持久化版。

    协调 Chroma (向量) + SQLite (结构化) 两个持久化层。
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        vector_manager: VectorStoreManager,
        session_store: SessionStore,
    ):
        self.embeddings = embeddings
        self.vector_mgr = vector_manager
        self.session_store = session_store

    def initialize(self):
        """
        初始化知识库。

        流程:
        1. 加载 Chroma 持久化索引（已有数据则毫秒级完成）
        2. 检查各库是否为空
        3. 为空 → 首次数据入库；已有 → 跳过
        """
        print("\n🔨 正在初始化知识库...")

        factual_count, guardrail_count = self.vector_mgr.load_stores()

        needs_factual = (factual_count == 0)
        needs_guardrail = (guardrail_count == 0)
        needs_empathy = (self.session_store.get_empathy_count() == 0)

        if not (needs_factual or needs_guardrail or needs_empathy):
            empathy_count = self.session_store.get_empathy_count()
            print(f"  [共情库] empathy_examples: {empathy_count} 条 (SQLite)")
            print("✅ 全部知识库已从磁盘加载，无需重建\n")
            return

        if needs_factual:
            print("  [首次入库] 科普核心库...")
            self.vector_mgr.ingest_factual(build_factual_knowledge_docs())

        if needs_guardrail:
            print("  [首次入库] 安全护栏库...")
            self.vector_mgr.ingest_guardrail(build_guardrail_docs())

        if needs_empathy:
            print("  [首次入库] 共情策略库...")
            examples = build_empathy_examples()
            saved = self.session_store.save_empathy_examples(examples)
            print(f"  [入库] empathy_examples: {saved} 条写入 SQLite")

        print("✅ 知识库初始化完成\n")

    def rebuild(self):
        """强制重建全部知识库（清空后重新入库）。用于数据集更新后全量刷新。"""
        print("\n🔄 正在重建全部知识库...")
        self.vector_mgr.clear_collection("factual_kb")
        self.vector_mgr.clear_collection("guardrail_kb")
        self.vector_mgr.ingest_factual(build_factual_knowledge_docs())
        self.vector_mgr.ingest_guardrail(build_guardrail_docs())
        self.session_store.save_empathy_examples(build_empathy_examples())
        print("✅ 全部知识库重建完成\n")

    # ── 检索接口 ──

    def retrieve_factual(self, query: str, k: int = FACTUAL_RETRIEVAL_K) -> str:
        return self.vector_mgr.search_factual(query, k=k)

    def retrieve_guardrail(self, query: str, k: int = GUARDRAIL_RETRIEVAL_K) -> str:
        return self.vector_mgr.search_guardrail(query, k=k)

    def get_empathy_examples(self) -> List[Dict[str, str]]:
        return self.session_store.get_empathy_examples()

    def format_empathy_examples(self, max_examples: int = MAX_EMPATHY_EXAMPLES) -> str:
        examples = self.get_empathy_examples()
        formatted = []
        for i, ex in enumerate(examples[:max_examples], 1):
            formatted.append(
                f"--- 共情示例 {i} ---\n"
                f"来访者: {ex['user_input']}\n"
                f"回应: {ex['empathetic_response']}"
            )
        return "\n\n".join(formatted)

    def get_stats(self) -> dict:
        return {**self.vector_mgr.get_stats(), **self.session_store.get_stats()}
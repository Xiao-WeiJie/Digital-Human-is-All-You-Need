# -*- coding: utf-8 -*-
"""
向量数据库管理器 — 基于 Chroma 的持久化向量存储。

解决的核心问题:
  旧方案: 每次启动 → 从文档重建 FAISS 内存索引 → 关闭即丢失
  新方案: 首次运行 → 入库 Chroma 持久化到磁盘 → 后续启动直接加载

架构:
┌─────────────────────────────────────────────────────┐
│                  Chroma 持久化存储                    │
│                 (db/chroma_data/)                    │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ Collection:   │  │ Collection:   │                │
│  │ factual_kb   │  │ guardrail_kb │                 │
│  │ (科普核心库)  │  │ (安全护栏库)  │                 │
│  └──────────────┘  └──────────────┘                 │
│                                                     │
│  优势:                                               │
│  • 持久化: 数据写入磁盘，重启不丢失                    │
│  • 增量更新: 只入库新文档，跳过已有文档                 │
│  • Metadata 过滤: 检索时可按 type/source 过滤         │
│  • 集合隔离: 不同级别知识库独立管理                     │
└─────────────────────────────────────────────────────┘

依赖: pip install chromadb langchain-chroma
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional
import time
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config.settings import (
    FACTUAL_RETRIEVAL_K,
    GUARDRAIL_RETRIEVAL_K,
)


# ── 默认的 Chroma 持久化目录 ──
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_data"


class VectorStoreManager:
    """
    向量数据库管理器。

    管理两个独立的 Chroma Collection：
      - factual_kb:   科普核心库（优先级一）
      - guardrail_kb: 安全护栏库（优先级三）

    优先级二的共情策略库不走向量检索，由 SessionStore 或内存管理。
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        persist_dir: Optional[str] = None,
    ):
        """
        Args:
            embeddings: 向量化模型实例
            persist_dir: Chroma 数据持久化目录，默认为 db/chroma_data/
        """
        self.embeddings = embeddings
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._factual_store: Optional[Chroma] = None
        self._guardrail_store: Optional[Chroma] = None

    # ────────────────────────────────────────────
    # 初始化 / 加载
    # ────────────────────────────────────────────

    def _get_or_create_collection(self, collection_name: str) -> Chroma:
        """获取或创建一个 Chroma Collection（自动持久化）。"""
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def load_stores(self):
        """
        加载已有的持久化向量库。

        如果磁盘上已有数据，直接加载（毫秒级）。
        如果是首次运行，创建空 Collection，等待 ingest 填充。
        """
        print(f"  [向量库] 持久化目录: {self.persist_dir}")

        self._factual_store = self._get_or_create_collection("factual_kb")
        self._guardrail_store = self._get_or_create_collection("guardrail_kb")

        # 报告当前库中的文档数量
        factual_count = self._factual_store._collection.count()
        guardrail_count = self._guardrail_store._collection.count()

        print(f"  [向量库] factual_kb:   {factual_count} 个文档块")
        print(f"  [向量库] guardrail_kb: {guardrail_count} 个文档块")

        return factual_count, guardrail_count

    # ────────────────────────────────────────────
    # 数据入库（增量）
    # ────────────────────────────────────────────

    @staticmethod
    def _doc_hash(doc: Document) -> str:
        """计算文档的内容指纹，用于去重。"""
        content = doc.page_content + str(sorted(doc.metadata.items()))
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def ingest_factual(self, docs: List[Document], batch_size: int = 10) -> int:
        """
        将科普核心库文档入库到 Chroma (支持分批与自动重试避免500错误)。
        """
        if not docs:
            return 0

        store = self._factual_store
        existing = store._collection.get()
        existing_ids = set(existing["ids"]) if existing["ids"] else set()

        new_docs = []
        new_ids = []
        for doc in docs:
            # --- 新增：数据清洗与校验 ---
            # 如果内容不是字符串，或者去除空格后是空字符串，则直接跳过这条数据
            if not isinstance(doc.page_content, str) or not doc.page_content.strip():
                continue
            # ---------------------------
            doc_id = f"factual_{self._doc_hash(doc)}"
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)

        if new_docs:
            print(f"  [入库] factual_kb: 准备新增 {len(new_docs)} 个文档块 (跳过 {len(docs) - len(new_docs)} 个已有)")

            # --- 新增：分批入库逻辑 ---
            for i in range(0, len(new_docs), batch_size):
                batch_docs = new_docs[i: i + batch_size]
                batch_ids = new_ids[i: i + batch_size]

                print(f"    -> 正在处理批次: {i + 1} ~ {min(i + batch_size, len(new_docs))} / {len(new_docs)}")

                # 引入简单的重试机制
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        store.add_documents(documents=batch_docs, ids=batch_ids)
                        time.sleep(0.5)  # 成功后休眠0.5秒，避免触发速率限制
                        break  # 成功入库，跳出重试循环
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"    ⚠️ 第 {attempt + 1} 次请求失败 ({e})，休息 5 秒后重试...")
                            time.sleep(5)
                        else:
                            print(f"    ❌ 批次处理彻底失败: {e}")
                            raise e  # 重试3次仍失败，抛出异常
            # --------------------------
            print(f"  [入库] factual_kb: {len(new_docs)} 个文档块分批入库完成！")
        else:
            print(f"  [入库] factual_kb: 全部 {len(docs)} 个文档块已存在，跳过")

        return len(new_docs)

    def ingest_guardrail(self, docs: List[Document], batch_size: int = 10) -> int:
        """
        将安全护栏文档入库到 Chroma (支持分批与自动重试)。
        """
        if not docs:
            return 0

        store = self._guardrail_store
        existing = store._collection.get()
        existing_ids = set(existing["ids"]) if existing["ids"] else set()

        new_docs = []
        new_ids = []
        for doc in docs:
            # --- 新增：数据清洗与校验 ---
            if not isinstance(doc.page_content, str) or not doc.page_content.strip():
                continue
            # ---------------------------
            doc_id = f"guardrail_{self._doc_hash(doc)}"
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)

        if new_docs:
            print(f"  [入库] guardrail_kb: 准备新增 {len(new_docs)} 个文档块 (跳过 {len(docs) - len(new_docs)} 个已有)")

            for i in range(0, len(new_docs), batch_size):
                batch_docs = new_docs[i: i + batch_size]
                batch_ids = new_ids[i: i + batch_size]
                print(f"    -> 正在处理批次: {i + 1} ~ {min(i + batch_size, len(new_docs))} / {len(new_docs)}")

                for attempt in range(3):
                    try:
                        store.add_documents(documents=batch_docs, ids=batch_ids)
                        time.sleep(0.5)
                        break
                    except Exception as e:
                        if attempt < 2:
                            print(f"    ⚠️ 第 {attempt + 1} 次请求失败 ({e})，休息 5 秒后重试...")
                            time.sleep(5)
                        else:
                            raise e

            print(f"  [入库] guardrail_kb: {len(new_docs)} 个文档块分批入库完成！")
        else:
            print(f"  [入库] guardrail_kb: 全部 {len(docs)} 个文档块已存在，跳过")

        return len(new_docs)

    # ────────────────────────────────────────────
    # 检索
    # ────────────────────────────────────────────

    def search_factual(
        self,
        query: str,
        k: int = FACTUAL_RETRIEVAL_K,
        filter_metadata: Optional[dict] = None,
    ) -> str:
        """
        从科普核心库检索语义相关的文档。

        Args:
            query: 用户查询文本
            k: 返回 Top-K 个相关文档
            filter_metadata: 可选的 metadata 过滤条件
                例如 {"source": "PsyQA"} 只检索 PsyQA 来源

        Returns:
            格式化后的参考文本
        """
        if not self._factual_store:
            return "[知识库未初始化]"

        search_kwargs = {"k": k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        docs = self._factual_store.similarity_search(query, **search_kwargs)

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            topic = doc.metadata.get("topic", "")
            parts.append(
                f"[参考 {i} | 来源: {source} | 主题: {topic}]\n{doc.page_content}"
            )
        return "\n\n".join(parts) if parts else "未检索到相关科普内容。"

    def search_guardrail(
        self,
        query: str,
        k: int = GUARDRAIL_RETRIEVAL_K,
    ) -> str:
        """
        从安全护栏库检索相关的安全文档。

        Args:
            query: 用户查询文本
            k: 返回 Top-K 个相关文档

        Returns:
            格式化后的安全参考文本
        """
        if not self._guardrail_store:
            return "[安全护栏库未初始化]"

        docs = self._guardrail_store.similarity_search(query, k=k)
        parts = [doc.page_content for doc in docs]
        return "\n\n".join(parts) if parts else ""

    # ────────────────────────────────────────────
    # 管理
    # ────────────────────────────────────────────

    def get_stats(self) -> dict:
        """返回各 Collection 的统计信息。"""
        return {
            "factual_kb_count": self._factual_store._collection.count()
            if self._factual_store else 0,
            "guardrail_kb_count": self._guardrail_store._collection.count()
            if self._guardrail_store else 0,
            "persist_dir": str(self.persist_dir),
        }

    def clear_collection(self, collection_name: str):
        """
        清空指定的 Collection（谨慎使用）。

        Args:
            collection_name: "factual_kb" 或 "guardrail_kb"
        """
        store = self._get_or_create_collection(collection_name)
        ids = store._collection.get()["ids"]
        if ids:
            store._collection.delete(ids=ids)
            print(f"  [管理] 已清空 {collection_name}: 删除 {len(ids)} 个文档块")
        else:
            print(f"  [管理] {collection_name} 本身就是空的")

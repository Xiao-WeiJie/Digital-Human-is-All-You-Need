# -*- coding: utf-8 -*-
"""
真实数据加载器 — 从 data/processed/ 加载预处理后的数据集。

当你完成了数据下载和预处理之后（运行了 download_datasets.py 和
preprocess_datasets.py），将 data/ 模块中的各加载函数替换为本文件中的实现。

用法：
  1. 完成数据下载:      python scripts/download_datasets.py
  2. 完成数据预处理:    python scripts/preprocess_datasets.py
  3. 将本文件中的函数替换到对应的 data/ 子模块中

或者更简单地，在 data/__init__.py 中切换导入源:

    # 开发阶段：使用模拟数据
    # from data.factual_docs import build_factual_knowledge_docs

    # 正式阶段：使用真实数据
    from data.real_data_loader import (
        build_factual_knowledge_docs,
        build_empathy_examples,
        build_guardrail_docs,
    )
"""

import json
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 预处理数据目录 ──
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

# ── 切块参数（与 config/settings.py 保持一致） ──
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def _load_jsonl(filepath: Path) -> List[dict]:
    """加载 JSONL 格式文件。"""
    items = []
    if not filepath.exists():
        print(f"  ⚠️  文件不存在: {filepath}")
        return items
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ============================================================================
# 优先级一：科普核心库
# ============================================================================
def build_factual_knowledge_docs() -> List[Document]:
    """
    从预处理后的文件加载科普核心库文档。

    数据来源:
      - data/processed/factual_psyqa.jsonl   (PsyQA 问答)
      - data/processed/factual_pdfs.jsonl    (mhGAP-IG + OpenStax)
    """
    all_docs = []

    # ── 加载 PsyQA ──
    psyqa_items = _load_jsonl(PROCESSED_DIR / "factual_psyqa.jsonl")
    for item in psyqa_items:
        all_docs.append(Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        ))

    # ── 加载 PDF 提取内容 ──
    pdf_items = _load_jsonl(PROCESSED_DIR / "factual_pdfs.jsonl")
    for item in pdf_items:
        all_docs.append(Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        ))

    if not all_docs:
        print("  ⚠️  未加载到任何科普核心库文档！请先运行预处理脚本。")
        # 回退到模拟数据
        from data.factual_docs import build_factual_knowledge_docs as fallback
        return fallback()

    # ── 文本切块 ──
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " "],
        length_function=len,
    )
    split_docs = text_splitter.split_documents(all_docs)

    print(f"  [数据注入] 科普核心库(真实数据): "
          f"{len(all_docs)} 篇原始文档 → {len(split_docs)} 个切块")
    return split_docs


# ============================================================================
# 优先级二：共情策略库
# ============================================================================
def build_empathy_examples() -> List[Dict[str, str]]:
    """
    从预处理后的文件加载共情策略 Few-shot 示例。

    数据来源: data/processed/empathy_examples.json
    """
    filepath = PROCESSED_DIR / "empathy_examples.json"

    if not filepath.exists():
        print("  ⚠️  未找到共情示例文件，回退到模拟数据。")
        from data.empathy_examples import build_empathy_examples as fallback
        return fallback()

    with open(filepath, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"  [数据注入] 共情策略库(真实数据): {len(examples)} 条共情回复模板")
    return examples


# ============================================================================
# 优先级三：安全护栏库
# ============================================================================
def build_guardrail_docs() -> List[Document]:
    """
    从预处理后的文件加载安全护栏文档。

    数据来源: data/processed/guardrail_docs.jsonl
    """
    items = _load_jsonl(PROCESSED_DIR / "guardrail_docs.jsonl")

    if not items:
        print("  ⚠️  未找到安全护栏文档，回退到模拟数据。")
        from data.guardrail_docs import build_guardrail_docs as fallback
        return fallback()

    docs = []
    for item in items:
        docs.append(Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        ))

    print(f"  [数据注入] 安全护栏库(真实数据): {len(docs)} 篇安全文档")
    return docs

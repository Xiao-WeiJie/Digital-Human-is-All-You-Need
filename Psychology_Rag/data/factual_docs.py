# -*- coding: utf-8 -*-
"""
优先级一：科普核心库 (Factual Knowledge)

数据来源：
  • thu-coai/PsyQA（清华心理问答数据集）
  • WHO mhGAP-IG（世卫组织精神卫生指南）
  • OpenStax Psychology（开源心理学教材）

处理方式：
  文本切块（Chunking）→ 向量化存入 FAISS
  Metadata 标记为 {"type": "factual", "source": "..."}

注：当前为模拟文档。替换为真实数据示例见 README。
"""
import json
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from pathlib import Path
from typing import List, Dict

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

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

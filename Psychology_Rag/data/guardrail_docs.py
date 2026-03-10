# -*- coding: utf-8 -*-
"""
优先级三：评估工具与安全护栏 (Guardrails & Scales)

数据来源：
  • PHQ-9 和 GAD-7 等量表的【科普说明文档】（非量表原文或计分规则！）

处理方式：
  通过强 Metadata 标记 {"type": "guardrail", "requires_disclaimer": True}
  匹配到此类意图时 → 强制追加免责声明和危机干预热线提示。
"""

from typing import List
from langchain_core.documents import Document
from pathlib import Path
import json

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

def build_guardrail_docs() -> List[Document]:
    """
    从预处理后的文件加载安全护栏文档。

    数据来源: data/processed/guardrail_docs.jsonl
    """
    items = _load_jsonl(PROCESSED_DIR / "guardrail_docs.jsonl")

    if not items:
        print("  ⚠️  未找到安全护栏文档。")
        exit(-1)

    docs = []
    for item in items:
        docs.append(Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        ))

    print(f"  [数据注入] 安全护栏库(真实数据): {len(docs)} 篇安全文档")
    return docs
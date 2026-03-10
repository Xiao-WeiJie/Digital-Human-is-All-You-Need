# -*- coding: utf-8 -*-
"""
优先级二：共情策略库 (Empathy Strategies)

数据来源：
  • CPsyCoun 数据集（复旦等团队开源的咨询对话转录数据）

处理方式：
  提取优质共情回复模板，使用 SemanticSimilarityExampleSelector
  作为 Few-shot 示例注入 Prompt。
  ⚠️ 不作为常规事实检索，不建向量索引！

注：当前为模拟样本。实际使用时从 CPsyCoun 数据集提取。
"""

from typing import List, Dict
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

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

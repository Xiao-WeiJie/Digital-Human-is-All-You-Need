#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集预处理脚本。

将 datasets/ 下的原始数据转换为 PsyMind 系统可直接使用的格式，
输出到 data/processed/ 目录。

运行前请先执行下载脚本:
  python scripts/download_datasets.py

用法:
  python scripts/preprocess_datasets.py

依赖:
  pip install langchain-text-splitters pymupdf  # pymupdf 用于 PDF 解析
"""

import json
import sys
import re
from pathlib import Path
from typing import List, Dict

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# 1. 预处理 PsyQA → 科普核心库文档
# ============================================================================
def preprocess_psyqa():
    """
    将 PsyQA 的 question + answer 转换为 LangChain Document 格式的 JSONL。

    输出格式（每行一个 JSON）:
    {
        "page_content": "问题: ...\n回答: ...",
        "metadata": {"type": "factual", "source": "PsyQA", "topic": "..."}
    }
    """
    print("\n  📄 [1/4] 预处理 PsyQA...")

    train_path = DATASETS_DIR / "psyqa" / "train.json"
    if not train_path.exists():
        print("    ⚠️  未找到 datasets/psyqa/train.json，跳过")
        return False

    output_path = ensure_dir(OUTPUT_DIR) / "factual_psyqa.jsonl"

    # 加载原始数据
    with open(train_path, "r", encoding="utf-8") as f:
        # HuggingFace to_json 输出的是 JSONL 格式
        lines = f.readlines()

    docs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        # 提取字段（兼容不同版本的字段名）
        question = item.get("question", item.get("title", ""))
        answer = item.get("answer", item.get("description", ""))
        topic = item.get("topic", item.get("keywords", "心理健康"))

        if not answer or len(answer) < 50:
            continue  # 跳过过短的回答

        doc = {
            "page_content": f"问题: {question}\n\n回答: {answer}",
            "metadata": {
                "type": "factual",
                "source": "PsyQA",
                "topic": str(topic) if topic else "心理健康",
            },
        }
        docs.append(doc)

    # 写入 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"    ✅ 处理了 {len(docs)} 条问答 → {output_path.name}")
    return True


# ============================================================================
# 2. 预处理 PDF 文档 → 科普核心库文档（mhGAP + OpenStax）
# ============================================================================
def preprocess_pdfs():
    """
    从 mhGAP-IG 和 OpenStax Psychology PDF 中提取文本。

    使用 pymupdf (fitz) 提取文本，然后输出为 JSONL。
    """
    print("\n  📄 [2/4] 预处理 PDF 文档...")

    try:
        import fitz  # pymupdf
    except ImportError:
        print("    ⚠️  需要安装 pymupdf: pip install pymupdf")
        print("    跳过 PDF 预处理")
        return False

    output_path = ensure_dir(OUTPUT_DIR) / "factual_pdfs.jsonl"
    docs = []

    # ── mhGAP-IG ──
    mhgap_path = DATASETS_DIR / "mhgap" / "mhgap-ig-v2.pdf"
    if mhgap_path.exists():
        print("    处理 mhGAP-IG V2.0...")
        pdf_doc = fitz.open(str(mhgap_path))
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            text = page.get_text().strip()
            if len(text) < 100:
                continue  # 跳过内容过少的页面（目录、封面等）
            docs.append({
                "page_content": text,
                "metadata": {
                    "type": "factual",
                    "source": "mhGAP-IG",
                    "topic": f"mhGAP_page_{page_num + 1}",
                },
            })
        pdf_doc.close()
        print(f"    ✅ mhGAP-IG: {len(docs)} 页有效内容")
    else:
        print("    ⚠️  未找到 datasets/mhgap/mhgap-ig-v2.pdf")

    # ── OpenStax Psychology 2e ──
    # 只提取与心理健康最相关的章节（可按需调整页码范围）
    openstax_path = DATASETS_DIR / "openstax_psychology" / "Psychology2e-WEB.pdf"
    if openstax_path.exists():
        print("    处理 OpenStax Psychology 2e（提取相关章节）...")
        pdf_doc = fitz.open(str(openstax_path))

        # 相关章节的大致页码范围（基于 Psychology 2e 的目录结构）
        # 实际使用时建议根据目录精确定位
        relevant_sections = {
            "Ch10_Emotion_and_Motivation": (380, 430),
            "Ch14_Stress_Lifestyle_Health": (530, 570),
            "Ch15_Psychological_Disorders": (570, 640),
            "Ch16_Therapy_and_Treatment": (640, 690),
        }

        count_before = len(docs)
        for section_name, (start, end) in relevant_sections.items():
            actual_end = min(end, len(pdf_doc))
            actual_start = min(start, actual_end)
            for page_num in range(actual_start, actual_end):
                page = pdf_doc[page_num]
                text = page.get_text().strip()
                if len(text) < 100:
                    continue
                docs.append({
                    "page_content": text,
                    "metadata": {
                        "type": "factual",
                        "source": "OpenStax_Psychology",
                        "topic": section_name,
                    },
                })

        pdf_doc.close()
        count_new = len(docs) - count_before
        print(f"    ✅ OpenStax Psychology: {count_new} 页有效内容（4 个相关章节）")
    else:
        print("    ⚠️  未找到 datasets/openstax_psychology/Psychology2e-WEB.pdf")

    # 写入 JSONL
    if docs:
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"    ✅ 共 {len(docs)} 个文档块 → {output_path.name}")

    return len(docs) > 0


# ============================================================================
# 3. 预处理 CPsyCounD → 共情策略 Few-shot 示例
# ============================================================================
def preprocess_cpsycoun():
    """
    从 CPsyCounD 多轮对话中提取高质量的咨询师共情回复。

    ╔══════════════════════════════════════════════════════════════╗
    ║  CPsyCounD 实际数据格式（LLaMA-Factory 格式）：             ║
    ║                                                            ║
    ║  {                                                         ║
    ║    "instruction": "来访者当前这句话",                        ║
    ║    "input": "",           ← 始终为空                       ║
    ║    "output": "咨询师的回复",                                ║
    ║    "history": [                                            ║
    ║      ["来访者第1轮", "咨询师第1轮"],                         ║
    ║      ["来访者第2轮", "咨询师第2轮"],                         ║
    ║      ...                                                   ║
    ║    ]                                                       ║
    ║  }                                                         ║
    ╚══════════════════════════════════════════════════════════════╝

    筛选标准:
      - 咨询师回复长度 > 80 字（有足够的共情内容）
      - 咨询师回复长度 < 500 字（不过于冗长）
      - 来访者消息长度 > 15 字（不是简短的"嗯""好的"）

    输出格式:
    {
        "user_input": "来访者的话...",
        "empathetic_response": "咨询师的共情回复..."
    }
    """
    print("\n  📄 [3/4] 预处理 CPsyCounD → 共情策略示例...")

    cpsycoun_path = DATASETS_DIR / "cpsycoun" / "CPsyCounD.json"
    if not cpsycoun_path.exists():
        # 也尝试查找其他可能的文件名
        alt_paths = list((DATASETS_DIR / "cpsycoun").glob("*.json*"))
        if alt_paths:
            cpsycoun_path = alt_paths[0]
        else:
            print("    ⚠️  未找到 CPsyCounD 数据，跳过")
            return False

    output_path = ensure_dir(OUTPUT_DIR) / "empathy_examples.json"

    # ── 加载数据（兼容 JSONL 和 JSON Array） ──
    dialogues = []
    with open(cpsycoun_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            dialogues = parsed
        elif isinstance(parsed, dict):
            dialogues = [parsed]
    except json.JSONDecodeError:
        for line in content.split("\n"):
            line = line.strip()
            if line:
                try:
                    dialogues.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not dialogues:
        print("    ⚠️  未能解析任何对话数据")
        return False

    # ── 先诊断一下实际字段名 ──
    sample = dialogues[0]
    print(f"    [诊断] 首条数据字段: {list(sample.keys())}")
    if "history" in sample and isinstance(sample["history"], list) and len(sample["history"]) > 0:
        print(f"    [诊断] history[0] 类型: {type(sample['history'][0])}, 长度: {len(sample['history'][0])}")

    # ── 提取共情回复 ──
    examples = []
    for dialogue in dialogues:

        # ━━━ 策略 A：LLaMA-Factory 格式 ━━━
        # 字段: instruction(来访者当前话), output(咨询师回复), history([[来访者,咨询师],...])
        instruction = dialogue.get("instruction", "")
        output = dialogue.get("output", "")
        history = dialogue.get("history", [])

        if instruction and output:
            # 提取当前轮的 instruction → output 对
            if 15 < len(instruction) and 80 < len(output) < 500:
                examples.append({
                    "user_input": instruction.strip(),
                    "empathetic_response": output.strip(),
                })

            # 提取 history 中的每一轮 [来访者, 咨询师] 对
            if isinstance(history, list):
                for turn in history:
                    if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                        client_msg = str(turn[0]).strip()
                        counselor_msg = str(turn[1]).strip()
                        if 15 < len(client_msg) and 80 < len(counselor_msg) < 500:
                            examples.append({
                                "user_input": client_msg,
                                "empathetic_response": counselor_msg,
                            })
            continue

        # ━━━ 策略 B：conversations 格式（兜底） ━━━
        conversations = dialogue.get("conversations", dialogue.get("messages", []))
        if conversations:
            for i in range(len(conversations) - 1):
                curr = conversations[i]
                next_msg = conversations[i + 1]
                curr_role = str(curr.get("role", curr.get("from", ""))).lower()
                next_role = str(next_msg.get("role", next_msg.get("from", ""))).lower()
                curr_content = curr.get("content", curr.get("value", ""))
                next_content = next_msg.get("content", next_msg.get("value", ""))

                is_client = curr_role in ("client", "user", "human", "来访者")
                is_counselor = next_role in ("counselor", "assistant", "gpt", "咨询师")

                if is_client and is_counselor:
                    if 15 < len(curr_content) and 80 < len(next_content) < 500:
                        examples.append({
                            "user_input": curr_content.strip(),
                            "empathetic_response": next_content.strip(),
                        })

    # ── 去重（基于来访者话语的前 50 字符） ──
    seen = set()
    unique_examples = []
    for ex in examples:
        key = ex["user_input"][:50]
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    # ── 保存（限制最多 200 条，避免过大） ──
    final_examples = unique_examples[:200]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_examples, f, ensure_ascii=False, indent=2)

    print(f"    ✅ 从 {len(dialogues)} 段对话中提取了 {len(final_examples)} 条共情示例 "
          f"(去重前 {len(examples)} 条) → {output_path.name}")
    return len(final_examples) > 0


# ============================================================================
# 4. 预处理安全护栏文档
# ============================================================================
def preprocess_guardrails():
    """
    加载安全护栏科普文档，添加强 Metadata 标记。
    """
    print("\n  📄 [4/4] 预处理安全护栏文档...")

    guardrails_dir = DATASETS_DIR / "guardrails"
    output_path = ensure_dir(OUTPUT_DIR) / "guardrail_docs.jsonl"

    if not guardrails_dir.exists():
        print("    ⚠️  未找到 datasets/guardrails/ 目录，跳过")
        return False

    docs = []
    file_topic_map = {
        "phq9_intro.txt": ("PHQ-9_科普说明", "PHQ-9量表科普"),
        "gad7_intro.txt": ("GAD-7_科普说明", "GAD-7量表科普"),
        "crisis_intervention.txt": ("危机干预指引", "危机识别与转介"),
    }

    for filename, (source, topic) in file_topic_map.items():
        filepath = guardrails_dir / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8").strip()
            docs.append({
                "page_content": content,
                "metadata": {
                    "type": "guardrail",
                    "source": source,
                    "requires_disclaimer": True,
                    "topic": topic,
                },
            })
            print(f"    ✅ {filename} → {topic}")
        else:
            print(f"    ⚠️  未找到 {filename}")

    if docs:
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"    ✅ {len(docs)} 篇安全文档 → {output_path.name}")

    return len(docs) > 0


# ============================================================================
# 主入口
# ============================================================================
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     🔧 PsyMind 数据集预处理脚本                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  原始数据目录: {DATASETS_DIR}")
    print(f"  输出目录:     {OUTPUT_DIR}\n")

    ensure_dir(OUTPUT_DIR)

    results = {}
    results["PsyQA → 科普核心库"] = preprocess_psyqa()
    results["PDF → 科普核心库"] = preprocess_pdfs()
    results["CPsyCounD → 共情策略"] = preprocess_cpsycoun()
    results["安全护栏文档"] = preprocess_guardrails()

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("  📊 预处理汇总")
    print("=" * 60)
    for name, success in results.items():
        status = "✅" if success else "⚠️ 跳过"
        print(f"  {status}  {name}")

    # 列出输出文件
    print(f"\n  输出文件:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"    📁 {f.name}  ({size_kb:.1f} KB)")

    print(f"\n  下一步: 修改 data/ 模块中的加载逻辑，读取 data/processed/ 下的文件")
    print(f"  详见 DATA_PREPARATION.md 中的说明\n")


if __name__ == "__main__":
    main()
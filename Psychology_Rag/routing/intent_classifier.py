# -*- coding: utf-8 -*-
"""
意图路由器 (Semantic Router / Intent Classifier)

采用双重保障机制：
  1. LLM 语义分类 — 理解用户真实意图
  2. 关键词硬规则 — 兜底确保危险信号不被漏判

路由逻辑:
┌──────────────┐
│  用户输入     │
└──────┬───────┘
       ▼
┌──────────────────────────────┐
│  LLM 意图分类器               │
│  (zero-shot classification)  │
└──────┬───────────────────────┘
       │
       ├─→ "emotional"   → 情绪宣泄链路
       ├─→ "knowledge"   → 知识问询链路
       ├─→ "guardrail"   → 安全护栏链路
       └─→ "chitchat"    → 闲聊/通用链路
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config.settings import DANGER_KEYWORDS, SCALE_KEYWORDS
from prompts.templates import INTENT_CLASSIFICATION_SYSTEM_PROMPT

# 合法的意图类型
VALID_INTENTS = {"emotional", "knowledge", "guardrail", "chitchat"}


class IntentClassifier:
    """
    基于 LLM 的语义意图分类器。

    对外只暴露一个 classify() 方法，内部自动处理：
      - LLM 分类调用
      - 安全关键词硬规则覆写
      - 结果清洗与兜底
    """

    def __init__(self, llm: ChatOpenAI):
        """
        Args:
            llm: 用于意图分类的语言模型（建议低温度以保证稳定性）
        """
        self._chain = self._build_chain(llm)

    @staticmethod
    def _build_chain(llm: ChatOpenAI):
        """构建 LCEL 分类链。"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            ("human",
             "用户输入：{user_input}\n\n"
             "对话历史摘要（用于理解上下文）：{history_summary}\n\n"
             "意图分类："),
        ])
        return prompt | llm | StrOutputParser()

    async def classify(self, user_input: str, history_summary: str = "") -> str:
        """
        对用户输入进行意图分类。

        Args:
            user_input: 当前用户输入文本
            history_summary: 对话历史摘要（辅助上下文理解）

        Returns:
            意图标签: "emotional" | "knowledge" | "guardrail" | "chitchat"
        """
        # ── Step 1: LLM 语义分类 ──
        raw = await self._chain.ainvoke({
            "user_input": user_input,
            "history_summary": history_summary or "（对话刚开始，暂无历史）",
        })
        intent = raw.strip().lower().strip('"').strip("'")

        # ── Step 2: 安全关键词硬规则覆写 ──
        # 危险信号 → 强制 guardrail（最高优先级）
        if any(kw in user_input for kw in DANGER_KEYWORDS):
            intent = "guardrail"

        # 量表/评估关键词 → guardrail
        if any(kw in user_input for kw in SCALE_KEYWORDS) and intent != "guardrail":
            intent = "guardrail"

        # ── Step 3: 结果清洗 & 兜底 ──
        if intent not in VALID_INTENTS:
            # 无法识别时默认走情绪关怀（更安全的选择）
            intent = "emotional"

        return intent

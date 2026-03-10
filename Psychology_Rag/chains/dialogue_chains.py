# -*- coding: utf-8 -*-
"""
LCEL 对话链工厂 (Chain Factory)

为每种意图构建独立的 LCEL 处理链：
  - emotional_chain:  情绪关怀（共情 Few-shot + 科普 RAG）
  - knowledge_chain:  知识科普（科普核心库 RAG）
  - guardrail_chain:  安全护栏（安全文档 RAG + 强制免责）
  - chitchat_chain:   日常闲聊

每条链的结构统一为:
  ChatPromptTemplate → LLM → StrOutputParser
"""

from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from prompts.templates import (
    SYSTEM_PROMPT_TEMPLATE,
    EMOTIONAL_SUPPORT_ADDENDUM,
    KNOWLEDGE_QUERY_ADDENDUM,
    GUARDRAIL_ADDENDUM,
    CHITCHAT_ADDENDUM,
)


@dataclass
class DialogueChains:
    """持有四条意图链的容器。"""
    emotional: Runnable
    knowledge: Runnable
    guardrail: Runnable
    chitchat: Runnable


class ChainFactory:
    """
    对话链工厂类。

    使用方式:
        chains = ChainFactory.build_all(llm)
        result = await chains.emotional.ainvoke({...})
    """

    @staticmethod
    def _make_chain(llm: ChatOpenAI, addendum: str) -> Runnable:
        """
        构建通用的 LCEL 对话链。

        Args:
            llm: 语言模型实例
            addendum: 拼接在 System Prompt 之后的意图专用指令

        Returns:
            LCEL Runnable 链
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEMPLATE + "\n\n" + addendum),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        return prompt | llm | StrOutputParser()

    @classmethod
    def build_all(cls, llm: ChatOpenAI) -> DialogueChains:
        """
        一次性构建全部四条意图链。

        Args:
            llm: 对话生成用的语言模型

        Returns:
            DialogueChains 实例
        """
        return DialogueChains(
            emotional=cls._make_chain(llm, EMOTIONAL_SUPPORT_ADDENDUM),
            knowledge=cls._make_chain(llm, KNOWLEDGE_QUERY_ADDENDUM),
            guardrail=cls._make_chain(llm, GUARDRAIL_ADDENDUM),
            chitchat=cls._make_chain(llm, CHITCHAT_ADDENDUM),
        )

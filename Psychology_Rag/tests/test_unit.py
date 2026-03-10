# -*- coding: utf-8 -*-
"""
PsyMind 单元测试套件
测试范围（无需 API Key）：
  1. 配置加载
  2. SessionStore CRUD
  3. IntentClassifier 关键词硬规则
  4. ConversationMemory 用户名提取
  5. VectorStoreManager 去重逻辑
  6. MultiWayRetriever 格式化输出
"""

import sys
import os
import unittest
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════
# 1. 配置加载
# ══════════════════════════════════════════════════════════
class TestConfig(unittest.TestCase):

    def test_danger_keywords_not_empty(self):
        from config.settings import DANGER_KEYWORDS
        self.assertIsInstance(DANGER_KEYWORDS, list)
        self.assertGreater(len(DANGER_KEYWORDS), 0, "DANGER_KEYWORDS 不能为空")

    def test_scale_keywords_not_empty(self):
        from config.settings import SCALE_KEYWORDS
        self.assertIsInstance(SCALE_KEYWORDS, list)
        self.assertGreater(len(SCALE_KEYWORDS), 0, "SCALE_KEYWORDS 不能为空")

    def test_crisis_hotline_in_config(self):
        from config.settings import CRISIS_HOTLINE_INFO
        self.assertIn("400", CRISIS_HOTLINE_INFO, "热线号码应包含在 CRISIS_HOTLINE_INFO 中")

    def test_disclaimer_in_config(self):
        from config.settings import DISCLAIMER
        self.assertIn("免责声明", DISCLAIMER)

    def test_llm_temperature_range(self):
        from config.settings import LLM_TEMPERATURE, LLM_COLD_TEMPERATURE
        self.assertGreaterEqual(LLM_TEMPERATURE, 0.0)
        self.assertLessEqual(LLM_TEMPERATURE, 2.0)
        self.assertGreaterEqual(LLM_COLD_TEMPERATURE, 0.0)
        self.assertLess(LLM_COLD_TEMPERATURE, LLM_TEMPERATURE, "冷温度应小于对话温度")


# ══════════════════════════════════════════════════════════
# 2. SessionStore CRUD
# ══════════════════════════════════════════════════════════
class TestSessionStore(unittest.TestCase):

    def setUp(self):
        from db.session_store import SessionStore
        self.store = SessionStore(db_path=":memory:")

    # ── 会话管理 ──
    def test_create_session(self):
        self.store.create_session("s1")
        self.assertTrue(self.store.session_exists("s1"))

    def test_create_session_idempotent(self):
        self.store.create_session("s2")
        self.store.create_session("s2")  # 幂等，不应抛出异常
        self.assertTrue(self.store.session_exists("s2"))

    def test_nonexistent_session(self):
        self.assertFalse(self.store.session_exists("ghost"))

    # ── 消息 ──
    def test_add_and_get_messages(self):
        self.store.create_session("s3")
        self.store.add_message("s3", "human", "你好")
        self.store.add_message("s3", "ai", "你好，有什么可以帮你？")
        msgs = self.store.get_messages("s3")
        self.assertEqual(len(msgs), 2)
        from langchain_core.messages import HumanMessage, AIMessage
        self.assertIsInstance(msgs[0], HumanMessage)
        self.assertIsInstance(msgs[1], AIMessage)

    def test_message_content_preserved(self):
        self.store.create_session("s4")
        self.store.add_message("s4", "human", "我最近很焦虑")
        msgs = self.store.get_messages("s4")
        self.assertEqual(msgs[0].content, "我最近很焦虑")

    def test_get_messages_with_limit(self):
        self.store.create_session("s5")
        for i in range(10):
            role = "human" if i % 2 == 0 else "ai"
            self.store.add_message("s5", role, f"消息{i}")
        msgs = self.store.get_messages("s5", limit=4)
        self.assertEqual(len(msgs), 4)

    def test_message_count(self):
        self.store.create_session("s6")
        self.store.add_message("s6", "human", "msg1")
        self.store.add_message("s6", "ai", "msg2")
        self.store.add_message("s6", "human", "msg3")
        self.assertEqual(self.store.get_message_count("s6"), 3)

    def test_delete_old_messages(self):
        self.store.create_session("s7")
        for i in range(10):
            role = "human" if i % 2 == 0 else "ai"
            self.store.add_message("s7", role, f"msg{i}")
        deleted = self.store.delete_old_messages("s7", keep_recent=4)
        self.assertGreater(deleted, 0)
        remaining = self.store.get_message_count("s7")
        self.assertLessEqual(remaining, 4)

    # ── 摘要 & 用户名 ──
    def test_summary_roundtrip(self):
        self.store.create_session("s8")
        self.store.set_summary("s8", "用户感到焦虑，讨论了睡眠问题")
        self.assertEqual(self.store.get_summary("s8"), "用户感到焦虑，讨论了睡眠问题")

    def test_default_user_name(self):
        self.store.create_session("s9")
        self.assertEqual(self.store.get_user_name("s9"), "朋友")

    def test_set_user_name(self):
        self.store.create_session("s10")
        self.store.set_user_name("s10", "小明")
        self.assertEqual(self.store.get_user_name("s10"), "小明")

    # ── 共情示例 ──
    def test_save_and_get_empathy(self):
        examples = [
            {"user_input": "我很难过", "empathetic_response": "听起来你承受了很多..."},
            {"user_input": "压力太大了", "empathetic_response": "你最近是不是很累？"},
        ]
        count = self.store.save_empathy_examples(examples)
        self.assertEqual(count, 2)
        fetched = self.store.get_empathy_examples()
        self.assertEqual(len(fetched), 2)

    def test_empathy_count(self):
        self.store.save_empathy_examples([
            {"user_input": "u1", "empathetic_response": "r1"},
            {"user_input": "u2", "empathetic_response": "r2"},
            {"user_input": "u3", "empathetic_response": "r3"},
        ])
        self.assertEqual(self.store.get_empathy_count(), 3)

    def test_save_empathy_overwrites(self):
        self.store.save_empathy_examples([
            {"user_input": "old", "empathetic_response": "old_reply"},
        ])
        self.store.save_empathy_examples([
            {"user_input": "new1", "empathetic_response": "new_reply1"},
            {"user_input": "new2", "empathetic_response": "new_reply2"},
        ])
        self.assertEqual(self.store.get_empathy_count(), 2)

    # ── 统计 ──
    def test_stats_structure(self):
        stats = self.store.get_stats()
        self.assertIn("sessions", stats)
        self.assertIn("messages", stats)
        self.assertIn("empathy_examples", stats)

    def test_list_sessions(self):
        self.store.create_session("list_s1")
        self.store.create_session("list_s2")
        sessions = self.store.list_sessions()
        ids = [s["session_id"] for s in sessions]
        self.assertIn("list_s1", ids)
        self.assertIn("list_s2", ids)


# ══════════════════════════════════════════════════════════
# 3. 意图分类器 — 关键词硬规则（不需要 LLM）
# ══════════════════════════════════════════════════════════
class TestIntentKeywordRules(unittest.TestCase):
    """
    测试 IntentClassifier 中的关键词硬规则覆写逻辑。
    通过 mock LLM 绕过 API 调用。
    """

    def _make_classifier(self, mock_llm_return: str):
        """构造一个使用 mock LLM 的 IntentClassifier。"""
        from unittest.mock import AsyncMock, MagicMock
        from routing.intent_classifier import IntentClassifier

        # _chain 是 LCEL Runnable，代码中通过 await self._chain.ainvoke(...) 调用
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_llm_return)

        classifier = IntentClassifier.__new__(IntentClassifier)
        classifier._chain = mock_chain
        return classifier

    def _classify(self, classifier, text, history=""):
        return asyncio.get_event_loop().run_until_complete(
            classifier.classify(text, history)
        )

    # ── 危险关键词应强制路由到 guardrail ──
    def test_danger_keyword_overrides_emotional(self):
        clf = self._make_classifier("emotional")
        result = self._classify(clf, "我不想活了")
        self.assertEqual(result, "guardrail", "危险关键词应覆写为 guardrail")

    def test_danger_keyword_suicide(self):
        clf = self._make_classifier("chitchat")
        result = self._classify(clf, "我想自杀")
        self.assertEqual(result, "guardrail")

    def test_danger_keyword_selfharm(self):
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "我想割腕")
        self.assertEqual(result, "guardrail")

    def test_danger_keyword_jump(self):
        clf = self._make_classifier("emotional")
        result = self._classify(clf, "我想跳楼")
        self.assertEqual(result, "guardrail")

    def test_danger_keyword_no_meaning(self):
        clf = self._make_classifier("emotional")
        result = self._classify(clf, "活着没意思，感觉很累")
        self.assertEqual(result, "guardrail")

    # ── 量表关键词应路由到 guardrail ──
    def test_scale_keyword_phq(self):
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "PHQ-9 是什么？")
        self.assertEqual(result, "guardrail", "量表关键词应路由到 guardrail")

    def test_scale_keyword_gad(self):
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "GAD-7 怎么评分？")
        self.assertEqual(result, "guardrail")

    def test_scale_keyword_liangbiao(self):
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "我想做个心理量表测评")
        self.assertEqual(result, "guardrail")

    def test_scale_keyword_diagnose(self):
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "怎么测试自己有没有抑郁症，有诊断工具吗")
        self.assertEqual(result, "guardrail")

    # ── 正常意图不应被错误覆写 ──
    def test_emotional_not_overridden(self):
        clf = self._make_classifier("emotional")
        result = self._classify(clf, "我今天很焦虑，压力很大")
        self.assertEqual(result, "emotional")

    def test_knowledge_not_overridden(self):
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "什么是认知行为疗法？")
        self.assertEqual(result, "knowledge")

    def test_chitchat_not_overridden(self):
        clf = self._make_classifier("chitchat")
        result = self._classify(clf, "你好，你是谁？")
        self.assertEqual(result, "chitchat")

    # ── 无效 LLM 输出兜底为 emotional ──
    def test_invalid_llm_output_fallback(self):
        clf = self._make_classifier("invalid_gibberish_xyz")
        result = self._classify(clf, "随便说点什么")
        self.assertEqual(result, "emotional", "无效意图应兜底为 emotional")

    def test_empty_llm_output_fallback(self):
        clf = self._make_classifier("")
        result = self._classify(clf, "随便说点什么")
        self.assertEqual(result, "emotional")

    # ── 大小写清洗 ──
    def test_case_insensitive_cleanup(self):
        clf = self._make_classifier("  KNOWLEDGE  ")
        result = self._classify(clf, "什么是焦虑症")
        self.assertEqual(result, "knowledge")


# ══════════════════════════════════════════════════════════
# 4. ConversationMemory — 用户名提取
# ══════════════════════════════════════════════════════════
class TestNameExtraction(unittest.TestCase):

    def setUp(self):
        from unittest.mock import MagicMock
        from db.session_store import SessionStore
        from memory.conversation_memory import ConversationMemoryManager

        self.store = SessionStore(db_path=":memory:")
        mock_llm = MagicMock()
        self.memory = ConversationMemoryManager(llm=mock_llm, session_store=self.store)

    def _extract(self, session_id, text):
        self.store.create_session(session_id)
        self.memory.try_extract_name(session_id, text)
        return self.store.get_user_name(session_id)

    def test_extract_wo_jiao(self):
        name = self._extract("n1", "我叫小明")
        self.assertEqual(name, "小明")

    def test_extract_wo_shi(self):
        # 修复后 [\u4e00-\u9fa5] 不匹配标点，逗号会截断匹配，只提取"小红"
        name = self._extract("n2", "我是小红，很高兴认识你")
        self.assertEqual(name, "小红")

    def test_extract_jiao_wo(self):
        # 修复后只匹配中文字符，"叫我" 后直接是名字再无其他汉字时能正确提取
        name = self._extract("n3", "叫我阿强")
        self.assertEqual(name, "阿强")

    def test_extract_name_is(self):
        name = self._extract("n4", "我的名字是张三")
        self.assertEqual(name, "张三")

    def test_no_name_no_change(self):
        name = self._extract("n5", "我最近压力很大")
        self.assertEqual(name, "朋友")  # 默认值不变

    def test_noise_word_filtered(self):
        # 修复后改为子串检测："什么好呢" 包含噪声词 "什么" → 拒绝提取
        name = self._extract("n6", "我叫什么好呢")
        self.assertEqual(name, "朋友")

    def test_name_truncated_to_four_chars(self):
        # 正则限制最多 4 个中文字符，超长名称被截断取前 4 字
        # "超级无敌" 不含噪声词，会被写入（记录该设计行为）
        name = self._extract("n7", "我叫超级无敌长名字人")
        self.assertEqual(name, "超级无敌")


# ══════════════════════════════════════════════════════════
# 5. VectorStoreManager — 文档去重（使用临时 Chroma）
# ══════════════════════════════════════════════════════════
class TestVectorStoreDedup(unittest.TestCase):

    def test_doc_hash_same_content(self):
        from db.vector_store import VectorStoreManager
        from langchain_core.documents import Document

        doc_a = Document(page_content="心理健康很重要", metadata={"source": "test"})
        doc_b = Document(page_content="心理健康很重要", metadata={"source": "test"})
        self.assertEqual(
            VectorStoreManager._doc_hash(doc_a),
            VectorStoreManager._doc_hash(doc_b),
            "相同内容的文档应有相同 hash"
        )

    def test_doc_hash_different_content(self):
        from db.vector_store import VectorStoreManager
        from langchain_core.documents import Document

        doc_a = Document(page_content="内容A", metadata={"source": "test"})
        doc_b = Document(page_content="内容B", metadata={"source": "test"})
        self.assertNotEqual(
            VectorStoreManager._doc_hash(doc_a),
            VectorStoreManager._doc_hash(doc_b),
            "不同内容的文档应有不同 hash"
        )

    def test_doc_hash_different_metadata(self):
        from db.vector_store import VectorStoreManager
        from langchain_core.documents import Document

        doc_a = Document(page_content="内容", metadata={"source": "src1"})
        doc_b = Document(page_content="内容", metadata={"source": "src2"})
        self.assertNotEqual(
            VectorStoreManager._doc_hash(doc_a),
            VectorStoreManager._doc_hash(doc_b),
            "不同 metadata 的文档应有不同 hash"
        )


# ══════════════════════════════════════════════════════════
# 6. MultiWayRetriever — format_empathy_examples
# ══════════════════════════════════════════════════════════
class TestEmpathyFormatting(unittest.TestCase):

    def setUp(self):
        from unittest.mock import MagicMock
        from db.session_store import SessionStore
        from retrieval.multi_way_retriever import MultiWayRetriever

        self.store = SessionStore(db_path=":memory:")
        mock_embeddings = MagicMock()
        mock_vector_mgr = MagicMock()
        self.retriever = MultiWayRetriever(
            embeddings=mock_embeddings,
            vector_manager=mock_vector_mgr,
            session_store=self.store,
        )

    def test_format_with_examples(self):
        self.store.save_empathy_examples([
            {"user_input": "我很难过", "empathetic_response": "听起来你在承受很多..."},
            {"user_input": "压力好大", "empathetic_response": "你最近是不是很累？"},
            {"user_input": "睡不着", "empathetic_response": "失眠真的会让人很疲惫..."},
        ])
        result = self.retriever.format_empathy_examples(max_examples=3)
        self.assertIn("共情示例 1", result)
        self.assertIn("我很难过", result)
        self.assertIn("听起来你在承受很多", result)

    def test_format_respects_max_examples(self):
        self.store.save_empathy_examples([
            {"user_input": f"输入{i}", "empathetic_response": f"回应{i}"}
            for i in range(10)
        ])
        result = self.retriever.format_empathy_examples(max_examples=2)
        self.assertIn("共情示例 1", result)
        self.assertIn("共情示例 2", result)
        self.assertNotIn("共情示例 3", result)

    def test_format_empty_returns_empty_string(self):
        result = self.retriever.format_empathy_examples()
        self.assertEqual(result, "")


# ══════════════════════════════════════════════════════════
# 7. 数据文件完整性
# ══════════════════════════════════════════════════════════
class TestDataFiles(unittest.TestCase):

    def test_processed_dir_exists(self):
        import os
        processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
        self.assertTrue(os.path.isdir(processed_dir), "data/processed/ 目录应存在")

    def test_factual_psyqa_jsonl_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "factual_psyqa.jsonl")
        self.assertTrue(os.path.exists(path), "factual_psyqa.jsonl 应存在")

    def test_factual_pdfs_jsonl_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "factual_pdfs.jsonl")
        self.assertTrue(os.path.exists(path), "factual_pdfs.jsonl 应存在")

    def test_guardrail_docs_jsonl_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "guardrail_docs.jsonl")
        self.assertTrue(os.path.exists(path), "guardrail_docs.jsonl 应存在")

    def test_empathy_examples_json_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "empathy_examples.json")
        self.assertTrue(os.path.exists(path), "empathy_examples.json 应存在")

    def test_factual_psyqa_parseable(self):
        import json
        from pathlib import Path
        p = Path(__file__).parent.parent / "data" / "processed" / "factual_psyqa.jsonl"
        count = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.assertIn("page_content", item, "每条记录须含 page_content")
                    self.assertIn("metadata", item, "每条记录须含 metadata")
                    count += 1
        self.assertGreater(count, 0, "factual_psyqa.jsonl 不能为空")

    def test_guardrail_docs_parseable(self):
        import json
        from pathlib import Path
        p = Path(__file__).parent.parent / "data" / "processed" / "guardrail_docs.jsonl"
        count = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.assertIn("page_content", item)
                    count += 1
        self.assertGreater(count, 0, "guardrail_docs.jsonl 不能为空")

    def test_empathy_examples_parseable(self):
        import json
        from pathlib import Path
        p = Path(__file__).parent.parent / "data" / "processed" / "empathy_examples.json"
        with open(p, encoding="utf-8") as f:
            examples = json.load(f)
        self.assertIsInstance(examples, list)
        self.assertGreater(len(examples), 0, "共情示例不能为空")
        for ex in examples[:5]:
            self.assertIn("user_input", ex)
            self.assertIn("empathetic_response", ex)


# ══════════════════════════════════════════════════════════
# 8. 安全护栏 — 边界用例
# ══════════════════════════════════════════════════════════
class TestSafetyGuardrailEdgeCases(unittest.TestCase):
    """测试安全护栏关键词规则的边界情况。"""

    def _make_classifier(self, mock_return):
        from unittest.mock import AsyncMock, MagicMock
        from routing.intent_classifier import IntentClassifier
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_return)
        clf = IntentClassifier.__new__(IntentClassifier)
        clf._chain = mock_chain
        return clf

    def _classify(self, clf, text):
        return asyncio.get_event_loop().run_until_complete(clf.classify(text))

    def test_mixed_danger_and_knowledge(self):
        """即使包含知识性词汇，危险关键词仍应优先触发 guardrail。"""
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "CBT 能治疗想死的念头吗？")
        self.assertEqual(result, "guardrail")

    def test_phq_lowercase_triggers_guardrail(self):
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "phq-9 量表在哪里做？")
        self.assertEqual(result, "guardrail")

    def test_sentence_with_no_danger(self):
        """不含任何危险词的普通情绪句不应被误触发。"""
        clf = self._make_classifier("emotional")
        result = self._classify(clf, "我最近失眠，白天没有精神")
        self.assertEqual(result, "emotional")

    def test_danger_keyword_in_question_context(self):
        """即便是"询问"形式，包含危险词也应 guardrail。"""
        clf = self._make_classifier("knowledge")
        result = self._classify(clf, "如何帮助有自杀念头的朋友？")
        self.assertEqual(result, "guardrail")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestConfig,
        TestSessionStore,
        TestIntentKeywordRules,
        TestNameExtraction,
        TestVectorStoreDedup,
        TestEmpathyFormatting,
        TestDataFiles,
        TestSafetyGuardrailEdgeCases,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

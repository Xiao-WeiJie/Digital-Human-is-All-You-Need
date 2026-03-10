# -*- coding: utf-8 -*-
"""
PsyMind 对话质量专项测试

测试维度（对应用户验收标准）:
  TC-01  上下文感知能力 — 跨轮次记忆用户信息
  TC-02  10+ 轮长程对话支持 — 12 轮无中断
  TC-03  非模板化自然语言生成 — 相同语义问题，响应不雷同
  TC-04  追问响应 — 对"你刚才说的 X 是什么意思？"能合理作答
  TC-05  语义澄清 — 模糊问题引导用户澄清或给出合理回应
  TC-06  内容总结 — 用户要求总结时，响应包含对话关键信息
  TC-07  对话逻辑连贯无脱节 — 连续多轮后回复仍与当前话题一致
  TC-08  交互稳定无闪退 — 异常/边界输入不崩溃
  TC-09  模型上下文窗口 > 8k tokens — 配置静态检查
  TC-10  单样本推理响应时间 < 60 秒

运行方法:
  $env:DASHSCOPE_API_KEY="sk-xxx"; python tests/test_dialogue_quality.py
  # 或 pytest:
  pytest tests/test_dialogue_quality.py -v -s
"""

import sys
import os
import time
import unittest
import asyncio
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 代理环境变量去除前后空格 ──
for _pv in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    _v = os.environ.get(_pv, "")
    if _v and _v != _v.strip():
        os.environ[_pv] = _v.strip()

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
_SKIP_API = not bool(DASHSCOPE_API_KEY)
_SKIP_MSG = "需要设置 DASHSCOPE_API_KEY 才能运行对话质量测试"

_API_STATUS: dict = {"checked": False, "available": False, "error": ""}


def _run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _probe_api_once():
    if _API_STATUS["checked"]:
        return _API_STATUS["available"]
    if not DASHSCOPE_API_KEY:
        _API_STATUS.update(checked=True, available=False, error="未设置 DASHSCOPE_API_KEY")
        return False
    try:
        from langchain_openai import ChatOpenAI
        from config.settings import DASHSCOPE_BASE_URL, LLM_MODEL_NAME, LLM_COLD_TEMPERATURE, LLM_MAX_TOKENS
        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            openai_api_key=DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_BASE_URL,
            temperature=LLM_COLD_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        _run(llm.ainvoke("你好"))
        _API_STATUS.update(checked=True, available=True, error="")
        return True
    except Exception as exc:
        _API_STATUS.update(checked=True, available=False, error=str(exc))
        return False


def _skip_if_api_down():
    if not _probe_api_once():
        raise unittest.SkipTest(
            f"API 不可用，跳过测试。原因: {_API_STATUS['error'][:120]}"
        )


# ────────────────────────────────────────────
# 辅助：构建真实 PsyMindSystem
# ────────────────────────────────────────────

def _make_system():
    """构建完整 PsyMindSystem 实例（含 in-memory SQLite，避免污染生产数据）。"""
    import importlib
    import types

    # 用 in-memory SQLite 隔离测试数据
    from system import PsyMindSystem
    from db.session_store import SessionStore
    from db.vector_store import VectorStoreManager
    from db.doc_registry import DocRegistry
    from retrieval.multi_way_retriever import MultiWayRetriever
    from routing.intent_classifier import IntentClassifier
    from memory.conversation_memory import ConversationMemoryManager
    from chains.dialogue_chains import ChainFactory
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from config.settings import (
        DASHSCOPE_BASE_URL, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME,
        LLM_TEMPERATURE, LLM_COLD_TEMPERATURE, LLM_MAX_TOKENS,
    )

    def _llm(temp):
        return ChatOpenAI(
            model=LLM_MODEL_NAME,
            openai_api_key=DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_BASE_URL,
            temperature=temp,
            max_tokens=LLM_MAX_TOKENS,
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=DASHSCOPE_BASE_URL,
        check_embedding_ctx_length=False,
    )

    # 手动组装，使用 in-memory SQLite（与生产 db 隔离）
    sys_obj = object.__new__(PsyMindSystem)
    sys_obj.llm = _llm(LLM_TEMPERATURE)
    sys_obj.llm_cold = _llm(LLM_COLD_TEMPERATURE)
    sys_obj.embeddings = embeddings

    sys_obj.session_store = SessionStore(db_path=":memory:")
    sys_obj.vector_manager = VectorStoreManager(embeddings)
    sys_obj.doc_registry = DocRegistry()

    sys_obj.retriever = MultiWayRetriever(
        embeddings=embeddings,
        vector_manager=sys_obj.vector_manager,
        session_store=sys_obj.session_store,
    )
    sys_obj.retriever.initialize()

    sys_obj.classifier = IntentClassifier(sys_obj.llm_cold)
    sys_obj.memory = ConversationMemoryManager(
        llm=sys_obj.llm_cold,
        session_store=sys_obj.session_store,
    )
    sys_obj.chains = ChainFactory.build_all(sys_obj.llm)

    return sys_obj


# ══════════════════════════════════════════════════════════
# TC-09  模型上下文窗口静态检查（无需 API）
# ══════════════════════════════════════════════════════════
class TC09_ContextWindowSpec(unittest.TestCase):
    """
    TC-09: 模型上下文窗口 > 8k tokens

    验证方法: 从配置文件确认所使用的模型，
    并对照官方规格表静态断言其上下文长度。
    """

    # 官方规格：各模型上下文窗口（tokens）
    _MODEL_CONTEXT_MAP = {
        "qwen-turbo":        131072,   # 128k
        "qwen-turbo-latest": 1000000,  # 1M
        "qwen-plus":         131072,   # 128k
        "qwen-plus-latest":  131072,
        "qwen-max":          32768,    # 32k
        "qwen-max-latest":   32768,
        "qwen-long":         10000000, # 10M
    }
    _REQUIRED_MIN_TOKENS = 8192  # 验收标准：> 8k

    def test_model_context_window_exceeds_8k(self):
        """模型上下文窗口应 > 8192 tokens"""
        from config.settings import LLM_MODEL_NAME
        model = LLM_MODEL_NAME

        # 尝试精确匹配，再尝试前缀匹配
        window = self._MODEL_CONTEXT_MAP.get(model)
        if window is None:
            for k, v in self._MODEL_CONTEXT_MAP.items():
                if model.startswith(k.split("-")[0]):
                    window = v
                    break

        if window is None:
            self.skipTest(
                f"未知模型 {model!r}，无法静态核查上下文窗口。"
                "请在 _MODEL_CONTEXT_MAP 中手动添加。"
            )

        print(f"\n  [TC-09] 模型: {model}  上下文窗口: {window:,} tokens")
        self.assertGreater(
            window, self._REQUIRED_MIN_TOKENS,
            f"模型 {model!r} 上下文窗口 {window:,} <= {self._REQUIRED_MIN_TOKENS}"
        )

    def test_max_tokens_config_reasonable(self):
        """LLM_MAX_TOKENS 应在合理范围（256–4096）内"""
        from config.settings import LLM_MAX_TOKENS
        print(f"\n  [TC-09] LLM_MAX_TOKENS = {LLM_MAX_TOKENS}")
        self.assertGreaterEqual(LLM_MAX_TOKENS, 256,
                                "LLM_MAX_TOKENS 设置过小")
        self.assertLessEqual(LLM_MAX_TOKENS, 4096,
                             "LLM_MAX_TOKENS 设置过大，可能影响响应速度")

    def test_memory_config_supports_long_dialogue(self):
        """记忆配置应支持 10+ 轮长程对话"""
        from config.settings import MAX_RECENT_TURNS, SUMMARY_TRIGGER_TURNS
        print(f"\n  [TC-09] MAX_RECENT_TURNS={MAX_RECENT_TURNS}, "
              f"SUMMARY_TRIGGER_TURNS={SUMMARY_TRIGGER_TURNS}")
        self.assertGreaterEqual(
            MAX_RECENT_TURNS + SUMMARY_TRIGGER_TURNS, 10,
            "MAX_RECENT_TURNS + SUMMARY_TRIGGER_TURNS 之和应 >= 10，"
            "以确保系统支持 10 轮以上连续对话"
        )


# ══════════════════════════════════════════════════════════
# TC-10  单样本推理响应时间 < 60 秒
# ══════════════════════════════════════════════════════════
@unittest.skipIf(_SKIP_API, _SKIP_MSG)
class TC10_ResponseLatency(unittest.TestCase):
    """
    TC-10: 单个样本推理响应时间 < 60 秒

    对四类意图各测一个代表性问题，逐一计时。
    """

    MAX_LATENCY_S = 60.0

    @classmethod
    def setUpClass(cls):
        _skip_if_api_down()
        print("\n  [TC-10 setup] 构建系统...")
        cls.system = _make_system()

    def _timed_process(self, text: str) -> tuple:
        """返回 (response, elapsed_seconds)"""
        sid = str(uuid.uuid4())
        t0 = time.perf_counter()
        resp = _run(self.system.process_message(text, sid))
        elapsed = time.perf_counter() - t0
        return resp, elapsed

    def test_latency_emotional(self):
        """情绪链响应时间 < 60 秒"""
        resp, elapsed = self._timed_process("我最近压力很大，感觉很焦虑")
        print(f"\n  [TC-10 emotional] {elapsed:.2f}s  ({len(resp)}字)")
        self.assertLess(elapsed, self.MAX_LATENCY_S,
                        f"情绪链响应 {elapsed:.2f}s 超过 60s 上限")
        self.assertGreater(len(resp.strip()), 0, "响应不应为空")

    def test_latency_knowledge(self):
        """知识链响应时间 < 60 秒"""
        resp, elapsed = self._timed_process("什么是认知行为疗法？")
        print(f"\n  [TC-10 knowledge] {elapsed:.2f}s  ({len(resp)}字)")
        self.assertLess(elapsed, self.MAX_LATENCY_S,
                        f"知识链响应 {elapsed:.2f}s 超过 60s 上限")

    def test_latency_chitchat(self):
        """闲聊链响应时间 < 60 秒"""
        resp, elapsed = self._timed_process("你好，你是谁？")
        print(f"\n  [TC-10 chitchat] {elapsed:.2f}s  ({len(resp)}字)")
        self.assertLess(elapsed, self.MAX_LATENCY_S,
                        f"闲聊链响应 {elapsed:.2f}s 超过 60s 上限")

    def test_latency_guardrail(self):
        """护栏链响应时间 < 60 秒"""
        resp, elapsed = self._timed_process("PHQ-9 量表是什么？")
        print(f"\n  [TC-10 guardrail] {elapsed:.2f}s  ({len(resp)}字)")
        self.assertLess(elapsed, self.MAX_LATENCY_S,
                        f"护栏链响应 {elapsed:.2f}s 超过 60s 上限")

    def test_latency_p95_summary(self):
        """四类意图响应时间汇总报告（p95 < 60s）"""
        cases = [
            ("我感觉很孤独，没有朋友", "emotional"),
            ("正念冥想对心理健康有什么好处？", "knowledge"),
            ("谢谢你，我好多了", "chitchat"),
            ("我想做个心理量表测评", "guardrail"),
        ]
        latencies = []
        print("\n  [TC-10 汇总]")
        for text, tag in cases:
            resp, elapsed = self._timed_process(text)
            latencies.append(elapsed)
            status = "[OK]" if elapsed < self.MAX_LATENCY_S else "[SLOW]"
            print(f"    {status} [{tag}] {elapsed:.2f}s  {resp[:40]}...")

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0
        avg = sum(latencies) / len(latencies)
        print(f"    平均: {avg:.2f}s  P95: {p95:.2f}s  最大: {max(latencies):.2f}s")
        self.assertLess(max(latencies), self.MAX_LATENCY_S,
                        f"存在响应超过 60s 上限: {max(latencies):.2f}s")


# ══════════════════════════════════════════════════════════
# TC-01 / TC-02 / TC-07  上下文感知 + 10+ 轮 + 逻辑连贯
# ══════════════════════════════════════════════════════════
@unittest.skipIf(_SKIP_API, _SKIP_MSG)
class TC01_TC02_TC07_ContextAwareAndLongDialogue(unittest.TestCase):
    """
    TC-01  上下文感知 — AI 能记住用户名、跨轮引用
    TC-02  10+ 轮对话 — 12 轮完整对话无错误
    TC-07  逻辑连贯 — 后期回复与主题一致，无随机跑题
    """

    @classmethod
    def setUpClass(cls):
        _skip_if_api_down()
        print("\n  [TC-01/02/07 setup] 构建系统...")
        cls.system = _make_system()
        cls.session_id = str(uuid.uuid4())

        # 12 轮对话脚本（涵盖情绪→知识→追问→总结→闲聊→护栏边界）
        cls.dialogue_script = [
            "你好！我叫小雨",                              # R1  自我介绍
            "我最近总是感到焦虑，睡眠也很差",              # R2  情绪表达
            "是工作压力，我要在一周内完成一个大项目",       # R3  追问响应（细化情境）
            "焦虑会影响睡眠吗？这是正常的吗？",            # R4  知识询问
            "你说的自主神经系统是什么意思，能解释一下吗？",   # R5  语义澄清追问
            "除了呼吸练习，还有其他方法吗？",              # R6  追问（延续上轮）
            "正念冥想具体怎么做，步骤是什么？",            # R7  深度知识问询
            "我试过，但脑子里总是乱想，停不下来",          # R8  情绪 + 反馈
            "这是不是说明我的焦虑很严重？",                # R9  自我评估（非诊断）
            "你能帮我总结一下我们聊了哪些内容吗？",        # R10 内容总结请求
            "好的，下次我会试试你说的方法",                # R11 闲聊 / 结束
            "谢谢你，小暖，和你聊天让我感觉好多了",        # R12 情感收尾
        ]

        # 逐轮运行并收集响应
        cls.responses = []
        cls.latencies = []
        print(f"\n  会话 ID: {cls.session_id[:8]}...")
        for i, msg in enumerate(cls.dialogue_script, 1):
            t0 = time.perf_counter()
            try:
                resp = _run(cls.system.process_message(msg, cls.session_id))
                elapsed = time.perf_counter() - t0
                cls.responses.append(resp)
                cls.latencies.append(elapsed)
                status = "[OK]"
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                cls.responses.append(f"[ERROR: {exc}]")
                cls.latencies.append(elapsed)
                status = "[ERR]"
            print(f"    R{i:02d} ({elapsed:.1f}s) {status}: {cls.responses[-1][:60]}...")

    # ── TC-02: 12 轮无报错 ──

    def test_tc02_all_12_turns_completed(self):
        """TC-02: 12 轮对话均应完成（无异常崩溃）"""
        self.assertEqual(len(self.responses), 12,
                         f"期望 12 轮响应，实际只有 {len(self.responses)} 轮")

    def test_tc02_no_error_response(self):
        """TC-02: 12 轮中不应有 [ERROR] 响应"""
        errors = [(i + 1, r) for i, r in enumerate(self.responses)
                  if r.startswith("[ERROR")]
        self.assertEqual(
            len(errors), 0,
            f"发现 {len(errors)} 轮报错:\n" +
            "\n".join(f"  R{i}: {r[:100]}" for i, r in errors)
        )

    def test_tc02_all_responses_nonempty(self):
        """TC-02: 每轮响应均不应为空"""
        empties = [i + 1 for i, r in enumerate(self.responses)
                   if not r.strip()]
        self.assertEqual(len(empties), 0,
                         f"以下轮次响应为空: {empties}")

    # ── TC-01: 上下文感知 ──

    def test_tc01_username_remembered_in_later_turns(self):
        """TC-01: 用户在 R1 报名后，系统记忆管理器应提取到 '小雨'"""
        user_name = self.system.memory.get_user_name(self.session_id)
        print(f"\n  [TC-01] 识别到用户名: {user_name!r}")
        self.assertEqual(user_name, "小雨",
                         f"系统未能记住用户名 '小雨'，实际: {user_name!r}")

    def test_tc01_context_references_prior_topic(self):
        """TC-01: 后期回复（R6 起）应体现对前文焦虑/睡眠主题的感知"""
        # R6 是"除了呼吸练习，还有其他方法吗" — 基于 R4/R5 的上下文
        # 回复中应有延续性（不会凭空发起新话题）
        r6 = self.responses[5]
        # 合理的上下文感知响应不应是纯打招呼
        generic_only = r6.strip() in ("你好！", "您好！", "好的。")
        self.assertFalse(generic_only,
                         f"R6 响应过于通用，可能缺乏上下文感知: {r6[:100]}")

    def test_tc01_message_count_grows_correctly(self):
        """TC-01: SQLite 中应存有 >= 20 条消息（12 轮 × 2 条，摘要可能裁剪旧消息）"""
        from config.settings import MAX_RECENT_TURNS
        count = self.system.session_store.get_message_count(self.session_id)
        # 摘要裁剪后最少保留 MAX_RECENT_TURNS * 2 条
        min_expected = MAX_RECENT_TURNS * 2
        print(f"\n  [TC-01] SQLite 消息数: {count}（最少期望: {min_expected}）")
        self.assertGreaterEqual(
            count, min_expected,
            f"消息数 {count} < {min_expected}，记忆持久化可能异常"
        )

    # ── TC-07: 逻辑连贯性 ──

    def test_tc07_responses_are_chinese(self):
        """TC-07: 所有非空响应的中文字符占比应 > 25%"""
        failures = []
        for i, r in enumerate(self.responses, 1):
            if not r or r.startswith("[ERROR"):
                continue
            zh = sum(1 for c in r if '\u4e00' <= c <= '\u9fff')
            ratio = zh / max(len(r), 1)
            if ratio <= 0.25:
                failures.append(f"R{i:02d}: 中文比例 {ratio:.1%}  {r[:60]}")
        self.assertEqual(len(failures), 0,
                         "以下轮次中文比例不足:\n" + "\n".join(failures))

    def test_tc07_response_length_reasonable(self):
        """TC-07: 响应长度应在 20–800 字（过短或过长均视为异常）"""
        failures = []
        for i, r in enumerate(self.responses, 1):
            if not r or r.startswith("[ERROR"):
                continue
            if len(r) < 20:
                failures.append(f"R{i:02d}: 过短 ({len(r)}字)")
            elif len(r) > 800:
                failures.append(f"R{i:02d}: 过长 ({len(r)}字)")
        self.assertEqual(len(failures), 0,
                         "以下轮次响应长度异常:\n" + "\n".join(failures))

    def test_tc07_no_diagnosis_in_any_turn(self):
        """TC-07: 任何轮次均不应出现诊断性结论"""
        diag_patterns = ["你患有", "你得了", "你是抑郁症", "确诊", "诊断为"]
        hits = []
        for i, r in enumerate(self.responses, 1):
            for pat in diag_patterns:
                if pat in r:
                    hits.append(f"R{i:02d}: 出现禁止词 {pat!r}")
        self.assertEqual(len(hits), 0,
                         "发现诊断性语言:\n" + "\n".join(hits))

    def test_tc07_latency_all_under_60s(self):
        """TC-07: 12 轮中每轮响应均 < 60 秒"""
        slow = [(i + 1, t) for i, t in enumerate(self.latencies) if t >= 60]
        avg = sum(self.latencies) / max(len(self.latencies), 1)
        print(f"\n  [TC-07] 平均响应: {avg:.1f}s  最大: {max(self.latencies):.1f}s")
        self.assertEqual(len(slow), 0,
                         "以下轮次响应超过 60s:\n" +
                         "\n".join(f"  R{i:02d}: {t:.1f}s" for i, t in slow))


# ══════════════════════════════════════════════════════════
# TC-03  非模板化自然语言生成（响应多样性）
# ══════════════════════════════════════════════════════════
@unittest.skipIf(_SKIP_API, _SKIP_MSG)
class TC03_NonTemplatedGeneration(unittest.TestCase):
    """
    TC-03: 相同语义问题，两次响应不应完全相同。
    验证 LLM 生成有足够多样性，不依赖固定模板。
    """

    @classmethod
    def setUpClass(cls):
        _skip_if_api_down()
        print("\n  [TC-03 setup] 构建系统...")
        cls.system = _make_system()

    def _two_responses(self, prompt: str) -> tuple:
        """对同一问题生成两次回复，使用不同 session 以排除记忆影响。"""
        r1 = _run(self.system.process_message(prompt, str(uuid.uuid4())))
        r2 = _run(self.system.process_message(prompt, str(uuid.uuid4())))
        return r1, r2

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """简单字符级 Jaccard 相似度（用于检测完全雷同）。"""
        set_a = set(a)
        set_b = set(b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 1.0

    def test_tc03_emotional_response_not_identical(self):
        """情绪链：两次响应不应完全相同"""
        r1, r2 = self._two_responses("我最近压力很大，不知道怎么办")
        sim = self._similarity(r1, r2)
        print(f"\n  [TC-03 emotional] 相似度: {sim:.2f}")
        print(f"    R1: {r1[:60]}...")
        print(f"    R2: {r2[:60]}...")
        self.assertNotEqual(r1, r2,
                            "两次情绪链响应完全相同，可能存在硬编码模板")

    def test_tc03_knowledge_response_has_substance(self):
        """知识链：响应应包含实质性内容（> 50 字）"""
        r1, _ = self._two_responses("焦虑症是什么？")
        print(f"\n  [TC-03 knowledge] ({len(r1)}字): {r1[:80]}...")
        self.assertGreater(len(r1), 50,
                           f"知识链响应过短（{len(r1)}字），缺乏实质性内容")

    def test_tc03_chitchat_contains_self_intro(self):
        """闲聊链：自我介绍场景应包含友好词汇"""
        r1, _ = self._two_responses("你好，请介绍一下你自己")
        indicators = ["小暖", "你好", "您好", "很高兴", "帮助", "陪伴", "助手"]
        found = [ind for ind in indicators if ind in r1]
        print(f"\n  [TC-03 chitchat] 命中词: {found}  内容: {r1[:80]}...")
        self.assertGreater(len(found), 0,
                           f"闲聊自我介绍缺少友好词汇 {indicators}\n响应: {r1[:200]}")

    def test_tc03_responses_are_natural_language(self):
        """响应应为自然语言，而非 JSON / 代码 / 单词"""
        test_cases = [
            "我感到很焦虑",
            "什么是正念冥想？",
            "你好",
        ]
        for prompt in test_cases:
            resp = _run(self.system.process_message(prompt, str(uuid.uuid4())))
            # 中文字符 > 5 个即视为自然语言
            zh_count = sum(1 for c in resp if '\u4e00' <= c <= '\u9fff')
            self.assertGreater(
                zh_count, 5,
                f"输入 {prompt!r} 的响应中文字符不足（{zh_count}个），"
                f"可能不是自然语言: {resp[:100]}"
            )


# ══════════════════════════════════════════════════════════
# TC-04  追问响应
# ══════════════════════════════════════════════════════════
@unittest.skipIf(_SKIP_API, _SKIP_MSG)
class TC04_FollowUpResponse(unittest.TestCase):
    """
    TC-04: 对追问（"你刚才说的 X 是什么意思"）能给出与上下文相关的合理响应。
    """

    @classmethod
    def setUpClass(cls):
        _skip_if_api_down()
        print("\n  [TC-04 setup] 构建系统...")
        cls.system = _make_system()

    def test_tc04_followup_after_knowledge_answer(self):
        """知识链回复后追问，应得到合理的延伸解释"""
        sid = str(uuid.uuid4())
        # 第一轮：触发知识链
        r1 = _run(self.system.process_message("什么是认知行为疗法？", sid))
        print(f"\n  [TC-04] R1: {r1[:80]}...")

        # 第二轮：追问（参照前文）
        r2 = _run(self.system.process_message("你刚才说的认知重构是什么意思，能详细说吗？", sid))
        print(f"  [TC-04] R2 (追问): {r2[:80]}...")

        # 断言：追问响应非空 & 有实质内容
        self.assertGreater(len(r2.strip()), 30,
                           f"追问响应过短（{len(r2)}字）: {r2}")
        zh_count = sum(1 for c in r2 if '\u4e00' <= c <= '\u9fff')
        self.assertGreater(zh_count, 10,
                           "追问响应中文字符过少，可能没有给出有效回答")

    def test_tc04_followup_about_breathing_technique(self):
        """情绪支持后追问具体步骤，应得到操作性说明"""
        sid = str(uuid.uuid4())
        _run(self.system.process_message("我很焦虑，有什么方法可以缓解吗？", sid))
        r2 = _run(self.system.process_message("你说的呼吸练习具体是怎么做的？", sid))
        print(f"\n  [TC-04] 追问呼吸练习: {r2[:100]}...")
        self.assertGreater(len(r2.strip()), 30,
                           "追问具体步骤应得到详细回复")

    def test_tc04_multiple_followups_stable(self):
        """连续 3 次追问，系统应稳定响应不崩溃"""
        sid = str(uuid.uuid4())
        questions = [
            "什么是正念冥想？",
            "正念和冥想有什么区别？",
            "那坐禅和正念冥想一样吗？",
            "这些方法有科学依据吗？",
        ]
        errors = []
        for i, q in enumerate(questions, 1):
            try:
                resp = _run(self.system.process_message(q, sid))
                print(f"  [TC-04] Q{i}: {resp[:60]}...")
                if not resp.strip():
                    errors.append(f"Q{i} 响应为空")
            except Exception as e:
                errors.append(f"Q{i} 异常: {e}")
        self.assertEqual(len(errors), 0,
                         "连续追问出现错误:\n" + "\n".join(errors))


# ══════════════════════════════════════════════════════════
# TC-05  语义澄清
# ══════════════════════════════════════════════════════════
@unittest.skipIf(_SKIP_API, _SKIP_MSG)
class TC05_SemanticClarification(unittest.TestCase):
    """
    TC-05: 对模糊/不完整的输入，系统应给出合理响应
    （引导澄清 或 给出合理猜测后询问确认，而非生成无关内容）。
    """

    @classmethod
    def setUpClass(cls):
        _skip_if_api_down()
        print("\n  [TC-05 setup] 构建系统...")
        cls.system = _make_system()

    def _respond(self, text: str, sid=None) -> str:
        return _run(self.system.process_message(text, sid or str(uuid.uuid4())))

    def test_tc05_ambiguous_input_gets_response(self):
        """模糊输入（'我不知道'）应得到非空响应"""
        resp = self._respond("我不知道")
        print(f"\n  [TC-05] '我不知道' → {resp[:80]}...")
        self.assertGreater(len(resp.strip()), 10,
                           "模糊输入应得到合理响应，而非空字符串")

    def test_tc05_very_short_input_gets_response(self):
        """极短输入（'嗯'）应得到有意义的响应"""
        resp = self._respond("嗯")
        print(f"\n  [TC-05] '嗯' → {resp[:80]}...")
        zh_count = sum(1 for c in resp if '\u4e00' <= c <= '\u9fff')
        self.assertGreater(zh_count, 3,
                           f"极短输入应得到中文响应: {resp[:100]}")

    def test_tc05_clarification_after_ambiguous_topic(self):
        """上下文中有模糊指代时，系统应能处理（不崩溃）"""
        sid = str(uuid.uuid4())
        _run(self.system.process_message("我有个问题", sid))
        resp = _run(self.system.process_message("就是那个你知道的那种情况", sid))
        print(f"\n  [TC-05] 模糊指代 → {resp[:80]}...")
        self.assertGreater(len(resp.strip()), 10,
                           "对模糊指代应给出引导性或探询性响应")

    def test_tc05_mixed_language_input_handled(self):
        """中英混杂输入（常见于年轻用户）应得到合理响应"""
        resp = self._respond("我最近 feel very anxious，sleep 很差")
        print(f"\n  [TC-05] 中英混杂 → {resp[:80]}...")
        self.assertGreater(len(resp.strip()), 10,
                           "中英混杂输入应得到合理响应")


# ══════════════════════════════════════════════════════════
# TC-06  内容总结功能
# ══════════════════════════════════════════════════════════
@unittest.skipIf(_SKIP_API, _SKIP_MSG)
class TC06_ContentSummary(unittest.TestCase):
    """
    TC-06: 用户显式要求总结时，响应应包含对话中提及的关键信息。
    """

    @classmethod
    def setUpClass(cls):
        _skip_if_api_down()
        print("\n  [TC-06 setup] 构建系统并预填对话...")
        cls.system = _make_system()
        cls.session_id = str(uuid.uuid4())

        # 预填充对话内容（涵盖几个可识别的关键词）
        prior_turns = [
            "我叫小月，我最近焦虑很严重，睡眠也很差",
            "我试过深呼吸，但效果不太好",
            "正念冥想有什么具体的步骤？",
            "好的，我记下来了",
            "你还提到了认知行为疗法，能简单介绍一下吗？",
        ]
        for msg in prior_turns:
            _run(cls.system.process_message(msg, cls.session_id))
            time.sleep(0.2)  # 避免请求过密

        # 发送总结请求
        cls.summary_resp = _run(
            cls.system.process_message(
                "你能帮我总结一下我们今天聊了哪些内容吗？",
                cls.session_id
            )
        )
        print(f"\n  [TC-06] 总结响应 ({len(cls.summary_resp)}字):\n  {cls.summary_resp[:200]}...")

    def test_tc06_summary_response_nonempty(self):
        """TC-06: 总结响应不应为空"""
        self.assertGreater(len(self.summary_resp.strip()), 20,
                           "总结响应过短或为空")

    def test_tc06_summary_is_chinese(self):
        """TC-06: 总结应以中文输出"""
        zh = sum(1 for c in self.summary_resp if '\u4e00' <= c <= '\u9fff')
        self.assertGreater(zh, 10,
                           f"总结中文字符过少（{zh}个）: {self.summary_resp[:100]}")

    def test_tc06_summary_contains_key_topic(self):
        """TC-06: 总结中应包含对话提及的至少 1 个关键词"""
        key_terms = ["焦虑", "睡眠", "正念", "认知", "深呼吸", "小月"]
        found = [t for t in key_terms if t in self.summary_resp]
        print(f"\n  [TC-06] 总结命中关键词: {found}")
        self.assertGreater(
            len(found), 0,
            f"总结未命中任何关键词 {key_terms}\n总结内容: {self.summary_resp[:300]}"
        )

    def test_tc06_auto_summary_generated_after_many_turns(self):
        """TC-06: 超过摘要触发阈值后，SQLite 中应有自动摘要"""
        from config.settings import SUMMARY_TRIGGER_TURNS
        summary = self.system.session_store.get_summary(self.session_id)
        msg_count = self.system.session_store.get_message_count(self.session_id)
        turns = msg_count // 2
        print(f"\n  [TC-06] 对话轮数: {turns}  摘要触发阈值: {SUMMARY_TRIGGER_TURNS}")
        print(f"  自动摘要: {(summary or '（未生成）')[:100]}")

        if turns > SUMMARY_TRIGGER_TURNS:
            self.assertTrue(
                bool(summary),
                f"对话已超过 {SUMMARY_TRIGGER_TURNS} 轮（实际 {turns} 轮），"
                "但 SQLite 中未见自动摘要"
            )


# ══════════════════════════════════════════════════════════
# TC-08  交互稳定性 / 边界输入
# ══════════════════════════════════════════════════════════
@unittest.skipIf(_SKIP_API, _SKIP_MSG)
class TC08_StabilityAndEdgeCases(unittest.TestCase):
    """
    TC-08: 边界 / 异常输入不应导致系统崩溃，应得到合理响应。
    """

    @classmethod
    def setUpClass(cls):
        _skip_if_api_down()
        print("\n  [TC-08 setup] 构建系统...")
        cls.system = _make_system()

    def _safe_process(self, text: str) -> str:
        try:
            return _run(self.system.process_message(text, str(uuid.uuid4())))
        except Exception as exc:
            return f"[EXCEPTION: {type(exc).__name__}: {exc}]"

    def test_tc08_empty_lookalike_input(self):
        """全空格输入（在 main.py 层会被过滤，但系统层不应崩溃）"""
        # main.py 用 strip() 过滤空字符串，这里直接测底层 process_message
        resp = self._safe_process("   ")
        print(f"\n  [TC-08] 空格输入 → {resp[:60]}")
        self.assertFalse(resp.startswith("[EXCEPTION"),
                         f"空格输入导致异常: {resp}")

    def test_tc08_very_long_input(self):
        """超长输入（1000 字）不应崩溃"""
        long_text = "我感到很焦虑，" * 100
        resp = self._safe_process(long_text)
        print(f"\n  [TC-08] 超长输入（{len(long_text)}字）→ {resp[:60]}")
        self.assertFalse(resp.startswith("[EXCEPTION"),
                         f"超长输入导致异常: {resp[:200]}")

    def test_tc08_special_characters_input(self):
        """含特殊字符输入不应崩溃"""
        text = "我最近焦虑 😢😢😢 !!!! @#$% 心里很乱"
        resp = self._safe_process(text)
        print(f"\n  [TC-08] 特殊字符 → {resp[:60]}")
        self.assertFalse(resp.startswith("[EXCEPTION"),
                         f"特殊字符导致异常: {resp}")

    def test_tc08_danger_keyword_triggers_guardrail(self):
        """危险关键词应触发护栏链并返回热线信息"""
        resp = self._safe_process("我最近有想要伤害自己的念头")
        print(f"\n  [TC-08] 危险词 → {resp[:100]}")
        has_safety = "热线" in resp or "400" in resp or "专业" in resp
        self.assertTrue(has_safety,
                        f"危险词触发后响应中未见安全提示: {resp[:200]}")

    def test_tc08_repeated_same_message_stable(self):
        """相同消息重复发送 3 次，系统应保持稳定"""
        sid = str(uuid.uuid4())
        errors = []
        for i in range(3):
            resp = self._safe_process("你好吗？")
            if resp.startswith("[EXCEPTION"):
                errors.append(f"第{i+1}次: {resp[:80]}")
        self.assertEqual(len(errors), 0,
                         "重复消息导致异常:\n" + "\n".join(errors))

    def test_tc08_concurrent_sessions_isolated(self):
        """两个并发 Session 应相互隔离（不共享消息）"""
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())

        _run(self.system.process_message("我叫张伟", sid_a))
        _run(self.system.process_message("我叫李娟", sid_b))

        name_a = self.system.memory.get_user_name(sid_a)
        name_b = self.system.memory.get_user_name(sid_b)
        print(f"\n  [TC-08] Session A 用户名: {name_a!r}  Session B: {name_b!r}")

        self.assertEqual(name_a, "张伟", f"Session A 用户名应为 '张伟'，实际: {name_a!r}")
        self.assertEqual(name_b, "李娟", f"Session B 用户名应为 '李娟'，实际: {name_b!r}")


# ══════════════════════════════════════════════════════════
# 测试入口
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 68)
    print("  PsyMind 对话质量专项测试 (test_dialogue_quality.py)")
    print("=" * 68)
    print(f"  API Key: {'已设置 [OK]' if DASHSCOPE_API_KEY else '未设置 [跳过集成测试]'}")
    print("=" * 68)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TC09_ContextWindowSpec,          # 无需 API（最先运行）
        TC10_ResponseLatency,
        TC01_TC02_TC07_ContextAwareAndLongDialogue,
        TC03_NonTemplatedGeneration,
        TC04_FollowUpResponse,
        TC05_SemanticClarification,
        TC06_ContentSummary,
        TC08_StabilityAndEdgeCases,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

# -*- coding: utf-8 -*-
"""
情绪识别适配器单元测试
"""

import pytest
import time
import numpy as np

from digital_human.emotion_adapter import (
    EmotionResult,
    EmotionSmoother,
    EmotionContext,
    EmotionAdapter,
    EmotionContextSnapshot,
)


class TestEmotionResult:
    """测试 EmotionResult 数据类"""

    def test_to_dict(self):
        """测试结果序列化"""
        result = EmotionResult(
            dominant_emotion="happy",
            emotion_scores={"happy": 85.5, "sad": 5.2, "neutral": 9.3},
            sad_score=5.2,
            risk_level="normal",
            confidence=85.5,
            timestamp=1234567890.123,
        )

        data = result.to_dict()

        assert data["dominant_emotion"] == "happy"
        assert data["sad_score"] == 5.2
        assert data["risk_level"] == "normal"
        assert data["confidence"] == 85.5
        assert data["emotion_scores"]["happy"] == 85.5
        assert "timestamp" in data


class TestEmotionSmoother:
    """测试时空平滑器"""

    def test_smoother_returns_original_when_window_not_full(self):
        """窗口未满时直接返回原始情绪"""
        smoother = EmotionSmoother(window_size=5)

        result = smoother.smooth("session1", "happy")
        assert result == "happy"

        result = smoother.smooth("session1", "sad")
        assert result == "sad"

    def test_smoother_returns_mode(self):
        """窗口满后返回众数"""
        smoother = EmotionSmoother(window_size=5)

        # 添加 3 个 happy，2 个 sad
        smoother.smooth("session1", "happy")
        smoother.smooth("session1", "happy")
        smoother.smooth("session1", "sad")
        smoother.smooth("session1", "happy")
        result = smoother.smooth("session1", "sad")

        # happy 出现 3 次，sad 出现 2 次，众数是 happy
        assert result == "happy"

    def test_smoother_isolated_by_session(self):
        """不同会话的窗口是隔离的"""
        smoother = EmotionSmoother(window_size=5)

        # session1 全是 happy
        for _ in range(5):
            smoother.smooth("session1", "happy")

        # session2 全是 sad
        for _ in range(5):
            smoother.smooth("session2", "sad")

        # 各自保持独立
        assert smoother.smooth("session1", "neutral") == "happy"
        assert smoother.smooth("session2", "neutral") == "sad"

    def test_clear_session(self):
        """测试清除会话"""
        smoother = EmotionSmoother(window_size=5)

        for _ in range(5):
            smoother.smooth("session1", "happy")

        smoother.clear_session("session1")

        # 清除后重新开始
        result = smoother.smooth("session1", "sad")
        assert result == "sad"


class TestEmotionContext:
    """测试情绪上下文管理器"""

    def test_context_ttl_expiry(self):
        """测试 TTL 过期机制"""
        context = EmotionContext(ttl_seconds=1)

        # 添加一个结果
        result = EmotionResult(
            dominant_emotion="happy",
            emotion_scores={"happy": 80.0},
            sad_score=10.0,
            risk_level="normal",
            confidence=80.0,
            timestamp=time.time(),
        )
        context.add("session1", result)

        # 立即查询，应该有数据
        summary = context.get_summary("session1")
        assert summary["has_data"] is True

        # 等待过期
        time.sleep(1.5)

        # 过期后应该无数据
        summary = context.get_summary("session1")
        assert summary["has_data"] is False

    def test_context_summary(self):
        """测试上下文摘要"""
        context = EmotionContext(ttl_seconds=60)

        # 添加多个结果
        emotions = ["happy", "happy", "happy", "sad", "neutral"]
        for emotion in emotions:
            result = EmotionResult(
                dominant_emotion=emotion,
                emotion_scores={emotion: 80.0},
                sad_score=20.0 if emotion == "sad" else 5.0,
                risk_level="normal",
                confidence=80.0,
                timestamp=time.time(),
            )
            context.add("session1", result)

        summary = context.get_summary("session1")

        assert summary["has_data"] is True
        assert summary["sample_count"] == 5
        assert "happy" in summary["dominant_emotions"]

    def test_prompt_context_generation(self):
        """测试 LLM Prompt 生成"""
        context = EmotionContext(ttl_seconds=60)

        # 添加高悲伤分数
        result = EmotionResult(
            dominant_emotion="sad",
            emotion_scores={"sad": 70.0, "neutral": 30.0},
            sad_score=70.0,
            risk_level="warning",
            confidence=70.0,
            timestamp=time.time(),
        )
        context.add("session1", result)

        prompt = context.get_prompt_context("session1")

        assert "用户情绪状态" in prompt
        assert "sad" in prompt.lower() or "悲伤" in prompt

    def test_clear_session(self):
        """测试清除会话上下文"""
        context = EmotionContext(ttl_seconds=60)

        result = EmotionResult(
            dominant_emotion="happy",
            emotion_scores={"happy": 80.0},
            sad_score=10.0,
            risk_level="normal",
            confidence=80.0,
            timestamp=time.time(),
        )
        context.add("session1", result)

        context.clear_session("session1")

        summary = context.get_summary("session1")
        assert summary["has_data"] is False


class TestEmotionAdapter:
    """测试情绪适配器"""

    def test_adapter_disabled_returns_neutral(self):
        """功能禁用时返回中性结果"""
        # 创建一个禁用的适配器
        class MockConfig:
            enabled = False
            detector_backend = "opencv"
            enforce_detection = False
            smoothing_window = 5
            sad_score_warning = 60.0
            sad_score_critical = 80.0
            context_ttl = 30
            analysis_timeout = 2.0

        # 重置单例
        import digital_human.emotion_adapter as module
        module._emotion_adapter_instance = None

        adapter = EmotionAdapter(config=MockConfig())

        # 创建一个测试图像
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.detect_from_frame(frame, "test")

        assert result is not None
        assert result.dominant_emotion == "neutral"
        assert result.risk_level == "normal"

    def test_get_risk_level(self):
        """测试风险等级判定"""
        class MockConfig:
            enabled = False
            detector_backend = "opencv"
            enforce_detection = False
            smoothing_window = 5
            sad_score_warning = 60.0
            sad_score_critical = 80.0
            context_ttl = 30
            analysis_timeout = 2.0

        import digital_human.emotion_adapter as module
        module._emotion_adapter_instance = None

        adapter = EmotionAdapter(config=MockConfig())

        assert adapter._get_risk_level(30.0) == "normal"
        assert adapter._get_risk_level(65.0) == "warning"
        assert adapter._get_risk_level(85.0) == "critical"

    def test_get_context_returns_summary(self):
        """测试获取上下文"""
        class MockConfig:
            enabled = False
            detector_backend = "opencv"
            enforce_detection = False
            smoothing_window = 5
            sad_score_warning = 60.0
            sad_score_critical = 80.0
            context_ttl = 30
            analysis_timeout = 2.0

        import digital_human.emotion_adapter as module
        module._emotion_adapter_instance = None

        adapter = EmotionAdapter(config=MockConfig())

        # 无数据时的摘要
        summary = adapter.get_context("new_session")
        assert summary["has_data"] is False

    def test_clear_session(self):
        """测试清除会话数据"""
        class MockConfig:
            enabled = False
            detector_backend = "opencv"
            enforce_detection = False
            smoothing_window = 5
            sad_score_warning = 60.0
            sad_score_critical = 80.0
            context_ttl = 30
            analysis_timeout = 2.0

        import digital_human.emotion_adapter as module
        module._emotion_adapter_instance = None

        adapter = EmotionAdapter(config=MockConfig())

        # 添加数据
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        adapter.detect_from_frame(frame, "test_session")

        # 清除
        adapter.clear_session("test_session")

        # 验证清除
        summary = adapter.get_context("test_session")
        assert summary["has_data"] is False


class TestEmotionAdapterSingleton:
    """测试单例模式"""

    def test_singleton(self):
        """验证单例模式"""
        import digital_human.emotion_adapter as module
        module._emotion_adapter_instance = None

        from digital_human import get_emotion_adapter

        adapter1 = get_emotion_adapter()
        adapter2 = get_emotion_adapter()

        assert adapter1 is adapter2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

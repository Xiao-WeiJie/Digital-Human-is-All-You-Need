# -*- coding: utf-8 -*-
"""
情绪融合器测试
"""

import pytest
from digital_human.emotion_fusion import (
    EmotionFusion,
    FusedEmotionContext,
    get_emotion_fusion,
)


class TestEmotionFusion:
    """情绪融合器测试类"""

    @pytest.fixture
    def fusion(self):
        """获取融合器实例"""
        return get_emotion_fusion()

    def test_fuse_text_emotion_only(self, fusion):
        """测试仅文本情绪融合"""
        result = fusion.fuse("我很难过", "test_session")
        assert result.primary_emotion == "sad"
        assert result.text_emotion == "sad"
        assert result.facial_emotion is None
        assert "悲伤" in result.prompt_context

    def test_fuse_with_facial_emotion_consistent(self, fusion):
        """测试文本和面部情绪一致时的融合"""
        # 模拟面部情绪上下文（悲伤）
        facial_context = {
            "has_data": True,
            "dominant_emotion": "sad",
            "emotion_scores": {"sad": 80.0, "neutral": 20.0},
            "latest_risk_level": "normal",
        }
        result = fusion.fuse("我很难过", "test_session", facial_context)
        # 文本情绪为主
        assert result.primary_emotion == "sad"
        # 一致时置信度应该更高
        assert result.confidence > 0.5

    def test_fuse_with_facial_emotion_inconsistent(self, fusion):
        """测��文本和面部情绪不一致时的融合"""
        # 文本是悲伤，面部是开心
        facial_context = {
            "has_data": True,
            "dominant_emotion": "happy",
            "emotion_scores": {"happy": 70.0, "neutral": 30.0},
            "latest_risk_level": "normal",
        }
        result = fusion.fuse("我很难过", "test_session", facial_context)
        # 文本情绪为主
        assert result.primary_emotion == "sad"
        assert result.facial_emotion == "happy"

    def test_fuse_crisis_detection(self, fusion):
        """测试危机检测"""
        result = fusion.fuse("我不想活了", "test_session")
        assert result.risk_level == "critical"
        assert result.needs_support is True
        assert "危机" in result.prompt_context
        # 危机情况应该有高强度
        assert result.intensity >= 0.5

    def test_fuse_needs_support(self, fusion):
        """测试需要关怀检测"""
        result = fusion.fuse("我很孤独，没人理解我", "test_session")
        assert result.needs_support is True

    def test_fuse_neutral_text(self, fusion):
        """测试中性文本融合"""
        result = fusion.fuse("你好", "test_session")
        # 可能是 neutral 或其他情绪
        assert result.primary_emotion is not None

    def test_fuse_facial_risk_warning(self, fusion):
        """测试面部情绪风险等级传递"""
        facial_context = {
            "has_data": True,
            "dominant_emotion": "sad",
            "emotion_scores": {"sad": 90.0},
            "latest_risk_level": "warning",
        }
        result = fusion.fuse("我有点难过", "test_session", facial_context)
        assert result.risk_level in ("warning", "critical")

    def test_get_text_only_context(self, fusion):
        """测试便捷方法"""
        context = fusion.get_text_only_context("我很难过")
        assert "[情绪感知上下文]" in context
        assert "悲伤" in context

    def test_positive_emotion_no_warning(self, fusion):
        """测试正面情绪不触发 warning"""
        result = fusion.fuse("今天真开心，非常高兴", "test_session")
        assert result.primary_emotion == "happy"
        # 正面情绪即使高强度也不应该触发 warning
        assert result.risk_level == "normal"

    def test_singleton(self):
        """测试单例模式"""
        fusion1 = get_emotion_fusion()
        fusion2 = get_emotion_fusion()
        assert fusion1 is fusion2

    def test_weights(self, fusion):
        """测试权重配置"""
        assert fusion.text_weight == 0.7
        assert fusion.facial_weight == 0.3
        assert fusion.text_weight + fusion.facial_weight == 1.0


class TestFusedEmotionContext:
    """融合情绪上下文测试"""

    def test_to_dict(self):
        """测试转换为字典"""
        context = FusedEmotionContext(
            primary_emotion="sad",
            confidence=0.8,
            intensity=0.7,
            text_emotion="sad",
            facial_emotion="sad",
            emotion_scores={"sad": 75.0, "neutral": 25.0},
            risk_level="normal",
            needs_support=True,
            prompt_context="[情绪感知上下文]\n用户当前情绪状态: 悲伤",
        )
        d = context.to_dict()
        assert d["primary_emotion"] == "sad"
        assert d["confidence"] == 0.8
        assert d["risk_level"] == "normal"
        assert d["needs_support"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

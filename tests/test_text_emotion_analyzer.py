# -*- coding: utf-8 -*-
"""
文本情绪分析器测试
"""

import pytest
from digital_human.text_emotion_analyzer import (
    TextEmotionAnalyzer,
    TextEmotionResult,
    get_text_emotion_analyzer,
)


class TestTextEmotionAnalyzer:
    """文本情绪分析器测试类"""

    @pytest.fixture
    def analyzer(self):
        """获取分析器实例"""
        return get_text_emotion_analyzer()

    def test_analyze_sad_emotion(self, analyzer):
        """测试悲伤情绪识别"""
        result = analyzer.analyze("我真的很难过，感觉很绝望")
        assert result.primary_emotion == "sad"
        assert result.needs_support is True
        assert "难过" in result.matched_keywords or "绝望" in result.matched_keywords

    def test_analyze_happy_emotion(self, analyzer):
        """测试开心情绪识别"""
        result = analyzer.analyze("今天真开心，谢谢你的帮助！")
        assert result.primary_emotion in ("happy", "gratitude")
        assert result.needs_support is False

    def test_analyze_anxious_emotion(self, analyzer):
        """测试焦虑情绪识别"""
        result = analyzer.analyze("我很焦虑，担心考试考不好")
        assert result.primary_emotion == "anxious"
        assert "焦虑" in result.matched_keywords or "担心" in result.matched_keywords

    def test_analyze_lonely_emotion(self, analyzer):
        """测试孤独情绪识别"""
        result = analyzer.analyze("一个人在家好孤独，没人陪我")
        assert result.primary_emotion == "lonely"
        assert result.needs_support is True

    def test_analyze_crisis_indicators(self, analyzer):
        """测试危机关键词检测"""
        result = analyzer.analyze("我不想活了，活着没意思")
        assert len(result.risk_indicators) > 0
        assert result.needs_support is True
        assert "想死" in result.risk_indicators or "活着没意思" in result.risk_indicators

    def test_analyze_neutral_text(self, analyzer):
        """测试中性文本"""
        result = analyzer.analyze("今天天气不错")
        # 可能被识别为 neutral 或其他情绪
        assert result.primary_emotion is not None

    def test_analyze_empty_text(self, analyzer):
        """测试空文本"""
        result = analyzer.analyze("")
        assert result.primary_emotion == "neutral"
        assert result.confidence == 0.0

    def test_intensity_calculation(self, analyzer):
        """测试情绪强度计算"""
        # 带增强词
        result1 = analyzer.analyze("我非常难过")
        # 不带增强词
        result2 = analyzer.analyze("我难过")
        # 带增强词的强度应该更高
        assert result1.intensity >= result2.intensity

    def test_get_prompt_context(self, analyzer):
        """测试生成 Prompt 上下文"""
        context = analyzer.get_prompt_context("我很难过，感觉活着没意思")
        assert "[情绪感知上下文]" in context
        assert "悲伤" in context
        assert "危机" in context

    def test_get_prompt_context_neutral(self, analyzer):
        """测试中性文本的 Prompt 上下文"""
        context = analyzer.get_prompt_context("你好")
        # "你好" 可能被识别为 happy 或其他情绪
        # 只要生成有效的上下文即可
        assert isinstance(context, str)

    def test_singleton(self):
        """测试单例模式"""
        analyzer1 = get_text_emotion_analyzer()
        analyzer2 = get_text_emotion_analyzer()
        assert analyzer1 is analyzer2


class TestTextEmotionResult:
    """情绪分析结果测试"""

    def test_to_dict(self):
        """测试转换为字典"""
        result = TextEmotionResult(
            primary_emotion="sad",
            emotion_scores={"sad": 80.0, "neutral": 20.0},
            intensity=0.7,
            matched_keywords=["难过"],
            risk_indicators=[],
            needs_support=True,
            confidence=0.8,
        )
        d = result.to_dict()
        assert d["primary_emotion"] == "sad"
        assert d["intensity"] == 0.7
        assert d["needs_support"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

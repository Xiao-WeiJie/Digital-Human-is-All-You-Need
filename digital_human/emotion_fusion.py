# -*- coding: utf-8 -*-
"""
情绪融合器

融合文本情绪（主）和面部情绪（辅），生成统一的情绪上下文供 LLM 使用。

融合策略：
- 文本情绪权重: 70% - 主要参考，反映用户主动表达
- 面部情绪权重: 30% - 辅助验证，增强判断置信度
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class FusedEmotionContext:
    """融合后的情绪上下文"""
    primary_emotion: str              # 主要情绪
    confidence: float                 # 置信度 (0-1)
    intensity: float                  # 强度 (0-1)
    text_emotion: str                 # 文本情绪
    facial_emotion: Optional[str]     # 面部情绪
    emotion_scores: Dict[str, float]  # 融合后的情绪分数
    risk_level: str                   # 风险等级: normal/warning/critical
    needs_support: bool               # 是否需要关怀
    prompt_context: str               # LLM Prompt 上下文

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "primary_emotion": self.primary_emotion,
            "confidence": self.confidence,
            "intensity": self.intensity,
            "text_emotion": self.text_emotion,
            "facial_emotion": self.facial_emotion,
            "emotion_scores": self.emotion_scores,
            "risk_level": self.risk_level,
            "needs_support": self.needs_support,
            "prompt_context": self.prompt_context,
        }


# 文本情绪到面部情绪的映射
# DeepFace 输出的情绪: angry, disgust, fear, happy, sad, surprise, neutral
_EMOTION_MAPPING = {
    "sad": "sad",
    "happy": "happy",
    "anxious": "fear",
    "angry": "angry",
    "lonely": "sad",      # 孤独映射到悲伤
    "confused": "neutral",
    "fear": "fear",
    "hope": "happy",      # 希望映射到开心
    "gratitude": "happy", # 感激映射到开心
    "neutral": "neutral",
}

# 反向映射：面部情绪到文本情绪
_FACIAL_TO_TEXT_MAPPING = {
    "sad": "sad",
    "happy": "happy",
    "fear": "anxious",
    "angry": "angry",
    "disgust": "angry",
    "surprise": "confused",
    "neutral": "neutral",
}

# 情绪中文名
_EMOTION_CN = {
    "sad": "悲伤",
    "happy": "开心",
    "anxious": "焦虑",
    "angry": "愤怒",
    "lonely": "孤独",
    "confused": "困惑",
    "fear": "恐惧",
    "hope": "希望",
    "gratitude": "感激",
    "neutral": "平静",
}


class EmotionFusion:
    """
    情绪融合器

    融合文本情绪和面部情绪，生成统一的情绪上下文。
    采用 70/30 权重分配，文本情绪为主。
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Any] = None):
        # 避免重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = False

        # 获取配置
        if config is None:
            try:
                from .config import get_default_config
                config = get_default_config().emotion_fusion
            except Exception:
                config = None

        # 默认权重
        self.text_weight = 0.7
        self.facial_weight = 0.3
        self.enabled = True

        # 从配置读取
        if config:
            self.text_weight = getattr(config, "text_weight", 0.7)
            self.facial_weight = getattr(config, "facial_weight", 0.3)
            self.enabled = getattr(config, "enabled", True)

        self._initialized = True
        logger.info(f"EmotionFusion initialized: text_weight={self.text_weight}, facial_weight={self.facial_weight}")

    def fuse(
        self,
        text: str,
        session_id: str = "default",
        facial_emotion_context: Optional[Dict] = None,
    ) -> FusedEmotionContext:
        """
        融合文本情绪和面部情绪

        Args:
            text: 用户输入文本
            session_id: 会话 ID
            facial_emotion_context: 面部情绪上下文（可选，从 emotion_adapter 获取）

        Returns:
            FusedEmotionContext: 融合后的情绪上下文
        """
        # 1. 分析文本情绪
        from .text_emotion_analyzer import get_text_emotion_analyzer
        text_analyzer = get_text_emotion_analyzer()
        text_result = text_analyzer.analyze(text)

        text_emotion = text_result.primary_emotion
        text_scores = text_result.emotion_scores.copy()
        intensity = text_result.intensity
        needs_support = text_result.needs_support
        risk_indicators = text_result.risk_indicators

        # 2. 获取面部情绪（如果有）
        facial_emotion = None
        facial_scores = {}

        if facial_emotion_context and facial_emotion_context.get("has_data"):
            facial_emotion = facial_emotion_context.get("dominant_emotion")
            # 将面部情绪分数映射到文本情绪分数
            raw_scores = facial_emotion_context.get("emotion_scores", {})
            for facial_key, score in raw_scores.items():
                text_key = _FACIAL_TO_TEXT_MAPPING.get(facial_key, facial_key)
                if text_key not in facial_scores:
                    facial_scores[text_key] = 0
                facial_scores[text_key] = max(facial_scores[text_key], score)

        # 3. 融合情绪分数
        fused_scores = self._fuse_scores(text_scores, facial_scores)

        # 4. 确定主要情绪
        primary_emotion = text_emotion  # 文本情绪为主

        # 5. 计算置信度
        confidence = self._calculate_confidence(
            text_emotion, facial_emotion, text_result.confidence
        )

        # 6. 确定风险等级
        risk_level = self._determine_risk_level(
            risk_indicators, facial_emotion_context, intensity, primary_emotion
        )

        # 7. 更新需要关怀标志
        if risk_level in ("warning", "critical"):
            needs_support = True

        # 8. 危机情况确保高强度
        if risk_indicators and intensity < 0.5:
            intensity = 0.8  # 危机情况默认高强度

        # 9. 生成 Prompt 上下文
        prompt_context = self._generate_prompt_context(
            primary_emotion=primary_emotion,
            confidence=confidence,
            intensity=intensity,
            text_emotion=text_emotion,
            facial_emotion=facial_emotion,
            risk_level=risk_level,
            needs_support=needs_support,
            text_result=text_result,
        )

        return FusedEmotionContext(
            primary_emotion=primary_emotion,
            confidence=confidence,
            intensity=intensity,
            text_emotion=text_emotion,
            facial_emotion=facial_emotion,
            emotion_scores=fused_scores,
            risk_level=risk_level,
            needs_support=needs_support,
            prompt_context=prompt_context,
        )

    def _fuse_scores(
        self,
        text_scores: Dict[str, float],
        facial_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        融合文本和面部情绪分数

        Args:
            text_scores: 文本情绪分数
            facial_scores: 面部情绪分数

        Returns:
            Dict[str, float]: 融合后的分数
        """
        all_emotions = set(text_scores.keys()) | set(facial_scores.keys())
        fused = {}

        for emotion in all_emotions:
            t_score = text_scores.get(emotion, 0) * self.text_weight
            f_score = facial_scores.get(emotion, 0) * self.facial_weight
            fused[emotion] = round(t_score + f_score, 2)

        return fused

    def _calculate_confidence(
        self,
        text_emotion: str,
        facial_emotion: Optional[str],
        text_confidence: float,
    ) -> float:
        """
        计算置信度

        如果文本和面部情绪一致，置信度更高。
        """
        base_confidence = text_confidence

        if facial_emotion:
            # 映射面部情绪到文本情绪
            mapped_facial = _EMOTION_MAPPING.get(text_emotion, text_emotion)

            if facial_emotion == mapped_facial:
                # 一致，提高置信度
                return min(1.0, base_confidence + 0.2)
            else:
                # 不一致，以文本为准，置信度略低
                return max(0.5, base_confidence)

        return base_confidence

    def _determine_risk_level(
        self,
        risk_indicators: list,
        facial_context: Optional[Dict],
        intensity: float,
        primary_emotion: str = "neutral",
    ) -> str:
        """
        确定风险等级

        Returns:
            str: normal/warning/critical
        """
        # 危机关键词 -> critical
        if risk_indicators:
            return "critical"

        # 面部情绪风险 -> 检查
        if facial_context:
            facial_risk = facial_context.get("latest_risk_level", "normal")
            if facial_risk == "critical":
                return "critical"
            if facial_risk == "warning":
                return "warning"

        # 高强度负面情绪 -> warning（只有负面情绪才触发）
        negative_emotions = {"sad", "anxious", "angry", "lonely", "fear", "confused"}
        if intensity > 0.7 and primary_emotion in negative_emotions:
            return "warning"

        return "normal"

    def _generate_prompt_context(
        self,
        primary_emotion: str,
        confidence: float,
        intensity: float,
        text_emotion: str,
        facial_emotion: Optional[str],
        risk_level: str,
        needs_support: bool,
        text_result,
    ) -> str:
        """
        生成 LLM Prompt 上下文
        """
        if primary_emotion == "neutral" and not needs_support:
            return ""

        lines = ["[情绪感知上下文]"]

        # 主要情绪
        emotion_cn = _EMOTION_CN.get(primary_emotion, primary_emotion)
        lines.append(f"用户当前情绪状态: {emotion_cn}")

        # 情绪强度
        if intensity > 0.7:
            lines.append("情绪强度: 强烈，请给予充分关注和共情")
        elif intensity > 0.4:
            lines.append("情绪强度: 较强，请给予充分关注")
        elif intensity > 0.1:
            lines.append("情绪强度: 轻微，适当关注即可")

        # 匹配到的关键词
        if text_result.matched_keywords:
            keywords_str = "、".join(text_result.matched_keywords[:5])
            lines.append(f"情绪相关表达: {keywords_str}")

        # 悲伤分数
        sad_score = text_result.emotion_scores.get("sad", 0)
        if sad_score > 0:
            lines.append(f"悲伤指数: {sad_score:.0f}%")

        # 文本和面部情绪对比
        if facial_emotion and text_emotion != "neutral":
            facial_cn = _EMOTION_CN.get(
                _FACIAL_TO_TEXT_MAPPING.get(facial_emotion, facial_emotion),
                facial_emotion
            )
            text_cn = _EMOTION_CN.get(text_emotion, text_emotion)
            if facial_emotion == _EMOTION_MAPPING.get(text_emotion):
                lines.append(f"面部表情与言语情绪一致（{text_cn}），可信度高")
            else:
                lines.append(f"言语表达: {text_cn}；面部表情: {facial_cn}")

        # 置信度
        if confidence > 0.8:
            lines.append(f"情绪判断置信度: 高")
        elif confidence > 0.5:
            lines.append(f"情绪判断置信度: 中")

        # 风险等级
        if risk_level == "critical":
            lines.append("")
            lines.append("⚠️ 重要: 用户可能处于情绪危机中，请以最高优先级给予关怀")
            if text_result.risk_indicators:
                crisis_keywords = "、".join(text_result.risk_indicators[:3])
                lines.append(f"危机信号: {crisis_keywords}")
        elif risk_level == "warning":
            lines.append("提示: 用户情绪需要关注，请温柔回应")
        elif needs_support:
            lines.append("提示: 用户可能需要情感支持，请温柔回应")

        return "\n".join(lines) + "\n"

    def get_text_only_context(self, text: str) -> str:
        """
        获取仅基于文本的情绪上下文（便捷方法）

        Args:
            text: 用户输入文本

        Returns:
            str: LLM Prompt 上下文
        """
        result = self.fuse(text, session_id="default", facial_emotion_context=None)
        return result.prompt_context


# 单例访问函数
_emotion_fusion_instance: Optional[EmotionFusion] = None
_emotion_fusion_lock = Lock()


def get_emotion_fusion(config: Optional[Any] = None) -> EmotionFusion:
    """获取情绪融合器单例"""
    global _emotion_fusion_instance

    if _emotion_fusion_instance is None:
        with _emotion_fusion_lock:
            if _emotion_fusion_instance is None:
                _emotion_fusion_instance = EmotionFusion(config)

    return _emotion_fusion_instance

# -*- coding: utf-8 -*-
"""
文本情绪分析器

基于关键词匹配分析用户文本情绪，用于增强 LLM 对话上下文。
支持多种情绪类型识别、危机关键词检测、情绪强度评估。
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class TextEmotion:
    """文本情绪枚举"""
    SAD = "sad"           # 悲伤
    HAPPY = "happy"       # 开心
    ANXIOUS = "anxious"   # 焦虑
    ANGRY = "angry"       # 愤怒
    LONELY = "lonely"     # 孤独
    CONFUSED = "confused" # 困惑
    FEAR = "fear"         # 恐惧
    HOPE = "hope"         # 希望
    GRATITUDE = "gratitude"  # 感激
    NEUTRAL = "neutral"   # 中性


@dataclass
class TextEmotionResult:
    """文本情绪分析结果"""
    primary_emotion: str              # 主要情绪
    emotion_scores: Dict[str, float]  # 各情绪分数
    intensity: float                  # 情绪强度 (0-1)
    matched_keywords: List[str]       # 匹配到的关键词
    risk_indicators: List[str]        # 危机指标
    needs_support: bool               # 是否需要关怀
    confidence: float                 # 置信度

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "primary_emotion": self.primary_emotion,
            "emotion_scores": self.emotion_scores,
            "intensity": self.intensity,
            "matched_keywords": self.matched_keywords,
            "risk_indicators": self.risk_indicators,
            "needs_support": self.needs_support,
            "confidence": self.confidence,
        }


# ============================================================================
# 情绪词典定义
# ============================================================================

# 基础情绪关键词词典
_EMOTION_KEYWORDS: Dict[str, Set[str]] = {
    "sad": {
        "难过", "悲伤", "伤心", "痛苦", "绝望", "想哭", "哭泣", "流泪",
        "失落", "沮丧", "消沉", "郁闷", "心碎", "崩溃", "灰心", "失望",
        "压抑", "憋屈", "无奈", "无助", "空虚", "绝望", "活不下去",
        "活着没意思", "没意思", "没劲", "没希望", "不想活", "想死",
        "没意义", "痛苦不堪", "煎熬", "折磨", "煎熬", "难以承受",
    },
    "happy": {
        "开心", "高兴", "快乐", "幸福", "谢谢", "感谢", "感激",
        "喜欢", "爱", "美好", "棒", "好", "太好了", "真好",
        "欣慰", "满足", "舒适", "轻松", "愉快", "兴奋", "期待",
        "值得", "有意义", "有希望", "有动力", "有信心",
    },
    "anxious": {
        "焦虑", "担心", "害怕", "紧张", "压力", "焦虑", "不安",
        "恐慌", "烦躁", "急躁", "焦躁", "忧虑", "忐忑", "心慌",
        "心悸", "失眠", "睡不着", "噩梦", "惊恐", "惶恐", "惧怕",
        "不知所措", "六神无主", "手足无措", "坐立不安",
    },
    "angry": {
        "生气", "愤怒", "火大", "烦", "讨厌", "恨", "厌恶",
        "恼火", "气愤", "暴怒", "恼怒", "憋气", "窝火", "不爽",
        "受够了", "忍不了", "无法忍受", "不可理喻", "气死",
    },
    "lonely": {
        "孤独", "寂寞", "没人", "一个人", "孤单", "孤僻", "落寞",
        "无人", "没人理解", "没人关心", "被遗忘", "被抛弃",
        "被冷落", "被忽视", "没有朋友", "没人陪", "独来独往",
    },
    "confused": {
        "困惑", "迷茫", "不知道", "不明白", "不懂", "疑惑",
        "不解", "纳闷", "纠结", "矛盾", "左右为难", "不知道怎么办",
        "不知所措", "搞不懂", "想不通", "理不清", "混乱",
    },
    "fear": {
        "害怕", "恐惧", "怕", "惧怕", "畏惧", "胆怯", "心虚",
        "不敢", "提心吊胆", "惶惶不安", "惊恐", "战栗", "颤抖",
        "做噩梦", "恐怖", "吓人", "可怕", "令人窒息",
    },
    "hope": {
        "希望", "期待", "盼望", "憧憬", "向往", "想要", "渴望",
        "努力", "加油", "坚持", "相信", "相信未来", "会好的",
        "有希望", "有信心", "有动力", "有方向", "有目标",
    },
    "gratitude": {
        "谢谢", "感谢", "感激", "感恩", "谢谢你", "感谢你",
        "多谢", "有劳", "辛苦", "麻烦", "费心", "承蒙",
        "谢谢帮助", "感谢帮助", "谢谢理解", "感谢理解",
    },
}

# 危机关键词（高优先级）
_RISK_KEYWORDS: Set[str] = {
    "想死", "自杀", "不想活", "活着没意思", "活着没意义",
    "结束生命", "自我了断", "轻生", "寻短见", "一了百了",
    "生不如死", "死了算了", "不如死了", "想结束", "想离开",
    "跳楼", "割腕", "服药", "过量", "上吊", "投河",
    "不想存在", "消失", "解脱", "永别", "告别", "遗书",
    "最后", "临终", "临走", "走之前", "最后的话",
}

# 需要关怀的关键词
_SUPPORT_KEYWORDS: Set[str] = {
    "难过", "痛苦", "绝望", "崩溃", "无助", "孤独", "寂寞",
    "没人理解", "没人关心", "被抛弃", "被遗忘", "焦虑", "恐惧",
    "害怕", "睡不着", "失眠", "做噩梦", "压力大", "承受不了",
    "受不了", "坚持不住", "撑不住", "熬不下去", "想放弃",
}

# 情绪增强词（提高强度）
_INTENSIFIER_WORDS: Set[str] = {
    "非常", "特别", "极其", "十分", "相当", "格外", "异常",
    "太", "好", "真", "真的", "实在", "确实", "的确",
    "特别", "尤其", "超级", "无比", "极为", "万分",
}

# 否定词（可能反转情绪）
_NEGATION_WORDS: Set[str] = {
    "不", "没", "无", "别", "莫", "非", "未", "没有", "不是",
    "不再", "不会", "不要", "不想", "不太", "不怎么",
}


class TextEmotionAnalyzer:
    """
    文本情绪分析器

    基于关键词匹配分析用户文本情绪，支持：
    - 多种情绪类型识别
    - 危机关键词检测
    - 情绪强度评估
    - LLM Prompt 上下文生成
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 避免重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = False
        self.emotion_keywords = _EMOTION_KEYWORDS
        self.risk_keywords = _RISK_KEYWORDS
        self.support_keywords = _SUPPORT_KEYWORDS
        self.intensifier_words = _INTENSIFIER_WORDS
        self.negation_words = _NEGATION_WORDS
        self._initialized = True

        logger.info(f"TextEmotionAnalyzer initialized with {len(self.emotion_keywords)} emotion categories")

    def analyze(self, text: str) -> TextEmotionResult:
        """
        分析文本情绪

        Args:
            text: 用户输入文本

        Returns:
            TextEmotionResult: 分析结果
        """
        if not text or not text.strip():
            return self._get_neutral_result()

        # 预处理文本
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text_lower)

        # 计算各情绪分数
        emotion_scores: Dict[str, float] = {}
        matched_keywords: List[str] = []
        total_matches = 0

        for emotion, keywords in self.emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower or keyword in text_clean:
                    score += 1.0
                    matched_keywords.append(keyword)
                    total_matches += 1

            emotion_scores[emotion] = score

        # 归一化分数
        if total_matches > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = round(emotion_scores[emotion] / total_matches * 100, 2)
        else:
            emotion_scores["neutral"] = 100.0

        # 确定主要情绪
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        if emotion_scores[primary_emotion] == 0:
            primary_emotion = "neutral"

        # 检测危机关键词
        risk_indicators = []
        for keyword in self.risk_keywords:
            if keyword in text_lower:
                risk_indicators.append(keyword)

        # 检测需要关怀
        needs_support = len(risk_indicators) > 0
        if not needs_support:
            for keyword in self.support_keywords:
                if keyword in text_lower:
                    needs_support = True
                    break

        # 计算情绪强度
        intensity = self._calculate_intensity(text_lower, matched_keywords)

        # 计算置信度
        confidence = min(1.0, total_matches / 3.0) if total_matches > 0 else 0.0

        return TextEmotionResult(
            primary_emotion=primary_emotion,
            emotion_scores=emotion_scores,
            intensity=intensity,
            matched_keywords=list(set(matched_keywords)),
            risk_indicators=risk_indicators,
            needs_support=needs_support,
            confidence=confidence,
        )

    def _calculate_intensity(self, text: str, matched_keywords: List[str]) -> float:
        """
        计算情绪强度

        Args:
            text: 原始文本
            matched_keywords: 匹配到的关键词

        Returns:
            float: 强度值 (0-1)
        """
        base_intensity = min(1.0, len(matched_keywords) / 5.0)

        # 检测增强词
        intensifier_count = sum(1 for word in self.intensifier_words if word in text)
        intensifier_boost = min(0.3, intensifier_count * 0.1)

        # 检测否定词（可能降低强度）
        negation_count = sum(1 for word in self.negation_words if word in text)
        negation_penalty = min(0.3, negation_count * 0.1)

        # 感叹号增强
        exclamation_boost = min(0.2, text.count('!') + text.count('！') * 0.1)

        intensity = base_intensity + intensifier_boost - negation_penalty + exclamation_boost
        return max(0.0, min(1.0, intensity))

    def _get_neutral_result(self) -> TextEmotionResult:
        """获取中性结果"""
        return TextEmotionResult(
            primary_emotion="neutral",
            emotion_scores={"neutral": 100.0},
            intensity=0.0,
            matched_keywords=[],
            risk_indicators=[],
            needs_support=False,
            confidence=0.0,
        )

    def get_prompt_context(self, text: str) -> str:
        """
        生成 LLM Prompt 上下文

        Args:
            text: 用户输入文本

        Returns:
            str: 供 LLM 使用的情绪上下文提示文本
        """
        result = self.analyze(text)

        if result.primary_emotion == "neutral" and not result.needs_support:
            return ""

        lines = ["[情绪感知上下文]"]

        # 主要情绪
        emotion_map = {
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
        emotion_cn = emotion_map.get(result.primary_emotion, result.primary_emotion)
        lines.append(f"用户当前情绪状态: {emotion_cn}")

        # 情绪强度
        if result.intensity > 0.7:
            lines.append("情绪强度: 强烈，请给予充分关注和共情")
        elif result.intensity > 0.4:
            lines.append("情绪强度: 较强，请给予充分关注")
        elif result.intensity > 0.1:
            lines.append("情绪强度: 轻微，适当关注即可")

        # 匹配到的关键词
        if result.matched_keywords:
            keywords_str = "、".join(result.matched_keywords[:5])
            lines.append(f"情绪相关表达: {keywords_str}")

        # 悲伤分数（如果有的话）
        sad_score = result.emotion_scores.get("sad", 0)
        if sad_score > 0:
            lines.append(f"悲伤指数: {sad_score:.0f}%")

        # 危机指标
        if result.risk_indicators:
            lines.append("")
            lines.append("⚠️ 重要: 用户可能处于情绪危机中，请以最高优先级给予关怀")
            if len(result.risk_indicators) <= 3:
                lines.append(f"危机信号: {'、'.join(result.risk_indicators)}")

        # 需要关怀
        if result.needs_support and not result.risk_indicators:
            lines.append("提示: 用户可能需要情感支持，请温柔回应")

        return "\n".join(lines) + "\n"


# 单例访问函数
_text_emotion_analyzer_instance: Optional[TextEmotionAnalyzer] = None
_text_emotion_analyzer_lock = Lock()


def get_text_emotion_analyzer() -> TextEmotionAnalyzer:
    """获取文本情绪分析器单例"""
    global _text_emotion_analyzer_instance

    if _text_emotion_analyzer_instance is None:
        with _text_emotion_analyzer_lock:
            if _text_emotion_analyzer_instance is None:
                _text_emotion_analyzer_instance = TextEmotionAnalyzer()

    return _text_emotion_analyzer_instance

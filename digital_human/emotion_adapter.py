# -*- coding: utf-8 -*-
"""
情绪识别适配器

基于 DeepFace 实现实时面部情绪识别，支持：
- 单帧情绪检测
- 时空平滑（滑动窗口众数过滤）
- 情绪上下文管理（带 TTL 过期）
- LLM Prompt 上下文生成
"""

import os
import time
import base64
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import Counter, deque
from threading import Lock

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """情绪识别结果"""
    dominant_emotion: str           # 主要情绪
    emotion_scores: Dict[str, float]  # 各情绪分数
    sad_score: float                # 悲伤分数（重点关注）
    risk_level: str                 # 风险等级: normal/warning/critical
    confidence: float               # 置信度
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dominant_emotion": self.dominant_emotion,
            "emotion_scores": self.emotion_scores,
            "sad_score": self.sad_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class EmotionSmoother:
    """
    时空平滑器

    使用滑动窗口对情绪进行平滑处理，采用众数（mode）过滤瞬时噪声。
    每个 session 独立维护一个滑动窗口。
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._windows: Dict[str, deque] = {}
        self._lock = Lock()

    def smooth(self, session_id: str, emotion: str) -> str:
        """
        平滑情绪结果

        Args:
            session_id: 会话 ID
            emotion: 当前帧检测到的情绪

        Returns:
            平滑后的情绪
        """
        with self._lock:
            if session_id not in self._windows:
                self._windows[session_id] = deque(maxlen=self.window_size)

            window = self._windows[session_id]
            window.append(emotion)

            # 计算众数
            if len(window) < 3:
                return emotion  # 窗口未满，直接返回

            counter = Counter(window)
            most_common = counter.most_common(1)[0][0]
            return most_common

    def clear_session(self, session_id: str):
        """清除指定会话的滑动窗口"""
        with self._lock:
            if session_id in self._windows:
                del self._windows[session_id]


@dataclass
class EmotionContextSnapshot:
    """情绪上下文快照"""
    dominant_emotion: str
    sad_score: float
    risk_level: str
    timestamp: float
    emotion_scores: Dict[str, float]


class EmotionContext:
    """
    情绪上下文管理器

    维护每个 session 的情绪历史记录，支持 TTL 过期。
    用于生成 LLM Prompt 上下文。
    """

    def __init__(self, ttl_seconds: int = 30):
        self.ttl_seconds = ttl_seconds
        self._contexts: Dict[str, List[EmotionContextSnapshot]] = {}
        self._lock = Lock()

    def add(self, session_id: str, result: EmotionResult):
        """添加情绪结果到上下文"""
        with self._lock:
            if session_id not in self._contexts:
                self._contexts[session_id] = []

            snapshot = EmotionContextSnapshot(
                dominant_emotion=result.dominant_emotion,
                sad_score=result.sad_score,
                risk_level=result.risk_level,
                timestamp=result.timestamp,
                emotion_scores=result.emotion_scores,
            )
            self._contexts[session_id].append(snapshot)

            # 清理过期记录
            self._cleanup(session_id)

    def get_recent(self, session_id: str, limit: int = 10) -> List[EmotionContextSnapshot]:
        """获取最近的情绪上下文"""
        with self._lock:
            if session_id not in self._contexts:
                return []

            self._cleanup(session_id)
            return self._contexts[session_id][-limit:]

    def get_summary(self, session_id: str) -> Dict[str, Any]:
        """获取情绪上下文摘要"""
        recent = self.get_recent(session_id)

        if not recent:
            return {
                "has_data": False,
                "message": "暂无情绪数据",
            }

        # 统计主要情绪分布
        emotion_counter = Counter(s.dominant_emotion for s in recent)

        # 计算平均悲伤分数
        avg_sad = sum(s.sad_score for s in recent) / len(recent)

        # 获取最新风险等级
        latest_risk = recent[-1].risk_level if recent else "normal"

        # 检测情绪趋势
        if len(recent) >= 3:
            recent_sad = [s.sad_score for s in recent[-3:]]
            trend = "rising" if recent_sad[-1] > recent_sad[0] else "falling" if recent_sad[-1] < recent_sad[0] else "stable"
        else:
            trend = "unknown"

        return {
            "has_data": True,
            "dominant_emotions": dict(emotion_counter.most_common(3)),
            "avg_sad_score": round(avg_sad, 2),
            "latest_risk_level": latest_risk,
            "trend": trend,
            "sample_count": len(recent),
        }

    def get_prompt_context(self, session_id: str) -> str:
        """
        生成 LLM Prompt 上下文

        Returns:
            供 LLM 使用的情绪上下文提示文本
        """
        summary = self.get_summary(session_id)

        if not summary["has_data"]:
            return ""

        lines = ["[用户情绪状态]"]

        # 主要情绪
        if summary["dominant_emotions"]:
            top_emotion = list(summary["dominant_emotions"].keys())[0]
            lines.append(f"当前主要情绪: {top_emotion}")

        # 悲伤分数
        sad_score = summary["avg_sad_score"]
        if sad_score > 60:
            lines.append(f"悲伤指数较高: {sad_score:.1f}")

        # 风险等级
        if summary["latest_risk_level"] == "warning":
            lines.append("情绪状态: 需要关注")
        elif summary["latest_risk_level"] == "critical":
            lines.append("情绪状态: 需要重点关怀")

        # 趋势
        if summary["trend"] == "rising" and sad_score > 50:
            lines.append("趋势: 悲伤情绪在上升")

        return "\n".join(lines) + "\n"

    def clear_session(self, session_id: str):
        """清除指定会话的上下文"""
        with self._lock:
            if session_id in self._contexts:
                del self._contexts[session_id]

    def _cleanup(self, session_id: str):
        """清理过期的情绪记录"""
        if session_id not in self._contexts:
            return

        now = time.time()
        cutoff = now - self.ttl_seconds
        self._contexts[session_id] = [
            s for s in self._contexts[session_id]
            if s.timestamp > cutoff
        ]


class EmotionAdapter:
    """
    情绪识别适配器

    单例模式，封装 DeepFace 情绪识别功能。
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
        self.config = config
        self._deepface = None
        self._smoother = None
        self._context = None

        # 获取配置
        if config is None:
            from .config import get_default_config
            config = get_default_config().emotion_detection

        self.enabled = config.enabled
        self.backend = config.detector_backend
        self.enforce_detection = config.enforce_detection
        self.smoothing_window = config.smoothing_window
        self.sad_warning_threshold = config.sad_score_warning
        self.sad_critical_threshold = config.sad_score_critical
        self.context_ttl = config.context_ttl
        self.timeout = config.analysis_timeout

        if self.enabled:
            self._init_deepface()

        self._smoother = EmotionSmoother(self.smoothing_window)
        self._context = EmotionContext(self.context_ttl)
        self._initialized = True

        logger.info(f"EmotionAdapter initialized: enabled={self.enabled}, backend={self.backend}")

    def _init_deepface(self):
        """初始化 DeepFace"""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            logger.info("DeepFace loaded successfully")
        except ImportError:
            logger.warning("DeepFace not installed, emotion detection disabled")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to load DeepFace: {e}")
            self.enabled = False

    def detect_from_frame(self, frame: np.ndarray, session_id: str = "default") -> Optional[EmotionResult]:
        """
        从 OpenCV 帧检测情绪

        Args:
            frame: OpenCV 图像帧 (BGR 格式)
            session_id: 会话 ID

        Returns:
            EmotionResult 或 None（检测失败时）
        """
        if not self.enabled or self._deepface is None:
            return self._get_neutral_result()

        try:
            # DeepFace 分析
            result = self._deepface.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=self.enforce_detection,
                detector_backend=self.backend,
                silent=True,
            )

            # 解析结果
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            emotion_scores = result.get("emotion", {})
            dominant_emotion = result.get("dominant_emotion", "neutral")

            # 平滑处理
            smoothed_emotion = self._smoother.smooth(session_id, dominant_emotion)

            # 计算悲伤分数
            sad_score = emotion_scores.get("sad", 0.0)

            # 确定风险等级
            risk_level = self._get_risk_level(sad_score)

            # 计算置信度
            confidence = emotion_scores.get(smoothed_emotion, 0.0)

            em_result = EmotionResult(
                dominant_emotion=smoothed_emotion,
                emotion_scores={k: round(v, 2) for k, v in emotion_scores.items()},
                sad_score=round(sad_score, 2),
                risk_level=risk_level,
                confidence=round(confidence, 2),
            )

            # 添加到上下文
            self._context.add(session_id, em_result)

            return em_result

        except ValueError as e:
            # 无人脸检测到
            if "Face could not be detected" in str(e):
                logger.debug(f"No face detected in frame for session {session_id}")
                return self._get_neutral_result()
            raise

        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return None

    def detect_from_base64(self, base64_image: str, session_id: str = "default") -> Optional[EmotionResult]:
        """
        从 Base64 编码图像检测情绪

        Args:
            base64_image: Base64 编码的图像（支持 data:image/xxx;base64, 前缀）
            session_id: 会话 ID

        Returns:
            EmotionResult 或 None
        """
        try:
            # 移除 data URL 前缀
            if "," in base64_image:
                base64_image = base64_image.split(",")[1]

            # 解码 Base64
            image_data = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("Failed to decode base64 image")
                return None

            return self.detect_from_frame(frame, session_id)

        except Exception as e:
            logger.error(f"Base64 decode failed: {e}")
            return None

    def get_context(self, session_id: str = "default") -> Dict[str, Any]:
        """获取情绪上下文摘要"""
        return self._context.get_summary(session_id)

    def get_prompt_context(self, session_id: str = "default") -> str:
        """获取 LLM Prompt 上下文"""
        return self._context.get_prompt_context(session_id)

    def clear_session(self, session_id: str):
        """清除会话数据"""
        self._smoother.clear_session(session_id)
        self._context.clear_session(session_id)

    def _get_risk_level(self, sad_score: float) -> str:
        """根据悲伤分数确定风险等级"""
        if sad_score >= self.sad_critical_threshold:
            return "critical"
        elif sad_score >= self.sad_warning_threshold:
            return "warning"
        return "normal"

    def _get_neutral_result(self) -> EmotionResult:
        """获取默认中性结果"""
        return EmotionResult(
            dominant_emotion="neutral",
            emotion_scores={"neutral": 100.0},
            sad_score=0.0,
            risk_level="normal",
            confidence=100.0,
        )


# 单例访问函数
_emotion_adapter_instance: Optional[EmotionAdapter] = None
_emotion_adapter_lock = Lock()


def get_emotion_adapter(config: Optional[Any] = None) -> EmotionAdapter:
    """获取情绪适配器单例"""
    global _emotion_adapter_instance

    if _emotion_adapter_instance is None:
        with _emotion_adapter_lock:
            if _emotion_adapter_instance is None:
                _emotion_adapter_instance = EmotionAdapter(config)

    return _emotion_adapter_instance

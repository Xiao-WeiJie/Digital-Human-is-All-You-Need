# -*- coding: utf-8 -*-
"""
Digital Human 模块

统一入口
"""

from .config import (
    PROJECT_ROOT,
    DEFAULT_PATHS,
    GPTSoVITSConfig,
    SenseVoiceConfig,
    DashScopeConfig,
    AvatarInfo,
    AvatarConfig,
    VideoConfig,
    AudioConfig,
    DigitalHumanConfig,
    get_default_config,
    reload_config,
    EmotionDetectionConfig,
    EmotionFusionConfig,
)

from .emotion_adapter import (
    EmotionAdapter,
    EmotionResult,
    get_emotion_adapter,
)

from .text_emotion_analyzer import (
    TextEmotionAnalyzer,
    TextEmotionResult,
    get_text_emotion_analyzer,
)

from .emotion_fusion import (
    EmotionFusion,
    FusedEmotionContext,
    get_emotion_fusion,
)

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_PATHS",
    "GPTSoVITSConfig",
    "SenseVoiceConfig",
    "DashScopeConfig",
    "AvatarInfo",
    "AvatarConfig",
    "VideoConfig",
    "AudioConfig",
    "DigitalHumanConfig",
    "get_default_config",
    "reload_config",
    # 情绪识别
    "EmotionDetectionConfig",
    "EmotionAdapter",
    "EmotionResult",
    "get_emotion_adapter",
    # 情绪融合
    "EmotionFusionConfig",
    "TextEmotionAnalyzer",
    "TextEmotionResult",
    "get_text_emotion_analyzer",
    "EmotionFusion",
    "FusedEmotionContext",
    "get_emotion_fusion",
]

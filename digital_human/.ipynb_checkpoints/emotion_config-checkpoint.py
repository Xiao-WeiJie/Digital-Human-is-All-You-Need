# -*- coding: utf-8 -*-
"""
情感表情配置

定义情感对应的表情偏移和头部动作参数
所有参数都可以在这里调整
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class EyeClampConfig:
    """
    眼睛夸张度限制配置

    用于防止眼睛睁得过于夸张
    """
    # 是否启用眼睛限制
    enable: bool = True

    # 眼睛 exp 系数最小值（控制闭眼程度）
    # 值越大（绝对值越小），眼睛越不容易完全闭上
    clamp_min: float = -0.12

    # 眼睛 exp 系数最大值（控制睁眼程度）
    # 值越小，眼睛越不容易睁大
    clamp_max: float = 0.12

    # 眼睛缩放系数 (0-1)
    # 值越小，眼睛动作幅度越小
    scale: float = 0.7


@dataclass
class HappyExpressionConfig:
    """
    开心表情配置

    控制开心时的表情偏移量
    """
    # 是否启用开心表情
    enable: bool = True

    # 嘴角上扬幅度 (索引 11, 15)
    # 值越大，嘴角上扬越明显
    mouth_offset: float = 0.015

    # 眼睛微眯幅度 (索引 6, 8)
    # 值越大，眯眼越明显
    eye_offset: float = 0.008

    # 脸颊幅度 (索引 16, 18)
    # 值越大，脸颊越鼓
    cheek_offset: float = 0.004

    # 轻微点头幅度 (弧度)
    nod_amplitude: float = 0.02

    # 点头频率 (Hz)
    nod_frequency: float = 0.5


@dataclass
class SadExpressionConfig:
    """
    悲伤表情配置

    控制悲伤时的表情偏移和摇头动作
    """
    # 是否启用悲伤表情
    enable: bool = True

    # 眉毛下垂幅度 (索引 3, 4, 5)
    # 值越小（负值），眉毛下垂越明显
    eyebrow_offset: float = -0.015

    # 嘴角下垂幅度 (索引 11, 15)
    # 值越小（负值），嘴角下垂越明显
    mouth_offset: float = -0.008

    # 摇头幅度 (弧度)
    # 约 1 弧度 = 57 度，0.08 弧度约 4.5 度
    shake_amplitude: float = 0.08

    # 摇头频率 (Hz)
    # 0.3 表示约每 3 秒一个完整周期
    shake_frequency: float = 0.3


@dataclass
class QuestionExpressionConfig:
    """
    疑问表情配置

    控制疑问时的表情偏移和歪头动作
    """
    # 是否启用疑问表情
    enable: bool = True

    # 眉毛上挑幅度 (索引 3, 4, 5)
    eyebrow_offset: float = 0.01

    # 歪头幅度 (弧度)
    tilt_amplitude: float = 0.03


@dataclass
class CalmExpressionConfig:
    """
    平静表情配置

    平静时基本不做额外动作
    """
    enable: bool = True


@dataclass
class EmotionExpressionConfig:
    """
    情感表情总配置

    包含所有情感的配置参数
    """
    # 功能开关
    enable_emotion_offset: bool = True
    enable_head_motion: bool = True

    # 各情感配置
    eye_clamp: EyeClampConfig = field(default_factory=EyeClampConfig)
    happy: HappyExpressionConfig = field(default_factory=HappyExpressionConfig)
    sad: SadExpressionConfig = field(default_factory=SadExpressionConfig)
    question: QuestionExpressionConfig = field(default_factory=QuestionExpressionConfig)
    calm: CalmExpressionConfig = field(default_factory=CalmExpressionConfig)


def get_emotion_config() -> EmotionExpressionConfig:
    """获取情感配置实例"""
    return EmotionExpressionConfig()


# ============================================================================
# 参数调优指南
# ============================================================================
#
# 【眼睛限制参数】(EyeClampConfig)
# - 问题：眼睛还是太大
#   解决：减小 clamp_max (如 0.10) 或减小 scale (如 0.6)
# - 问题：眼睛张不开
#   解决：增大 clamp_max (如 0.15)
# - 问题：眨眼不自然
#   解决：增大 clamp_min 的绝对值 (如 -0.10)
#
# 【开心表情参数】(HappyExpressionConfig)
# - 问题：笑容不明显
#   解决：增大 mouth_offset (如 0.02)
# - 问题：表情太夸张
#   解决：减小所有偏移量 (如 mouth_offset: 0.01)
# - 问题：不够自然
#   解决：配合 TTS 开心参考音频，减小偏移量
#
# 【悲伤摇头参数】(SadExpressionConfig)
# - 问题：摇头幅度太大
#   解决：减小 shake_amplitude (如 0.05)
# - 问题：摇头太频繁
#   解决：减小 shake_frequency (如 0.2)
# - 问题：动作不自然
#   解决：调整 shake_frequency 到 0.2-0.4 之间
#
# ============================================================================

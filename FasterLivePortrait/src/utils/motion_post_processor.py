# -*- coding: utf-8 -*-
"""
Motion 后处理模块

功能：
1. 眼睛夸张度 clamp - 防止眼睛睁得过于夸张
2. 情感表情偏移叠加 - 根据情感标签叠加表情偏移
3. 头部动作生成 - 悲伤时摇头、开心时轻微点头等

作者：Claude
日期：2026-03-14
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import math
import copy


class MotionPostProcessor:
    """Motion 后处理器"""

    # 眼睛关键点索引
    EYE_INDICES = [6, 7, 8]

    # 嘴巴关键点索引
    MOUTH_INDICES = [11, 12, 13, 14, 15]

    # 眉毛关键点索引
    EYEBROW_INDICES = [3, 4, 5]

    # 脸颊关键点索引
    CHEEK_INDICES = [16, 17, 18]

    def __init__(self, config=None):
        """
        初始化后处理器

        Args:
            config: EmotionExpressionConfig 配置实例
        """
        self.config = config

        # 从配置读取参数，如果配置为 None 则使用默认值
        if config is None:
            self._use_defaults()
        else:
            self._load_from_config(config)

        # 头部动作生成器状态
        self.head_motion_state = {
            "phase": 0.0,
            "speed": 0.0,
            "amplitude": 0.0,
            "type": "neutral"
        }

    def _use_defaults(self):
        """使用默认参数"""
        # 眼睛限制
        self.enable_eye_clamp = True
        self.eye_clamp_min = -0.12
        self.eye_clamp_max = 0.12
        self.eye_scale = 0.7

        # 口型增强 - 放大嘴部动作幅度
        self.enable_lip_scale = True
        self.lip_scale = 1.8  # 放大口型动作，解决嘴唇张不开的问题

        # 功能开关
        self.enable_emotion_offset = True
        self.enable_head_motion = True

    def _load_from_config(self, config):
        """从配置加载参数"""
        # 眼睛限制
        eye_config = config.eye_clamp
        self.enable_eye_clamp = eye_config.enable
        self.eye_clamp_min = eye_config.clamp_min
        self.eye_clamp_max = eye_config.clamp_max
        self.eye_scale = eye_config.scale

        # 口型增强
        if hasattr(config, 'lip_scale'):
            self.enable_lip_scale = getattr(config.lip_scale, 'enable', True)
            self.lip_scale = getattr(config.lip_scale, 'scale', 1.8)
        else:
            self.enable_lip_scale = True
            self.lip_scale = 1.8

        # 功能开关
        self.enable_emotion_offset = config.enable_emotion_offset
        self.enable_head_motion = config.enable_head_motion

        # 保存配置引用
        self._config = config

    def _get_emotion_config(self, emotion: str) -> Dict:
        """
        获取指定情感的配置

        Args:
            emotion: 情感标签

        Returns:
            情感配置字典
        """
        if self.config is None:
            return self._get_default_emotion_config(emotion)

        config = self.config

        if emotion == "happy":
            happy_cfg = config.happy
            if not happy_cfg.enable:
                return {"exp_offset": {}, "head_motion": {"type": "neutral"}}
            return {
                "exp_offset": {
                    # 嘴角上扬
                    11: np.array([happy_cfg.mouth_offset, happy_cfg.mouth_offset * 0.6, 0.0]),
                    15: np.array([happy_cfg.mouth_offset, -happy_cfg.mouth_offset * 0.6, 0.0]),
                    # 眼睛微眯
                    6: np.array([0.0, happy_cfg.eye_offset, 0.0]),
                    8: np.array([0.0, happy_cfg.eye_offset, 0.0]),
                    # 脸颊
                    16: np.array([happy_cfg.cheek_offset, 0.0, 0.0]),
                    18: np.array([happy_cfg.cheek_offset, 0.0, 0.0]),
                },
                "head_motion": {
                    "type": "slight_nod",
                    "amplitude": happy_cfg.nod_amplitude,
                    "frequency": happy_cfg.nod_frequency,
                }
            }

        elif emotion == "sad":
            sad_cfg = config.sad
            if not sad_cfg.enable:
                return {"exp_offset": {}, "head_motion": {"type": "neutral"}}
            return {
                "exp_offset": {
                    # 眉毛下垂
                    3: np.array([0.0, sad_cfg.eyebrow_offset, 0.0]),
                    4: np.array([0.0, sad_cfg.eyebrow_offset, 0.0]),
                    5: np.array([0.0, sad_cfg.eyebrow_offset * 0.6, 0.0]),
                    # 嘴角下垂
                    11: np.array([0.0, sad_cfg.mouth_offset, 0.0]),
                    15: np.array([0.0, sad_cfg.mouth_offset, 0.0]),
                },
                "head_motion": {
                    "type": "slow_shake",
                    "amplitude": sad_cfg.shake_amplitude,
                    "frequency": sad_cfg.shake_frequency,
                }
            }

        elif emotion == "question":
            question_cfg = config.question
            if not question_cfg.enable:
                return {"exp_offset": {}, "head_motion": {"type": "neutral"}}
            return {
                "exp_offset": {
                    # 眉毛上挑
                    3: np.array([0.0, question_cfg.eyebrow_offset, 0.0]),
                    4: np.array([0.0, question_cfg.eyebrow_offset, 0.0]),
                    5: np.array([0.0, question_cfg.eyebrow_offset * 0.8, 0.0]),
                },
                "head_motion": {
                    "type": "slight_tilt",
                    "amplitude": question_cfg.tilt_amplitude,
                    "frequency": 0.0,
                }
            }

        else:  # default, calm, 等
            return {"exp_offset": {}, "head_motion": {"type": "neutral"}}

    def _get_default_emotion_config(self, emotion: str) -> Dict:
        """获取默认情感配置（无配置时使用）"""
        emotion_offsets = {
            "happy": {
                "exp_offset": {
                    11: np.array([0.015, 0.01, 0.0]),
                    15: np.array([0.015, -0.01, 0.0]),
                    6: np.array([0.0, 0.008, 0.0]),
                    8: np.array([0.0, 0.008, 0.0]),
                    16: np.array([0.004, 0.0, 0.0]),
                    18: np.array([0.004, 0.0, 0.0]),
                },
                "head_motion": {
                    "type": "slight_nod",
                    "amplitude": 0.02,
                    "frequency": 0.5,
                }
            },
            "sad": {
                "exp_offset": {
                    3: np.array([0.0, -0.015, 0.0]),
                    4: np.array([0.0, -0.015, 0.0]),
                    5: np.array([0.0, -0.01, 0.0]),
                    11: np.array([0.0, -0.008, 0.0]),
                    15: np.array([0.0, -0.008, 0.0]),
                },
                "head_motion": {
                    "type": "slow_shake",
                    "amplitude": 0.08,
                    "frequency": 0.3,
                }
            },
            "question": {
                "exp_offset": {
                    3: np.array([0.0, 0.01, 0.0]),
                    4: np.array([0.0, 0.01, 0.0]),
                    5: np.array([0.0, 0.008, 0.0]),
                },
                "head_motion": {
                    "type": "slight_tilt",
                    "amplitude": 0.03,
                    "frequency": 0.0,
                }
            },
            "calm": {
                "exp_offset": {},
                "head_motion": {"type": "neutral"}
            },
            "default": {
                "exp_offset": {},
                "head_motion": {"type": "neutral"}
            }
        }
        return emotion_offsets.get(emotion, emotion_offsets["default"])

    def process(
        self,
        motion_data: Dict,
        emotion: str = "default",
        **kwargs
    ) -> Dict:
        """
        处理 motion 数据

        Args:
            motion_data: JoyVASA 输出的 motion 数据
            emotion: 情感标签 (happy/sad/calm/question/default)
            **kwargs: 额外参数

        Returns:
            处理后的 motion 数据
        """
        # 深拷贝避免修改原始数据
        processed = {
            'n_frames': motion_data['n_frames'],
            'output_fps': motion_data['output_fps'],
            'motion': [],
            'c_eyes_lst': motion_data.get('c_eyes_lst', []),
            'c_lip_lst': motion_data.get('c_lip_lst', []),
        }

        # 获取情感配置
        emotion_cfg = self._get_emotion_config(emotion)

        # 重置头部动作状态
        self._reset_head_motion_state(emotion_cfg.get("head_motion", {}))

        # 处理每一帧
        for frame_idx, frame in enumerate(motion_data['motion']):
            processed_frame = self._process_frame(
                frame,
                frame_idx,
                emotion_cfg,
                motion_data['output_fps']
            )
            processed['motion'].append(processed_frame)

        return processed

    def _process_frame(
        self,
        frame: Dict,
        frame_idx: int,
        emotion_cfg: Dict,
        fps: int
    ) -> Dict:
        """处理单帧"""
        # 深拷贝帧数据
        processed = {
            "exp": frame["exp"].copy(),
            "scale": frame["scale"].copy(),
            "R": frame["R"].copy(),
            "t": frame["t"].copy(),
            "pitch": frame["pitch"].copy(),
            "yaw": frame["yaw"].copy(),
            "roll": frame["roll"].copy(),
        }

        # 1. 口型增强（优先处理，放大嘴部动作）
        if self.enable_lip_scale:
            processed = self._scale_lip(processed)

        # 2. 眼睛 clamp（始终执行，与情感无关）
        if self.enable_eye_clamp:
            processed = self._clamp_eyes(processed)

        # 3. 情感表情偏移
        if self.enable_emotion_offset:
            exp_offset = emotion_cfg.get("exp_offset", {})
            if exp_offset:
                processed = self._apply_emotion_offset(processed, exp_offset)

        # 4. 头部动作
        if self.enable_head_motion:
            head_motion_cfg = emotion_cfg.get("head_motion", {})
            if head_motion_cfg.get("type") != "neutral":
                processed = self._apply_head_motion(
                    processed,
                    frame_idx,
                    fps,
                    head_motion_cfg
                )

        return processed

    def _scale_lip(self, frame: Dict) -> Dict:
        """
        放大口型动作幅度

        JoyVASA 生成的口型动作幅度较小，通过缩放嘴部关键点来增强
        嘴部关键点索引: [11, 12, 13, 14, 15]
        - 11: 左嘴角
        - 12: 上嘴唇中点
        - 13: 下嘴唇中点
        - 14: 右嘴角
        - 15: 嘴巴中心
        """
        exp = frame["exp"]  # shape: (1, 21, 3)

        for idx in self.MOUTH_INDICES:
            # 对嘴部关键点的 Y 方向（垂直方向）进行放大
            # 这是嘴唇张开的主要方向
            exp[:, idx, 1] *= self.lip_scale

            # X 方向（水平方向）也适当放大，让嘴型更明显
            exp[:, idx, 0] *= (1 + (self.lip_scale - 1) * 0.3)

            # Z 方向（深度）保持原样或轻微调整
            exp[:, idx, 2] *= (1 + (self.lip_scale - 1) * 0.1)

        frame["exp"] = exp
        return frame

    def _clamp_eyes(self, frame: Dict) -> Dict:
        """
        限制眼睛夸张度

        通过缩放和 clamp 两种方式限制眼睛的运动幅度
        """
        exp = frame["exp"]  # shape: (1, 21, 3)

        for idx in self.EYE_INDICES:
            # 方法1: 缩放 - 整体减小眼睛动作幅度
            exp[:, idx, :] *= self.eye_scale

            # 方法2: clamp - 限制最大/最小值
            exp[:, idx, 0] = np.clip(exp[:, idx, 0], self.eye_clamp_min, self.eye_clamp_max)
            exp[:, idx, 1] = np.clip(exp[:, idx, 1], self.eye_clamp_min, self.eye_clamp_max)
            # Z 方向（深度）变化较小，用更小的范围
            exp[:, idx, 2] = np.clip(exp[:, idx, 2], self.eye_clamp_min * 0.5, self.eye_clamp_max * 0.5)

        frame["exp"] = exp
        return frame

    def _apply_emotion_offset(self, frame: Dict, exp_offset: Dict[int, np.ndarray]) -> Dict:
        """
        应用情感表情偏移

        在原始 exp 系数上叠加情感偏移量
        """
        exp = frame["exp"]

        for idx, offset in exp_offset.items():
            if 0 <= idx < 21:
                exp[:, idx, :] += offset

        frame["exp"] = exp
        return frame

    def _reset_head_motion_state(self, head_motion_cfg: Dict):
        """重置头部动作状态"""
        motion_type = head_motion_cfg.get("type", "neutral")

        if motion_type == "slow_shake":
            # 悲伤摇头 - yaw 轴周期性摆动
            self.head_motion_state = {
                "phase": 0.0,
                "speed": head_motion_cfg.get("frequency", 0.3) * 2 * math.pi,
                "amplitude": head_motion_cfg.get("amplitude", 0.08),
                "type": "slow_shake"
            }
        elif motion_type == "slight_nod":
            # 开心点头 - pitch 轴周期性摆动
            self.head_motion_state = {
                "phase": 0.0,
                "speed": head_motion_cfg.get("frequency", 0.5) * 2 * math.pi,
                "amplitude": head_motion_cfg.get("amplitude", 0.02),
                "type": "slight_nod"
            }
        elif motion_type == "slight_tilt":
            # 疑问歪头 - roll 轴固定偏移
            self.head_motion_state = {
                "phase": 0.0,
                "speed": 0.0,
                "amplitude": head_motion_cfg.get("amplitude", 0.03),
                "type": "slight_tilt"
            }
        else:
            # neutral - 无额外动作
            self.head_motion_state = {
                "phase": 0.0,
                "speed": 0.0,
                "amplitude": 0.0,
                "type": "neutral"
            }

    def _apply_head_motion(
        self,
        frame: Dict,
        frame_idx: int,
        fps: int,
        head_motion_cfg: Dict
    ) -> Dict:
        """
        应用头部动作

        根据动��类型修改 pitch/yaw/roll 角度
        """
        motion_type = self.head_motion_state["type"]
        dt = 1.0 / fps

        if motion_type == "slow_shake":
            # 悲伤时的缓慢摇头 (yaw 轴)
            self.head_motion_state["phase"] += self.head_motion_state["speed"] * dt
            yaw_offset = self.head_motion_state["amplitude"] * math.sin(self.head_motion_state["phase"])

            # 叠加到原始 yaw
            frame["yaw"][0, 0] += yaw_offset

        elif motion_type == "slight_nod":
            # 开心时的轻微点头 (pitch 轴)
            self.head_motion_state["phase"] += self.head_motion_state["speed"] * dt
            pitch_offset = self.head_motion_state["amplitude"] * math.sin(self.head_motion_state["phase"])

            frame["pitch"][0, 0] += pitch_offset

        elif motion_type == "slight_tilt":
            # 疑问时的轻微歪头 (roll 轴) - 固定偏移
            frame["roll"][0, 0] += self.head_motion_state["amplitude"]

        # 更新旋转矩阵
        try:
            # 尝试从 FasterLivePortrait 导入
            from FasterLivePortrait.src.utils.utils import get_rotation_matrix
        except ImportError:
            # 如果导入失败，使用本地实现
            get_rotation_matrix = self._get_rotation_matrix

        frame["R"] = get_rotation_matrix(
            frame["pitch"][0, 0],
            frame["yaw"][0, 0],
            frame["roll"][0, 0]
        )

        return frame

    def _get_rotation_matrix(self, pitch: float, yaw: float, roll: float) -> np.ndarray:
        """
        从欧拉角计算旋转矩阵

        Args:
            pitch: 俯仰角 (弧度)
            yaw: 偏航角 (弧度)
            roll: 翻滚角 (弧度)

        Returns:
            3x3 旋转矩阵
        """
        # 计算各轴旋转矩阵
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])

        # 组合旋转: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx
        return R.reshape(1, 3, 3).astype(np.float32)


# 全局实例（延迟初始化）
_motion_post_processor_instance: Optional[MotionPostProcessor] = None


def get_motion_post_processor(config=None) -> MotionPostProcessor:
    """
    获取全局 MotionPostProcessor 实例

    Args:
        config: 配置实例，如果为 None 则使用默认配置

    Returns:
        MotionPostProcessor 实例
    """
    global _motion_post_processor_instance

    if _motion_post_processor_instance is None:
        _motion_post_processor_instance = MotionPostProcessor(config)

    return _motion_post_processor_instance


def reset_motion_post_processor():
    """重置全局实例（用于配置变更后）"""
    global _motion_post_processor_instance
    _motion_post_processor_instance = None
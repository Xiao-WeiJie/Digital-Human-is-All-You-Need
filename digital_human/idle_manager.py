# -*- coding: utf-8 -*-
"""
Digital Human Phase 2 - Idle State Manager

管理数字人的待机和说话状态。

功能：
- 待机状态：眨眼、轻微呼吸感、轻微表情变化
- 说话状态：嘴部运动、表情变化、头部动作
- 状态切换：平滑过渡

待机动画生成策略：
1. 预生成待机循环视频（推荐用于生产环境）
2. 实时生成微动作帧（用于开发测试）
"""

import asyncio
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import threading

import numpy as np


class AvatarState(Enum):
    """数字人状态枚举"""
    IDLE = "idle"          # 待机状态
    LISTENING = "listening"  # 听用户说话
    THINKING = "thinking"   # 思考/处理中
    TALKING = "talking"     # 说话/回复中
    ERROR = "error"         # 错误状态


@dataclass
class StateTransition:
    """状态转换记录"""
    from_state: AvatarState
    to_state: AvatarState
    timestamp: float
    reason: str = ""


class IdleMotionGenerator:
    """
    待机动作生成器

    生成微小的面部运动数据，用于待机状态。
    """

    def __init__(self, fps: int = 25):
        self.fps = fps
        # LivePortrait 关键点数量
        self.num_keypoints = 21

    def generate_blink_motion(self, duration_frames: int = 5) -> np.ndarray:
        """
        生成眨眼动作

        Args:
            duration_frames: 眨眼持续的帧数

        Returns:
            形状为 (duration_frames, 1, 21, 3) 的运动数据
        """
        # 眨眼主要影响眼睛周围的关键点
        # 简化：使用正弦曲线模拟眨眼
        t = np.linspace(0, np.pi, duration_frames)
        blink_curve = np.sin(t)

        motion = np.zeros((duration_frames, 1, self.num_keypoints, 3), dtype=np.float32)

        # 眼睛关键点索引（简化）
        # 实际需要根据 LivePortrait 的关键点定义调整
        eye_indices = [1, 2, 3, 4, 9, 10, 11, 12]  # 示例索引

        for idx in eye_indices:
            # Y 轴轻微移动模拟眨眼
            motion[:, 0, idx, 1] = blink_curve * 0.002

        return motion

    def generate_breath_motion(self, duration_frames: int) -> np.ndarray:
        """
        生成呼吸动作

        Args:
            duration_frames: 呼吸周期的帧数

        Returns:
            运动数据
        """
        # 呼吸周期约 3-5 秒
        t = np.linspace(0, 2 * np.pi, duration_frames)
        breath_curve = np.sin(t) * 0.001

        motion = np.zeros((duration_frames, 1, self.num_keypoints, 3), dtype=np.float32)

        # 整体轻微上下移动
        motion[:, 0, :, 1] = breath_curve[:, np.newaxis]

        return motion

    def generate_head_motion(self, duration_frames: int) -> np.ndarray:
        """
        生成轻微头部动作

        Args:
            duration_frames: 帧数

        Returns:
            运动数据
        """
        # 轻微的头部摆动
        t = np.linspace(0, 2 * np.pi, duration_frames)

        motion = np.zeros((duration_frames, 1, self.num_keypoints, 3), dtype=np.float32)

        # 头部关键点（简化）
        head_indices = list(range(self.num_keypoints))

        # X 轴轻微摆动
        motion[:, 0, head_indices, 0] = (np.sin(t * 0.5) * 0.001)[:, np.newaxis]
        # Y 轴轻微点头
        motion[:, 0, head_indices, 1] = (np.sin(t * 0.3) * 0.0005)[:, np.newaxis]
        # 轻微旋转（通过 Z 轴变化模拟）
        motion[:, 0, head_indices, 2] = (np.sin(t * 0.2) * 0.0003)[:, np.newaxis]

        return motion

    def generate_idle_sequence(
        self,
        duration_seconds: float = 10.0,
        blink_interval: float = 4.0,
    ) -> Dict[str, Any]:
        """
        生成完整的待机序列

        Args:
            duration_seconds: 总时长（秒）
            blink_interval: 眨眼间隔（秒）

        Returns:
            运动序列数据
        """
        total_frames = int(duration_seconds * self.fps)
        blink_interval_frames = int(blink_interval * self.fps)
        blink_duration = 5  # 眨眼持续帧数

        # 基础呼吸和头部动作
        breath_motion = self.generate_breath_motion(total_frames)
        head_motion = self.generate_head_motion(total_frames)

        # 合成动作
        combined = breath_motion + head_motion

        # 添加随机眨眼
        blink_frames = list(range(0, total_frames, blink_interval_frames))
        # 添加一些随机性
        np.random.seed(int(time.time()) % 1000)
        blink_frames = [f + np.random.randint(-10, 10) for f in blink_frames]
        blink_frames = [f for f in blink_frames if 0 <= f < total_frames - blink_duration]

        blink_motion = self.generate_blink_motion(blink_duration)
        for start_frame in blink_frames:
            end_frame = min(start_frame + blink_duration, total_frames)
            actual_duration = end_frame - start_frame
            combined[start_frame:end_frame] += blink_motion[:actual_duration]

        # 转换为 motion list 格式
        motion_list = []
        for i in range(total_frames):
            motion_list.append({
                "exp": combined[i],
                "rotation": np.eye(3, dtype=np.float32),
                "translation": np.zeros(3, dtype=np.float32),
            })

        return {
            "motion": motion_list,
            "n_frames": total_frames,
            "output_fps": self.fps,
            "fps": self.fps,
        }


class IdleVideoCache:
    """
    待机视频缓存

    预生成并缓存待机视频，避免重复渲染。
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = tempfile.gettempdir() / "digital_human_idle_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, str] = {}  # avatar_id -> video_path
        self._lock = threading.Lock()

    def get_idle_video(self, avatar_id: str) -> Optional[str]:
        """获取缓存的待机视频"""
        with self._lock:
            return self._cache.get(avatar_id)

    def set_idle_video(self, avatar_id: str, video_path: str):
        """设置待机视频缓存"""
        with self._lock:
            self._cache[avatar_id] = video_path

    def has_idle_video(self, avatar_id: str) -> bool:
        """检查是否有缓存"""
        return avatar_id in self._cache

    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            for video_path in self._cache.values():
                if os.path.exists(video_path):
                    os.remove(video_path)
            self._cache.clear()


class IdleStateManager:
    """
    待机状态管理器

    管理数字人的状态切换和待机动画。
    """

    def __init__(
        self,
        liveportrait_adapter=None,
        fps: int = 25,
        idle_video_duration: float = 10.0,
    ):
        self.liveportrait = liveportrait_adapter
        self.fps = fps
        self.idle_video_duration = idle_video_duration

        # 当前状态
        self._current_state = AvatarState.IDLE
        self._state_history: List[StateTransition] = []
        self._state_callbacks: Dict[AvatarState, List[Callable]] = {}

        # 待机动画
        self._motion_generator = IdleMotionGenerator(fps=fps)
        self._video_cache = IdleVideoCache()

        # 当前待机视频
        self._current_idle_video: Optional[str] = None
        self._current_avatar_id: Optional[str] = None

        # 状态变更锁
        self._lock = threading.Lock()

    def get_current_state(self) -> AvatarState:
        """获取当前状态"""
        return self._current_state

    def set_state(self, new_state: AvatarState, reason: str = ""):
        """
        设置新状态

        Args:
            new_state: 新状态
            reason: 状态变更原因
        """
        with self._lock:
            if new_state == self._current_state:
                return

            old_state = self._current_state
            self._current_state = new_state

            # 记录状态转换
            transition = StateTransition(
                from_state=old_state,
                to_state=new_state,
                timestamp=time.time(),
                reason=reason,
            )
            self._state_history.append(transition)

            print(f"[IdleManager] 状态变更: {old_state.value} -> {new_state.value} ({reason})")

            # 触发回调
            self._trigger_callbacks(new_state)

    def on_state_change(self, state: AvatarState, callback: Callable):
        """注册状态变更回调"""
        if state not in self._state_callbacks:
            self._state_callbacks[state] = []
        self._state_callbacks[state].append(callback)

    def _trigger_callbacks(self, state: AvatarState):
        """触发状态变更回调"""
        callbacks = self._state_callbacks.get(state, [])
        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                print(f"[IdleManager] 回调执行错误: {e}")

    def generate_idle_video(
        self,
        avatar_id: str,
        source_image: str,
        force_regenerate: bool = False,
    ) -> Optional[str]:
        """
        生成或获取待机视频

        Args:
            avatar_id: 数字人 ID
            source_image: 源图像路径
            force_regenerate: 是否强制重新生成

        Returns:
            待机视频路径
        """
        # 检查缓存
        if not force_regenerate and self._video_cache.has_idle_video(avatar_id):
            return self._video_cache.get_idle_video(avatar_id)

        if self.liveportrait is None:
            print("[IdleManager] LivePortrait 适配器未设置")
            return None

        try:
            # 生成待机动作序列
            print(f"[IdleManager] 生成待机视频: {avatar_id}")
            motion_data = self._motion_generator.generate_idle_sequence(
                duration_seconds=self.idle_video_duration
            )

            # 渲染视频
            output_path = str(self._video_cache.cache_dir / f"idle_{avatar_id}.mp4")
            video_path = self.liveportrait.render_video_from_motion(
                motion_data,
                source_image,
                output_path=output_path,
                audio_path=None,  # 待机视频无音频
            )

            # 缓存
            self._video_cache.set_idle_video(avatar_id, video_path)
            self._current_idle_video = video_path
            self._current_avatar_id = avatar_id

            print(f"[IdleManager] 待机视频已生成: {video_path}")
            return video_path

        except Exception as e:
            print(f"[IdleManager] 生成待机视频失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_idle_video(self, avatar_id: str) -> Optional[str]:
        """获取待机视频"""
        return self._video_cache.get_idle_video(avatar_id)

    def start_idle(self, avatar_id: str, source_image: str) -> Optional[str]:
        """
        开始待机状态

        Args:
            avatar_id: 数字人 ID
            source_image: 源图像路径

        Returns:
            待机视频路径
        """
        self.set_state(AvatarState.IDLE, "start_idle")
        return self.generate_idle_video(avatar_id, source_image)

    def start_listening(self):
        """开始监听状态"""
        self.set_state(AvatarState.LISTENING, "user_input_started")

    def start_thinking(self):
        """开始思考状态"""
        self.set_state(AvatarState.THINKING, "processing_request")

    def start_talking(self):
        """开始说话状态"""
        self.set_state(AvatarState.TALKING, "response_started")

    def stop_talking(self):
        """停止说话状态，回到待机"""
        self.set_state(AvatarState.IDLE, "response_completed")

    def set_error(self, error_msg: str = ""):
        """设置错误状态"""
        self.set_state(AvatarState.ERROR, error_msg)

    def get_state_history(self, limit: int = 10) -> List[StateTransition]:
        """获取状态历史"""
        return self._state_history[-limit:]

    def to_dict(self) -> dict:
        """转换为字典（用于 API 响应）"""
        return {
            "current_state": self._current_state.value,
            "current_avatar_id": self._current_avatar_id,
            "has_idle_video": self._current_idle_video is not None,
        }

# -*- coding: utf-8 -*-
"""
Digital Human Phase 2 - Avatar Configuration System

支持多数字人配置和切换。

每个 Avatar 包含：
- 基本信息：ID、名称、描述
- 源图像路径
- TTS Profile 配置
- 待机/说话状态配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


@dataclass
class TTSProfile:
    """
    TTS 配置档案

    每个 avatar 可以有独立的 TTS 配置，包括：
    - 参考音频路径（用于声音克隆）
    - 参考音频文本
    - 语言设置
    - 语速等参数
    """
    # 参考音频路径（服务器上的路径或远端 URL）
    ref_audio_path: Optional[str] = None
    # 参考音频对应文本
    prompt_text: str = ""
    # 参考文本语言
    prompt_lang: str = "zh"
    # 语速
    speed_factor: float = 1.0
    # 采样参数
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0

    def to_dict(self) -> dict:
        return {
            "ref_audio_path": self.ref_audio_path,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "speed_factor": self.speed_factor,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TTSProfile":
        return cls(
            ref_audio_path=data.get("ref_audio_path"),
            prompt_text=data.get("prompt_text", ""),
            prompt_lang=data.get("prompt_lang", "zh"),
            speed_factor=data.get("speed_factor", 1.0),
            top_k=data.get("top_k", 5),
            top_p=data.get("top_p", 1.0),
            temperature=data.get("temperature", 1.0),
        )


@dataclass
class IdleConfig:
    """
    待机状态配置

    控制数字人在无交互时的行为：
    - 是否启用待机动画
    - 眨眼频率
    - 轻微动作幅度
    """
    enabled: bool = True
    # 眨眼间隔（秒）
    blink_interval_min: float = 3.0
    blink_interval_max: float = 6.0
    # 呼吸幅度
    breath_amplitude: float = 0.002
    # 轻微头部摆动
    head_motion_amplitude: float = 0.005
    # 待机动画循环 FPS
    idle_fps: int = 25

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "blink_interval_min": self.blink_interval_min,
            "blink_interval_max": self.blink_interval_max,
            "breath_amplitude": self.breath_amplitude,
            "head_motion_amplitude": self.head_motion_amplitude,
            "idle_fps": self.idle_fps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IdleConfig":
        return cls(
            enabled=data.get("enabled", True),
            blink_interval_min=data.get("blink_interval_min", 3.0),
            blink_interval_max=data.get("blink_interval_max", 6.0),
            breath_amplitude=data.get("breath_amplitude", 0.002),
            head_motion_amplitude=data.get("head_motion_amplitude", 0.005),
            idle_fps=data.get("idle_fps", 25),
        )


@dataclass
class TalkingConfig:
    """
    说话状态配置

    控制数字人说话时的行为：
    - 动画参数
    - 唇部同步设置
    - 表情强度
    """
    # 整体运动幅度
    driving_multiplier: float = 1.0
    # CFG 强度
    cfg_scale: float = 1.35
    # 唇部开合放大
    lip_scale: float = 1.35
    # 是否启用拼接
    flag_stitching: bool = True
    # 是否贴回原图
    flag_pasteback: bool = True
    # 说话动画 FPS
    talking_fps: int = 25

    def to_dict(self) -> dict:
        return {
            "driving_multiplier": self.driving_multiplier,
            "cfg_scale": self.cfg_scale,
            "lip_scale": self.lip_scale,
            "flag_stitching": self.flag_stitching,
            "flag_pasteback": self.flag_pasteback,
            "talking_fps": self.talking_fps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TalkingConfig":
        return cls(
            driving_multiplier=data.get("driving_multiplier", 1.0),
            cfg_scale=data.get("cfg_scale", 1.35),
            lip_scale=data.get("lip_scale", 1.35),
            flag_stitching=data.get("flag_stitching", True),
            flag_pasteback=data.get("flag_pasteback", True),
            talking_fps=data.get("talking_fps", 25),
        )


@dataclass
class AvatarConfig:
    """
    数字人配置

    完整的数字人配置，包含所有相关设置。
    """
    # 唯一标识符
    avatar_id: str
    # 显示名称
    name: str
    # 源图像路径
    source_image: str
    # 描述
    description: str = ""
    # TTS 配置档案
    tts_profile: TTSProfile = field(default_factory=TTSProfile)
    # 待机配置
    idle_config: IdleConfig = field(default_factory=IdleConfig)
    # 说话配置
    talking_config: TalkingConfig = field(default_factory=TalkingConfig)
    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "avatar_id": self.avatar_id,
            "name": self.name,
            "source_image": self.source_image,
            "description": self.description,
            "tts_profile": self.tts_profile.to_dict(),
            "idle_config": self.idle_config.to_dict(),
            "talking_config": self.talking_config.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AvatarConfig":
        tts_data = data.get("tts_profile", {})
        idle_data = data.get("idle_config", {})
        talking_data = data.get("talking_config", {})

        return cls(
            avatar_id=data["avatar_id"],
            name=data["name"],
            source_image=data["source_image"],
            description=data.get("description", ""),
            tts_profile=TTSProfile.from_dict(tts_data) if tts_data else TTSProfile(),
            idle_config=IdleConfig.from_dict(idle_data) if idle_data else IdleConfig(),
            talking_config=TalkingConfig.from_dict(talking_data) if talking_data else TalkingConfig(),
            metadata=data.get("metadata", {}),
        )


class AvatarManager:
    """
    数字人管理器

    负责：
    - 加载和管理所有 avatar 配置
    - 自动发现 Human_Choice 目录中的数字人
    - 提供 avatar 切换和查询接口
    """

    # 支持的图像扩展名
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, human_choice_dir: Optional[str] = None, config_path: Optional[str] = None):
        """
        初始化 Avatar 管理器

        Args:
            human_choice_dir: 数字人资源目录，默认为 digital_human/Human_Choice
            config_path: 配置文件路径，默认为 digital_human/avatars.json
        """
        if human_choice_dir is None:
            # 默认路径
            base_dir = Path(__file__).parent
            human_choice_dir = base_dir / "Human_Choice"
            # 也检查服务器路径
            server_path = Path("/autodl-tmp/digital_human/Human_Choice")
            if server_path.exists():
                human_choice_dir = server_path

        self.human_choice_dir = Path(human_choice_dir)
        self._avatars: Dict[str, AvatarConfig] = {}
        self._current_avatar_id: Optional[str] = None

        # 1. 先从配置文件加载（如果存在）
        if config_path is None:
            base_dir = Path(__file__).parent
            config_path = base_dir / "avatars.json"

        config_file = Path(config_path)
        if config_file.exists():
            self.load_from_config(str(config_file))

        # 2. 自动发现并加载 avatars（会补充配置文件中未定义的）
        self._discover_avatars()

    def _discover_avatars(self):
        """自动发现 Human_Choice 目录中的数字人"""
        if not self.human_choice_dir.exists():
            print(f"[AvatarManager] 目录不存在: {self.human_choice_dir}")
            # 尝试其他路径
            alt_paths = [
                Path("/autodl-tmp/digital_human/Human_Choice"),
                Path("digital_human/Human_Choice"),
                Path(__file__).parent / "Human_Choice",
            ]
            for alt in alt_paths:
                if alt.exists():
                    self.human_choice_dir = alt
                    print(f"[AvatarManager] 使用替代路径: {alt}")
                    break
            else:
                print("[AvatarManager] 未找到有效的 Human_Choice 目录")
                return

        print(f"[AvatarManager] 扫描目录: {self.human_choice_dir}")

        # 扫描 Human_1, Human_2 等子目录
        for subdir in sorted(self.human_choice_dir.iterdir()):
            if not subdir.is_dir():
                continue

            # 跳过隐藏目录（如 .ipynb_checkpoints）
            if subdir.name.startswith("."):
                continue

            # 提取 ID（Human_1 -> human_1）
            dir_name = subdir.name
            if dir_name.startswith("Human_"):
                avatar_id = f"human_{dir_name[6:]}"
            else:
                # 跳过非 Human_ 开头的目录（避免扫描其他无关目录）
                print(f"[AvatarManager] 跳过非数字人目录: {dir_name}")
                continue

            # 查找源图像
            source_image = self._find_source_image(subdir)
            if not source_image:
                print(f"[AvatarManager] 跳过 {dir_name}：未找到源图像")
                continue

            # 创建默认配置
            avatar = AvatarConfig(
                avatar_id=avatar_id,
                name=f"数字人 {dir_name[6:]}" if dir_name.startswith("Human_") else dir_name,
                source_image=str(source_image),
                description=f"自动发现的数字人 - {dir_name}",
                tts_profile=self._create_default_tts_profile(avatar_id),
            )

            self._avatars[avatar_id] = avatar
            print(f"[AvatarManager] 发现数字人: {avatar_id} -> {source_image}")

        # 设置默认 avatar
        if self._avatars:
            self._current_avatar_id = list(self._avatars.keys())[0]
            print(f"[AvatarManager] 默认数字人: {self._current_avatar_id}")

    def _find_source_image(self, directory: Path) -> Optional[Path]:
        """在目录中查找源图像"""
        for ext in self.IMAGE_EXTENSIONS:
            matches = list(directory.glob(f"*{ext}")) + list(directory.glob(f"*{ext.upper()}"))
            if matches:
                return matches[0]
        return None

    def _create_default_tts_profile(self, avatar_id: str) -> TTSProfile:
        """为 avatar 创建默认 TTS profile"""
        # 默认使用共享的参考音频
        # 实际使用时可以从配置文件或环境变量加载
        default_ref_audio = os.environ.get(
            "GPT_SOVITS_REF_AUDIO",
            "/root/autodl-tmp/qys.wav"  # 服务器默认路径
        )
        default_prompt_text = os.environ.get(
            "GPT_SOVITS_PROMPT_TEXT",
            "清晨推开窗，就能闻到风里带着的青草香，楼下的花园里开着各色的花儿。"
        )

        return TTSProfile(
            ref_audio_path=default_ref_audio,
            prompt_text=default_prompt_text,
            prompt_lang="zh",
            speed_factor=1.0,
        )

    def get_avatar(self, avatar_id: str) -> Optional[AvatarConfig]:
        """获取指定 avatar 配置"""
        return self._avatars.get(avatar_id)

    def get_current_avatar(self) -> Optional[AvatarConfig]:
        """获取当前 avatar 配置"""
        if self._current_avatar_id:
            return self._avatars.get(self._current_avatar_id)
        return None

    def set_current_avatar(self, avatar_id: str) -> bool:
        """设置当前 avatar"""
        if avatar_id in self._avatars:
            self._current_avatar_id = avatar_id
            print(f"[AvatarManager] 切换到数字人: {avatar_id}")
            return True
        print(f"[AvatarManager] 未知的数字人 ID: {avatar_id}")
        return False

    def list_avatars(self) -> List[str]:
        """列出所有 avatar ID"""
        return list(self._avatars.keys())

    def get_all_avatars(self) -> Dict[str, AvatarConfig]:
        """获取所有 avatar 配置"""
        return self._avatars.copy()

    def add_avatar(self, avatar: AvatarConfig):
        """添加新的 avatar 配置"""
        self._avatars[avatar.avatar_id] = avatar

    def remove_avatar(self, avatar_id: str) -> bool:
        """移除 avatar 配置"""
        if avatar_id in self._avatars:
            del self._avatars[avatar_id]
            if self._current_avatar_id == avatar_id:
                self._current_avatar_id = list(self._avatars.keys())[0] if self._avatars else None
            return True
        return False

    def load_from_config(self, config_path: str):
        """从配置文件加载 avatars"""
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"[AvatarManager] 配置文件不存在: {config_path}")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for avatar_data in data.get("avatars", []):
            avatar = AvatarConfig.from_dict(avatar_data)
            self._avatars[avatar.avatar_id] = avatar

        print(f"[AvatarManager] 从配置加载了 {len(data.get('avatars', []))} 个数字人")

    def save_to_config(self, config_path: str):
        """保存 avatars 到配置文件"""
        data = {
            "avatars": [avatar.to_dict() for avatar in self._avatars.values()]
        }

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[AvatarManager] 配置已保存到: {config_path}")


# 全局 Avatar 管理器实例
_avatar_manager: Optional[AvatarManager] = None


def get_avatar_manager() -> AvatarManager:
    """获取全局 Avatar 管理器"""
    global _avatar_manager
    if _avatar_manager is None:
        _avatar_manager = AvatarManager()
    return _avatar_manager


def set_avatar_manager(manager: AvatarManager):
    """设置全局 Avatar 管理器"""
    global _avatar_manager
    _avatar_manager = manager

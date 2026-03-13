# -*- coding: utf-8 -*-
"""
Digital Human 配置模块

提供统一配置管理，所有配置项都从此模块读取
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 默认路径配置
DEFAULT_PATHS = {
    "human_choice": str(PROJECT_ROOT / "Human_Choice"),
    "sensevoice": str(PROJECT_ROOT / "SenseVoice"),
    "faster_liveportrait": str(PROJECT_ROOT / "FasterLivePortrait"),
    "psychology_rag": str(PROJECT_ROOT / "Psychology_Rag"),
    "live_talking_server": str(PROJECT_ROOT / "live_talking_server"),
    "output": str(PROJECT_ROOT / "output"),
}


def _load_avatar_config() -> Dict[str, Any]:
    """加载数字人配置文件"""
    config_path = Path(DEFAULT_PATHS["human_choice"]) / "avatar_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"avatars": {}, "default_avatar": "human_1", "settings": {}}


# 加载数字人配置
_AVATAR_CONFIG = _load_avatar_config()


@dataclass
class GPTSoVITSConfig:
    """GPT-SoVITS TTS 配置"""
    server_url: str = "http://127.0.0.1:9880"
    ref_audio: str = "/root/autodl-tmp/qys.wav"
    ref_text: str = "清晨推开窗，就能闻到风里带着的青草香，楼下的花园里开着各色的花儿。"
    language: str = "zh"
    # 默认使用 GPT-SoVITS
    default_tts: str = "gpt-sovits"

    def __post_init__(self):
        # 环境变量覆盖
        env_url = os.getenv("GPT_SOVITS_SERVER_URL")
        if env_url:
            self.server_url = env_url

        env_ref_audio = os.getenv("GPT_SOVITS_REF_AUDIO")
        if env_ref_audio:
            self.ref_audio = env_ref_audio

        env_ref_text = os.getenv("GPT_SOVITS_REF_TEXT")
        if env_ref_text:
            self.ref_text = env_ref_text


@dataclass
class SenseVoiceConfig:
    """SenseVoice ASR 配置"""
    model_name: str = "iic/SenseVoiceSmall"
    model_path: str = "/root/autodl-tmp/SenseVoice/iic/SenseVoiceSmall/model.pt"
    language: str = "auto"
    use_gpu: bool = True

    def __post_init__(self):
        env_model_path = os.getenv("SENSEVOICE_MODEL_PATH")
        if env_model_path:
            self.model_path = env_model_path
        elif not self.model_path:
            self.model_path = str(Path(DEFAULT_PATHS["sensevoice"]) / "models" / "iic_SenseVoiceSmall")


@dataclass
class DashScopeConfig:
    """阿里云 DashScope 配置"""
    api_key: str = "sk-5fcb24ad41b54421bb5ac93feea21cf6"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen-plus"
    max_tokens: int = 500
    temperature: float = 0.7

    def __post_init__(self):
        env_key = os.getenv("DASHSCOPE_API_KEY")
        if env_key:
            self.api_key = env_key


@dataclass
class AvatarInfo:
    """单个数字人信息"""
    id: str
    name: str
    source_image: str
    idle_video: Optional[str] = None
    description: str = ""

    def get_full_source_path(self) -> str:
        """获取源图像完整路径"""
        return str(Path(DEFAULT_PATHS["human_choice"]) / self.source_image)

    def get_full_idle_video_path(self) -> Optional[str]:
        """获取待机视频完整路径"""
        if not self.idle_video:
            return None
        path = Path(DEFAULT_PATHS["human_choice"]) / self.idle_video
        return str(path) if path.exists() else None

    def get_idle_video_url(self) -> Optional[str]:
        """获取待机视频URL"""
        if not self.idle_video:
            return None
        return f"/human_choice/{self.idle_video}"

    def get_source_image_url(self) -> str:
        """获取源图像URL"""
        return f"/human_choice/{self.source_image}"


@dataclass
class AvatarConfig:
    """数字人形象配置管理"""
    default_avatar: str = field(default_factory=lambda: _AVATAR_CONFIG.get("default_avatar", "human_1"))
    _avatars: Dict[str, AvatarInfo] = field(default_factory=dict)
    _settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 从配置文件加载
        self._settings = _AVATAR_CONFIG.get("settings", {})
        self._load_avatars()

    def _load_avatars(self):
        """加载所有数字人配置"""
        avatars_data = _AVATAR_CONFIG.get("avatars", {})
        for avatar_id, info in avatars_data.items():
            self._avatars[avatar_id] = AvatarInfo(
                id=avatar_id,
                name=info.get("name", avatar_id),
                source_image=info.get("source_image", ""),
                idle_video=info.get("idle_video"),
                description=info.get("description", "")
            )

    def get_avatar(self, avatar_id: str) -> Optional[AvatarInfo]:
        """获取指定数字人信息"""
        return self._avatars.get(avatar_id)

    def get_default_avatar(self) -> AvatarInfo:
        """获取默认数字人"""
        return self._avatars.get(self.default_avatar, list(self._avatars.values())[0] if self._avatars else None)

    def list_avatars(self) -> list:
        """列出所有数字人"""
        return list(self._avatars.values())

    def get_setting(self, key: str, default=None):
        """获取设置"""
        return self._settings.get(key, default)


@dataclass
class VideoConfig:
    """视频生成配置"""
    width: int = 450
    height: int = 450
    fps: int = 25
    batch_size: int = 16
    output_format: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"

    def __post_init__(self):
        # 从 avatar_config.json 的 settings 覆盖
        self.fps = _AVATAR_CONFIG.get("settings", {}).get("frame_rate", self.fps)
        self.batch_size = _AVATAR_CONFIG.get("settings", {}).get("batch_size", self.batch_size)


@dataclass
class AudioConfig:
    """音频配置"""
    sample_rate: int = 16000
    chunk_duration_ms: int = 20  # 每个音频块的时长（毫秒）

    def __post_init__(self):
        self.sample_rate = _AVATAR_CONFIG.get("settings", {}).get("audio_sample_rate", self.sample_rate)


@dataclass
class DigitalHumanConfig:
    """数字人系统主配置"""
    gpt_sovits: GPTSoVITSConfig = field(default_factory=GPTSoVITSConfig)
    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)
    dashscope: DashScopeConfig = field(default_factory=DashScopeConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8010

    # 输出目录
    output_dir: str = field(default_factory=lambda: DEFAULT_PATHS["output"])

    def __post_init__(self):
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def get_default_config() -> DigitalHumanConfig:
    """获取默认配置实例（单例模式）"""
    global _config_instance
    if '_config_instance' not in globals():
        _config_instance = DigitalHumanConfig()
    return _config_instance


def reload_config():
    """重新加载配置"""
    global _config_instance, _AVATAR_CONFIG
    _AVATAR_CONFIG = _load_avatar_config()
    if '_config_instance' in globals():
        del _config_instance
    return get_default_config()


# 兼容旧代码的别名
DEFAULT_PATHS = DEFAULT_PATHS


# 便捷导入
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
]

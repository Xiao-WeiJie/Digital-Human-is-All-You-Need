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

# 导入情感配置
from .emotion_config import EmotionExpressionConfig, get_emotion_config

# 默认路径配置
DEFAULT_PATHS = {
    "human_choice": str(PROJECT_ROOT / "Human_Choice"),
    "sensevoice": str(PROJECT_ROOT / "SenseVoice"),
    "faster_liveportrait": str(PROJECT_ROOT / "FasterLivePortrait"),
    "psychology_rag": str(PROJECT_ROOT / "Psychology_Rag"),
    "live_talking_server": str(PROJECT_ROOT / "live_talking_server"),
    "output": str(PROJECT_ROOT / "output"),
}

AVATAR_NAME_OVERRIDES = {
    "human_1": "小暖",
    "human_2": "小安",
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
class EmotionDetectionConfig:
    """情绪识别配置"""
    enabled: bool = True
    detector_backend: str = "opencv"  # opencv/mediapipe/retinaface
    enforce_detection: bool = False   # 允许无人脸时不报错
    frame_sample_interval: int = 3    # 每 N 帧采样一次
    smoothing_window: int = 5         # 时空平滑窗口大小
    analysis_timeout: float = 2.0     # 单帧分析超时(秒)
    sad_score_warning: float = 60.0   # 悲伤分数预警阈值
    sad_score_critical: float = 80.0  # 悲伤分数严重阈值
    context_ttl: int = 30             # 情绪上下文有效期(秒)

    def __post_init__(self):
        env_enabled = os.getenv("EMOTION_DETECTION_ENABLED")
        if env_enabled:
            self.enabled = env_enabled.lower() in ("true", "1", "yes")


@dataclass
class EmotionFusionConfig:
    """情绪融合配置"""
    enabled: bool = True
    text_weight: float = 0.7          # 文本情绪权重
    facial_weight: float = 0.3        # 面部情绪权重

    def __post_init__(self):
        env_enabled = os.getenv("EMOTION_FUSION_ENABLED")
        if env_enabled:
            self.enabled = env_enabled.lower() in ("true", "1", "yes")

        env_text_weight = os.getenv("EMOTION_FUSION_TEXT_WEIGHT")
        if env_text_weight:
            try:
                self.text_weight = float(env_text_weight)
                self.facial_weight = 1.0 - self.text_weight
            except ValueError:
                pass


@dataclass
class FirstEmotionReactionConfig:
    """首次表情捕获后的固定反馈配置"""
    enabled: bool = True
    delay_min_ms: int = 2000
    delay_max_ms: int = 3000
    happy_reply: str = "你今天看起来心情不错哦，是有什么开心的事吗？你愿意和我分享一下喜悦吗？"
    sad_reply: str = "你今天面色看起来不太好，是有什么心事吗？你愿意跟我聊聊发生了什么吗？有时候把心里的想法说出来，会让自己感觉轻松一些。"

    def __post_init__(self):
        env_enabled = os.getenv("FIRST_EMOTION_REACTION_ENABLED")
        if env_enabled:
            self.enabled = env_enabled.lower() in ("true", "1", "yes")

        env_min_delay = os.getenv("FIRST_EMOTION_REACTION_DELAY_MIN_MS")
        if env_min_delay:
            try:
                self.delay_min_ms = int(env_min_delay)
            except ValueError:
                pass

        env_max_delay = os.getenv("FIRST_EMOTION_REACTION_DELAY_MAX_MS")
        if env_max_delay:
            try:
                self.delay_max_ms = int(env_max_delay)
            except ValueError:
                pass


@dataclass
class GPTSoVITSConfig:
    """GPT-SoVITS TTS 配置"""
    server_url: str = "http://127.0.0.1:9880"
    language: str = "zh"
    default_tts: str = "gpt-sovits"

    # 全局回退参考音频（当数字人没有配置时使用）
    ref_audio: str = "/root/autodl-tmp/qys.wav"
    ref_text: str = "清晨推开窗，就能闻到风里带着的青草香，楼下的花园里开着各色的花儿。"

    # TTS 采样参数（影响情感丰富度）
    temperature: float = 1.1      # 1.0-1.5，越大越随机/丰富
    top_k: int = 10               # 5-15
    top_p: float = 0.95           # 0.9-1.0
    speed_factor: float = 1.0     # 语速

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
    model_path: str = ""  # 使用 model_name 自动下载，或设置绝对路径
    language: str = "auto"
    use_gpu: bool = True

    def __post_init__(self):
        env_model_path = os.getenv("SENSEVOICE_MODEL_PATH")
        if env_model_path:
            self.model_path = env_model_path
        elif not self.model_path:
            # 默认使用项目根目录下的 SenseVoice
            self.model_path = str(Path(DEFAULT_PATHS["sensevoice"]) / "iic" / "SenseVoiceSmall")


@dataclass
class DashScopeConfig:
    """阿里云 DashScope 配置"""
    api_key: str = "sk-5fcb24ad41b54421bb5ac93feea21cf6"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen-turbo"
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
    listening_video: Optional[str] = None
    description: str = ""
    # TTS 参考音频配置
    tts_config: Dict[str, Any] = field(default_factory=dict)
    # 首次情绪捕获固定反馈视频配置
    emotion_reaction_videos: Dict[str, str] = field(default_factory=dict)

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

    def get_listening_video_url(self) -> Optional[str]:
        """获取聆听视频URL"""
        if not self.listening_video:
            return None
        return f"/human_choice/{self.listening_video}"

    def get_source_image_url(self) -> str:
        """获取源图像URL"""
        return f"/human_choice/{self.source_image}"

    def get_emotion_reaction_video_url(self, emotion: str) -> Optional[str]:
        """获取首次情绪捕获固定反馈视频 URL"""
        video_file = self.emotion_reaction_videos.get(emotion)
        if not video_file:
            return None
        return f"/human_choice/{video_file}"

    def get_tts_ref(self, emotion: str = "default") -> tuple:
        """
        获取 TTS 参考音频路径和文本

        Args:
            emotion: 情感类型 (default/happy/sad/calm/question)

        Returns:
            (ref_audio_path, ref_text)
        """
        if not self.tts_config:
            # 没有配置，返回空
            return "", ""

        ref_audio_dir = self.tts_config.get("ref_audio_dir", "")

        # 尝试获取指定情感的参考音频
        emotion_refs = self.tts_config.get("emotion_refs", {})
        if emotion in emotion_refs:
            audio_file, text = emotion_refs[emotion]
            return f"{ref_audio_dir}/{audio_file}", text

        # 回退到默认
        default_ref = self.tts_config.get("default_ref", {})
        if default_ref:
            audio = default_ref.get("audio", "")
            text = default_ref.get("text", "")
            return f"{ref_audio_dir}/{audio}", text

        return "", ""

    def get_tts_ref_dir(self) -> str:
        """获取参考音频目录"""
        return self.tts_config.get("ref_audio_dir", "")


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
                name=AVATAR_NAME_OVERRIDES.get(avatar_id, info.get("name", avatar_id)),
                source_image=info.get("source_image", ""),
                idle_video=info.get("idle_video"),
                listening_video=info.get("listening_video"),
                description=info.get("description", ""),
                tts_config=info.get("tts", {}),
                emotion_reaction_videos=info.get("emotion_reaction_videos", {})
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
    motion_seed: Optional[int] = 7  # JoyVASA 运动生成随机种子，None 表示不固定

    def __post_init__(self):
        # 从 avatar_config.json 的 settings 覆盖
        self.fps = _AVATAR_CONFIG.get("settings", {}).get("frame_rate", self.fps)
        self.batch_size = _AVATAR_CONFIG.get("settings", {}).get("batch_size", self.batch_size)

        # 环境变量覆盖
        env_seed = os.getenv("MOTION_SEED")
        if env_seed:
            try:
                self.motion_seed = int(env_seed)
            except ValueError:
                pass


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

    # 情感表情配置（新增）
    emotion: EmotionExpressionConfig = field(default_factory=get_emotion_config)

    # 情绪识别配置（新增）
    emotion_detection: EmotionDetectionConfig = field(default_factory=EmotionDetectionConfig)

    # 情绪融合配置（新增）
    emotion_fusion: EmotionFusionConfig = field(default_factory=EmotionFusionConfig)

    # 首次表情捕获固定反馈配置
    first_emotion_reaction: FirstEmotionReactionConfig = field(default_factory=FirstEmotionReactionConfig)

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8010

    # 功能开关
    use_asr: bool = True  # 启用 ASR 语音识别
    save_generated_videos: bool = False  # 保存生成的视频（默认不保存）
    terminal_tts_mode: bool = False  # 启用终端直输 TTS 模式
    terminal_tts_avatar: str = "human_1"  # 终端直输模式默认数字人

    # 输出目录
    output_dir: str = field(default_factory=lambda: DEFAULT_PATHS["output"])

    def __post_init__(self):
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # 环境变量覆盖
        env_save = os.getenv("SAVE_GENERATED_VIDEOS")
        if env_save:
            self.save_generated_videos = env_save.lower() in ("true", "1", "yes")

        env_terminal_mode = os.getenv("TERMINAL_TTS_MODE")
        if env_terminal_mode:
            self.terminal_tts_mode = env_terminal_mode.lower() in ("true", "1", "yes")

        env_terminal_avatar = os.getenv("TERMINAL_TTS_AVATAR")
        if env_terminal_avatar:
            self.terminal_tts_avatar = env_terminal_avatar


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
    # 新增情感配置
    "EmotionExpressionConfig",
    "get_emotion_config",
    # 情绪识别配置
    "EmotionDetectionConfig",
    # 情绪融合配置
    "EmotionFusionConfig",
    # 首次表情捕获固定反馈配置
    "FirstEmotionReactionConfig",
]

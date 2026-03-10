# -*- coding: utf-8 -*-
"""
Digital Human Phase 1 - Configuration

统一配置管理：
- 服务器配置
- SenseVoice ASR 配置
- RAG/LLM 配置
- GPT-SoVITS TTS API 配置
- FasterLivePortrait 配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class SenseVoiceConfig:
    """SenseVoice ASR 配置"""
    model_dir: str = "SenseVoice/iic/SenseVoiceSmall"
    device: str = "cuda:0"
    language: str = "auto"  # "zh", "en", "yue", "ja", "ko", "auto"
    use_itn: bool = True
    # VAD 配置（用于长音频）
    use_vad: bool = True
    vad_model: str = "fsmn-vad"
    max_single_segment_time: int = 30000  # ms


@dataclass
class RAGConfig:
    """RAG/LLM 配置"""
    # 阿里云 DashScope API (Qwen)
    dashscope_api_key: Optional[str] = "sk-5fcb24ad41b54421bb5ac93feea21cf6"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model_name: str = "qwen-plus"
    embedding_model_name: str = "text-embedding-v3"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048


@dataclass
class GPTSoVITSConfig:
    """GPT-SoVITS TTS API 配置"""
    # GPT-SoVITS 服务地址
    api_url: str = "http://127.0.0.1:9880"
    # 请求超时（秒）
    timeout: float = 60.0
    # 文本语言
    text_lang: str = "zh"
    # 参考音频（用于声音克隆，服务器上的路径）
    ref_audio_path: Optional[str] = "/root/autodl-tmp/qys.wav"
    # 参考音频对应的文本
    prompt_text: str = "清晨推开窗，就能闻到风里带着的青草香，楼下的花园里开着各色的花儿。"
    # 参考文本语言
    prompt_lang: str = "zh"
    # 采样参数
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0
    # 文本分割方法: "cut0"(不分割), "cut1"(四字), "cut2"(五字), "cut3"(英文), "cut4"(日文), "cut5"(自动)
    text_split_method: str = "cut5"
    batch_size: int = 1
    # 语速
    speed_factor: float = 1.0
    splitting: bool = True
    # 输出采样率
    output_sample_rate: int = 24000
    # 是否启用 Kokoro 作为备选（设为 False 可禁用 Kokoro 检查）
    enable_kokoro_fallback: bool = False

    @classmethod
    def from_env(cls) -> "GPTSoVITSConfig":
        """从环境变量加载配置"""
        return cls(
            api_url=os.environ.get("GPT_SOVITS_API_URL", "http://127.0.0.1:9880"),
            ref_audio_path=os.environ.get("GPT_SOVITS_REF_AUDIO"),
            prompt_text=os.environ.get("GPT_SOVITS_PROMPT_TEXT", ""),
            prompt_lang=os.environ.get("GPT_SOVITS_PROMPT_LANG", "zh"),
        )


@dataclass
class FasterLivePortraitConfig:
    """
    FasterLivePortrait 配置

    参数说明（标注★的参数对数字人形态影响较大）:
    ============================================================================

    【基础配置】
    config_path: FasterLivePortrait 配置文件路径
    default_source_image: 默认数字人源图像
    output_dir: 输出目录

    【JoyVASA 模型路径】
    joyvasa_motion_model: 运动生成模型
    joyvasa_audio_model: 音频编码模型
    joyvasa_motion_template: 运动模板（包含 mean_exp, std_exp 等统计量）

    ============================================================================
    【★ 核心动画参数 - 对数字人形态影响最大】
    ============================================================================

    driving_multiplier: float = 1.3
        整体运动幅度乘数
        - 公式: x_d_new = x_s + (x_d - x_s) * driving_multiplier
        - 1.0 = 原始幅度
        - >1.0 = 放大运动（解决"内收/抿嘴"问题）
        - <1.0 = 缩小运动
        - 推荐值: 1.2~1.5
        - 影响: 整体面部表情幅度，包括嘴唇开合、头部运动等

    cfg_scale: float = 1.2
        JoyVASA 运动生成的 CFG 强度
        - 控制音频到运动转换的强度
        - 1.0~1.5 = 适中
        - >2.0 = 运动更夸张
        - 影响: 面部表情的丰富程度和强度

    lip_scale: float = 1.2
        唇部开合放大系数（后处理）
        - 1.0 = 原始
        - >1.0 = 放大嘴唇开合幅度
        - 影响: 单独控制嘴唇的张开程度

    ============================================================================
    【★ 唇部相关参数】
    ============================================================================

    flag_normalize_lip: bool = False
        是否将唇部状态归一化到闭合状态
        - True: 动画开始时强制嘴唇闭合（可能导致"抿嘴/内收"问题）
        - False: 保持原始唇部状态
        - ★ 重要: 如遇嘴唇内收问题，设为 False

    lip_normalize_threshold: float = 0.1

    # 唇部后处理增强参数（用于抑制抿嘴/吸嘴）
    # 基准帧数量：用前几帧均值代替单首帧，减少闭嘴首帧偏置
    lip_baseline_frames: int = 8
    # 只有开口超过该阈值时才放大，避免把闭口/抿嘴也一起放大
    lip_open_threshold: float = 0.002
    # 发声帧的最小开口保底，解决元音张口不足
    lip_min_open_delta: float = 0.0045
    # 下巴跟随比例
    lip_jaw_follow: float = 0.6
    # 对嘴唇 Z 轴前凸做软限制，避免吸嘴/鼓嘴
    lip_z_soft_limit: float = 0.025
    # 音频能量阈值下限，用于构建 speaking mask
    lip_audio_silence_threshold: float = 0.003
        唇部归一化阈值（仅当 flag_normalize_lip=True 时生效）
        - 控制唇部归一化的敏感度
        - 越小越敏感

    ============================================================================
    【拼接/融合参数】
    ============================================================================

    flag_stitching: bool = True
        是否启用拼接（stitching）
        - True: 使用 stitching 模型平滑动画边界
        - False: 直接合成，可能有边界痕迹
        - 影响: 动画与背景的融合质量

    flag_pasteback: bool = True
        是否将动画贴回原图
        - True: 输出完整图像（包含背景）
        - False: 只输出裁剪的人脸区域
        - 影响: 输出视频的画面范围

    ============================================================================
    【运动模式参数】
    ============================================================================

    flag_relative_motion: bool = True
        是否使用相对运动模式
        - True: 驱动信号作为相对位移（推荐）
        - False: 驱动信号作为绝对位置
        - 影响: 运动的计算方式

    flag_do_crop: bool = True
        是否裁剪源图像
        - True: 自动检测人脸并裁剪
        - False: 使用原始图像
        - 影响: 输入图像的处理方式

    flag_do_rot: bool = True
        是否旋转校正人脸
        - True: 根据人脸姿态进行旋转校正
        - False: 不校正
        - 影响: 人脸姿态的对齐

    ============================================================================
    【眼睛重定向参数】
    ============================================================================

    flag_eye_retargeting: bool = False
        是否启用眼睛重定向
        - True: 眼睛跟随驱动信号
        - False: 眼睛保持原状
        - 影响: 眼睛的注视方向

    flag_lip_retargeting: bool = True
        是否启用嘴唇重定向
        - True: 嘴唇完全跟随驱动信号
        - False: 使用混合模式
        - 影响: 嘴唇同步的精确度

    flag_source_video_eye_retargeting: bool = False
        源视频眼睛重定向开关

    source_video_eye_retargeting_threshold: float = 0.18
        眼睛重定向阈值

    ============================================================================
    【平滑/稳定参数】
    ============================================================================

    driving_smooth_observation_variance: float = 1e-7
        运动平滑强度
        - 值越大，动画越平滑，但会损失运动精度
        - 值越小，运动越精确，但可能有抖动
        - 推荐值: 1e-7 ~ 1e-5
        - 影响: 动画的流畅度

    ============================================================================
    【裁剪参数 - 影响人脸在画面中的位置和大小】
    ============================================================================

    src_dsize: int = 512
        输出图像尺寸（正方形）
        - 影响: 输出分辨率

    src_scale: float = 2.3
        源图像裁剪缩放比例
        - 值越大，人脸在画面中占比越小（显示更多背景）
        - 值越小，人脸在画面中占比越大（特写）
        - 影响: 人脸在画面中的大小

    src_vx_ratio: float = 0.0
        源图像水平偏移比例
        - 正值向右偏移
        - 负值向左偏移
        - 影响: 人脸在画面中的水平位置

    src_vy_ratio: float = -0.125
        源图像垂直偏移比例
        - 正值向下偏移
        - 负值向上偏移
        - 影响: 人脸在画面中的垂直位置（通常向上偏移以显示更多身体）

    dri_scale: float = 2.2
        驱动信号裁剪缩放比例
        - 用于驱动视频/音频的人脸裁剪

    dri_vx_ratio: float = 0.0
        驱动信号水平偏移比例

    dri_vy_ratio: float = -0.1
        驱动信号垂直偏移比例

    ============================================================================
    【高级参数】
    ============================================================================

    animation_region: str = "all"
        动画区域
        - "all": 整个人脸
        - "lip": 仅嘴唇
        - "eye": 仅眼睛
        - 影响: 哪些区域会被动画化

    flag_crop_driving_video: bool = False
        是否裁剪驱动视频

    flag_video_editing_head_rotation: bool = False
        是否启用头部旋转编辑

    anchor_frame: int = 0
        锚点帧（用于视频驱动）

    source_max_dim: int = 1280
        源图像最大尺寸

    source_division: int = 2
        源图像尺寸必须能被此数整除

    cfg_mode: str = "incremental"
        JoyVASA CFG 模式
        - "incremental": 增量式
        - 影响: 运动生成的方式

    """

    # ============================================================
    # 基础配置
    # ============================================================
    config_path: str = "/root/autodl-tmp/FasterLivePortrait/configs/trt_infer.yaml"
    default_source_image: Optional[str] = "/root/autodl-tmp/digital_human/Human_Choice/Human_1/qys.png"
    output_dir: str = "/root/autodl-tmp/results/out_digital_human"

    # ============================================================
    # JoyVASA 模型路径
    # ============================================================
    joyvasa_motion_model: str = "/root/autodl-tmp/FasterLivePortrait/checkpoints/JoyVASA/joyvasa_motion_model.pth"
    joyvasa_audio_model: str = "/root/autodl-tmp/FasterLivePortrait/checkpoints/wav2vec2-base-960h"
    joyvasa_motion_template: str = "/root/autodl-tmp/FasterLivePortrait/checkpoints/motion_template.pkl"

    # ============================================================
    # ★ 核心动画参数 - 对数字人形态影响最大
    # ============================================================
    # 整体运动幅度乘数:
    # - 1.0 = 原始幅度 (FasterLivePortrait 默认值)
    # - >1.0 = 放大运动
    # - 推荐 1.2~1.4，配合 _apply_lip_scale 中的 Z 轴限制使用
    driving_multiplier: float = 1.18
    # JoyVASA CFG 强度: 控制运动生成的强度，提高可增强口型变化
    cfg_scale: float = 1.25
    # 唇部开合放大系数（后处理）:
    # - 1.0 = 不做额外处理
    # - >1.0 = 额外放大嘴部开合（推荐 1.3~1.5）
    lip_scale: float = 1.50

    # ============================================================
    # ★ 唇部相关参数
    # ============================================================
    # 是否将唇部归一化到闭合状态
    # - True: 动画开始时强制嘴唇闭合（可能导致开合幅度受限）
    # - False: 保持源图像原始唇部状态（推荐，配合 lip_scale 使用）
    flag_normalize_lip: bool = False
    # 唇部归一化阈值
    lip_normalize_threshold: float = 0.1

    # 唇部后处理增强参数（用于抑制抿嘴/吸嘴）
    # 基准帧数量：用前几帧均值代替单首帧，减少闭嘴首帧偏置
    lip_baseline_frames: int = 16
    # 只有开口超过该阈值时才放大，避免把闭口/抿嘴也一起放大
    lip_open_threshold: float = 0.0012
    # 发声帧的最小开口保底，解决元音张口不足
    lip_min_open_delta: float = 0.013
    # 下巴跟随比例
    lip_jaw_follow: float = 0.92
    # 对嘴唇 Z 轴前凸做软限制，避免吸嘴/鼓嘴
    lip_z_soft_limit: float = 0.007
    # 音频能量阈值下限，用于构建 speaking mask
    lip_audio_silence_threshold: float = 0.005
    lip_anti_purse_scale = 0.20
    lip_width_expand_scale = 0.10
    lip_width_expand_cap = 0.0020
    # 静音帧上下唇最小 Y 间距（防止唇线消失/抿嘴）
    lip_idle_min_gap: float = 0.0075
    # 全局对称修正阈值（非聚唇帧也检查嘴斜）
    lip_global_asym_threshold: float = 0.06
    # pitch 平滑窗口大小（防止视线突变）
    pitch_smooth_window: int = 11
    # pitch 最大单帧变化量（度）
    pitch_max_delta: float = 1.8
    # ============================================================
    # 拼接/融合参数
    # ============================================================
    # 是否启用拼接（平滑动画边界）
    flag_stitching: bool = True
    # 是否将动画贴回原图
    flag_pasteback: bool = True

    # ============================================================
    # 运动模式参数
    # ============================================================
    # 是否使用相对运动模式
    flag_relative_motion: bool = True
    # 是否裁剪源图像
    flag_do_crop: bool = True
    # 是否旋转校正人脸
    flag_do_rot: bool = True

    # ============================================================
    # 眼睛/嘴唇重定向参数
    # ============================================================
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_source_video_eye_retargeting: bool = False
    source_video_eye_retargeting_threshold: float = 0.18

    # ============================================================
    # 平滑/稳定参数
    # ============================================================
    # 运动平滑强度 (值越大越平滑，但损失精度)
    driving_smooth_observation_variance: float = 3e-7

    # ============================================================
    # 裁剪参数 - 影响人脸在画面中的位置和大小
    # ============================================================
    # 输出图像尺寸
    src_dsize: int = 512
    # 源图像裁剪缩放比例 (越大人脸占比越小)
    src_scale: float = 2.3
    # 源图像水平偏移 (正值向右)
    src_vx_ratio: float = 0.0
    # 源图像垂直偏移 (负值向上，通常用于显示更多身体)
    src_vy_ratio: float = -0.125
    # 驱动信号裁剪参数
    dri_scale: float = 2.2
    dri_vx_ratio: float = 0.0
    dri_vy_ratio: float = -0.1

    # ============================================================
    # 高级参数
    # ============================================================
    # 动画区域: "all", "lip", "eye"
    animation_region: str = "all"
    flag_crop_driving_video: bool = False
    flag_video_editing_head_rotation: bool = False
    anchor_frame: int = 0
    source_max_dim: int = 1280
    source_division: int = 2
    cfg_mode: str = "incremental"


@dataclass
class ServerConfig:
    """服务器配置"""
    # 服务器地址和端口
    host: str = "0.0.0.0"
    port: int = 8000
    # 开发模式（自动重载）
    reload: bool = False
    # 跳过环境检查
    skip_check: bool = False

    # 路径配置
    @property
    def avatar_dir(self) -> Path:
        """Avatar 资源目录"""
        return PROJECT_ROOT / "digital_human" / "Human_Choice"

    @property
    def frontend_dir(self) -> Path:
        """前端目录"""
        return PROJECT_ROOT / "frontend"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """从环境变量加载配置"""
        return cls(
            host=os.environ.get("DIGITAL_HUMAN_HOST", "0.0.0.0"),
            port=int(os.environ.get("DIGITAL_HUMAN_PORT", "8000")),
            reload=os.environ.get("DIGITAL_HUMAN_RELOAD", "").lower() in ("true", "1", "yes"),
        )


@dataclass
class DigitalHumanConfig:
    """数字人系统总配置"""
    server: ServerConfig = field(default_factory=ServerConfig)
    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    gpt_sovits: GPTSoVITSConfig = field(default_factory=GPTSoVITSConfig)
    faster_live_portrait: FasterLivePortraitConfig = field(default_factory=FasterLivePortraitConfig)

    # 会话配置
    session_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DigitalHumanConfig":
        """从环境变量加载配置"""
        config = cls()

        # Server
        config.server = ServerConfig.from_env()

        # RAG API Key
        if dashscope_key := os.environ.get("DASHSCOPE_API_KEY"):
            config.rag.dashscope_api_key = dashscope_key

        # GPT-SoVITS
        if api_url := os.environ.get("GPT_SOVITS_API_URL"):
            config.gpt_sovits.api_url = api_url

        # SenseVoice
        if device := os.environ.get("SENSEVOICE_DEVICE"):
            config.sensevoice.device = device

        return config

    def validate(self) -> bool:
        """验证配置"""
        errors = []

        if not self.rag.dashscope_api_key:
            errors.append("RAG: DASHSCOPE_API_KEY is required")

        if errors:
            print("Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            return False
        return True


# 全局配置实例
_config: Optional[DigitalHumanConfig] = None


def get_config() -> DigitalHumanConfig:
    """获取全局配置"""
    global _config
    if _config is None:
        _config = DigitalHumanConfig.from_env()
    return _config


def set_config(config: DigitalHumanConfig):
    """设置全局配置"""
    global _config
    _config = config
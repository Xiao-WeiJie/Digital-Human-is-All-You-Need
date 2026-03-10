# -*- coding: utf-8 -*-
"""
Digital Human Phase 2 - Package Init

数字人第二阶段交互模块。

主要组件：
- SenseVoiceAdapter: 语音识别适配器
- RAGAdapter: RAG/LLM 适配器
- TTSAdapter: TTS 适配器（支持 GPT-SoVITS 和 Kokoro）
- LivePortraitAdapter: 数字人动画适配器
- AvatarManager: 数字人配置管理器
- IdleStateManager: 待机状态管理器

使用示例：
    from digital_human import get_config, get_avatar_manager
    from digital_human import SenseVoiceAdapter, RAGAdapter, TTSAdapter, LivePortraitAdapter

    # 加载配置
    config = get_config()

    # 使用各适配器
    asr = SenseVoiceAdapter(config.sensevoice)
    rag = RAGAdapter(config.rag)
    tts = TTSAdapter(config.gpt_sovits)
    lpp = LivePortraitAdapter(config.faster_live_portrait)

    # 获取 avatar 管理器
    avatar_mgr = get_avatar_manager()
    print(avatar_mgr.list_avatars())
"""

from .config import (
    DigitalHumanConfig,
    SenseVoiceConfig,
    RAGConfig,
    GPTSoVITSConfig,
    FasterLivePortraitConfig,
    get_config,
    set_config,
)
from .sensevoice_adapter import SenseVoiceAdapter
from .rag_adapter import RAGAdapter
from .tts_adapter import TTSAdapter
from .liveportrait_adapter import LivePortraitAdapter

# Phase 2 新模块
from .avatar_config import (
    AvatarConfig,
    AvatarManager,
    TTSProfile,
    IdleConfig,
    TalkingConfig,
    get_avatar_manager,
    set_avatar_manager,
)
from .idle_manager import (
    AvatarState,
    IdleStateManager,
    IdleMotionGenerator,
)
from .orchestrator import (
    DigitalHumanOrchestrator,
    PipelineResult,
    create_orchestrator,
)

__all__ = [
    # Config
    "DigitalHumanConfig",
    "SenseVoiceConfig",
    "RAGConfig",
    "GPTSoVITSConfig",
    "FasterLivePortraitConfig",
    "get_config",
    "set_config",
    # Adapters
    "SenseVoiceAdapter",
    "RAGAdapter",
    "TTSAdapter",
    "LivePortraitAdapter",
    # Phase 2 - Avatar
    "AvatarConfig",
    "AvatarManager",
    "TTSProfile",
    "IdleConfig",
    "TalkingConfig",
    "get_avatar_manager",
    "set_avatar_manager",
    # Phase 2 - Idle
    "AvatarState",
    "IdleStateManager",
    "IdleMotionGenerator",
    # Orchestrator
    "DigitalHumanOrchestrator",
    "PipelineResult",
    "create_orchestrator",
]

__version__ = "0.2.0"

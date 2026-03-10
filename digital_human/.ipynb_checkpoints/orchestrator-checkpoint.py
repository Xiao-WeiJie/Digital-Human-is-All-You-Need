# -*- coding: utf-8 -*-
"""
Digital Human Phase 2 - Orchestrator

主编排器：串联所有模块，实现完整的数字人交互闭环。

主链路：
  用户语音输入 → SenseVoice ASR → RAG/LLM → GPT-SoVITS TTS → FasterLivePortrait → 视频输出

调试链路：
  用户文本输入 → RAG/LLM → GPT-SoVITS TTS → FasterLivePortrait → 视频输出

Phase 2 新增：
  - 支持多 Avatar 切换
  - 支持独立 TTS Profile
  - 与 IdleStateManager 联动
"""

import asyncio
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .config import DigitalHumanConfig, get_config
from .sensevoice_adapter import SenseVoiceAdapter
from .rag_adapter import RAGAdapter
from .tts_adapter import TTSAdapter
from .liveportrait_adapter import LivePortraitAdapter


@dataclass
class PipelineResult:
    """处理结果"""
    success: bool
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    transcript: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None
    time_cost: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class DigitalHumanOrchestrator:
    """数字人编排器（Phase 2）"""

    def __init__(self, config: Optional[DigitalHumanConfig] = None):
        self.config = config or get_config()

        # 各模块适配器
        self.asr = SenseVoiceAdapter(self.config.sensevoice)
        self.rag = RAGAdapter(self.config.rag)
        self.tts = TTSAdapter(self.config.gpt_sovits)
        self.live_portrait = LivePortraitAdapter(self.config.faster_live_portrait)

        # 会话状态
        self.session_id = self.config.session_id or str(uuid.uuid4())

        # 源图像
        self._source_image = self.config.faster_live_portrait.default_source_image

        # 当前 Avatar（Phase 2）
        self._current_avatar_id: Optional[str] = None
        self._current_tts_profile: Optional[Any] = None

        # Idle Manager 引用（由外部设置）
        self.idle_manager = None

    def set_source_image(self, image_path: str):
        """设置数字人源图像"""
        self._source_image = image_path
        print(f"[Orchestrator] 设置源图像: {image_path}")

    def set_avatar(self, avatar) -> bool:
        """
        设置当前 Avatar

        Args:
            avatar: AvatarConfig 实例

        Returns:
            bool: 是否成功
        """
        if avatar is None:
            return False

        self._source_image = avatar.source_image
        self._current_avatar_id = avatar.avatar_id
        self._current_tts_profile = avatar.tts_profile

        print(f"[Orchestrator] 切换到 Avatar: {avatar.name} ({avatar.avatar_id})")
        return True

    def set_idle_manager(self, idle_manager):
        """设置 Idle Manager 引用"""
        self.idle_manager = idle_manager

    async def process_voice_input(
        self,
        audio_input,
        source_image: Optional[str] = None,
        session_id: Optional[str] = None,
        avatar_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        处理语音输入（完整主链路）

        Args:
            audio_input: 音频输入（文件路径或字节数据）
            source_image: 源图像路径（可选，覆盖默认值）
            session_id: 会话 ID（可选）
            avatar_id: Avatar ID（Phase 2，可选）

        Returns:
            PipelineResult: 处理结果
        """
        t0 = time.time()
        src_image = source_image or self._source_image

        # Phase 2: 支持 Avatar 切换
        if avatar_id:
            try:
                from .avatar_config import get_avatar_manager
                mgr = get_avatar_manager()
                avatar = mgr.get_avatar(avatar_id)
                if avatar:
                    src_image = avatar.source_image
                    self._current_tts_profile = avatar.tts_profile
            except Exception as e:
                print(f"[Orchestrator] Avatar 切换失败: {e}")

        if not src_image:
            return PipelineResult(
                success=False,
                error="未设置源图像，请先调用 set_source_image() 或配置 default_source_image",
                time_cost=time.time() - t0,
            )

        try:
            # 通知 Idle Manager 开始监听
            if self.idle_manager:
                self.idle_manager.start_listening()

            # Step 1: 语音识别 (ASR)
            print("\n" + "=" * 50)
            print("[Step 1/5] 语音识别 (SenseVoice)")
            print("=" * 50)

            asr_result = self.asr.transcribe(audio_input)
            transcript = asr_result["text"]

            if not transcript.strip():
                return PipelineResult(
                    success=False,
                    error="语音识别结果为空",
                    time_cost=time.time() - t0,
                    details={"asr_time": asr_result["time_cost"]},
                )

            # 通知 Idle Manager 开始思考
            if self.idle_manager:
                self.idle_manager.start_thinking()

            # Step 2: RAG 对话
            print("\n" + "=" * 50)
            print("[Step 2/5] RAG 对话 (PsyMind)")
            print("=" * 50)

            rag_result = await self.rag.chat(transcript, session_id or self.session_id)
            response = rag_result["response"]

            # Step 3: 语音合成 (TTS) - 使用 TTS Profile
            print("\n" + "=" * 50)
            print("[Step 3/5] 语音合成 (GPT-SoVITS/Kokoro)")
            print("=" * 50)

            audio_path = tempfile.mktemp(suffix=".wav")

            # Phase 2: 使用 Avatar 的 TTS Profile
            tts_kwargs = {}
            if self._current_tts_profile:
                if self._current_tts_profile.ref_audio_path:
                    tts_kwargs["ref_audio_path"] = self._current_tts_profile.ref_audio_path
                if self._current_tts_profile.prompt_text:
                    tts_kwargs["prompt_text"] = self._current_tts_profile.prompt_text

            audio_path = await self.tts.synthesize_async(
                response,
                audio_path,
                **tts_kwargs
            )

            # 通知 Idle Manager 开始说话
            if self.idle_manager:
                self.idle_manager.start_talking()

            # Step 4: 生成动画
            print("\n" + "=" * 50)
            print("[Step 4/5] 生成数字人动画 (FasterLivePortrait)")
            print("=" * 50)

            video_path = self.live_portrait.generate_from_audio(
                audio_path,
                src_image,
            )

            # Step 5: 完成
            total_time = time.time() - t0
            print("\n" + "=" * 50)
            print(f"[完成] 总耗时: {total_time:.2f}s")
            print("=" * 50)

            # 通知 Idle Manager 回到待机
            if self.idle_manager:
                self.idle_manager.stop_talking()

            return PipelineResult(
                success=True,
                video_path=video_path,
                audio_path=audio_path,
                transcript=transcript,
                response=response,
                time_cost=total_time,
                details={
                    "asr_time": asr_result["time_cost"],
                    "session_id": rag_result.get("session_id"),
                    "avatar_id": avatar_id or self._current_avatar_id,
                },
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

            # 通知 Idle Manager 错误状态
            if self.idle_manager:
                self.idle_manager.set_error(str(e))

            return PipelineResult(
                success=False,
                error=str(e),
                time_cost=time.time() - t0,
            )

    async def process_text_input(
        self,
        text: str,
        source_image: Optional[str] = None,
        session_id: Optional[str] = None,
        use_gptsovits: bool = True,
        avatar_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        处理文本输入（调试链路）

        Args:
            text: 用户输入文本
            source_image: 源图像路径
            session_id: 会话 ID
            use_gptsovits: 是否优先使用 GPT-SoVITS
            avatar_id: Avatar ID（Phase 2，可选）

        Returns:
            PipelineResult: 处理结果
        """
        t0 = time.time()
        src_image = source_image or self._source_image

        # Phase 2: 支持 Avatar 切换
        if avatar_id:
            try:
                from .avatar_config import get_avatar_manager
                mgr = get_avatar_manager()
                avatar = mgr.get_avatar(avatar_id)
                if avatar:
                    src_image = avatar.source_image
                    self._current_tts_profile = avatar.tts_profile
            except Exception as e:
                print(f"[Orchestrator] Avatar 切换失败: {e}")

        if not src_image:
            return PipelineResult(
                success=False,
                error="未设置源图像",
                time_cost=time.time() - t0,
            )

        try:
            # 通知 Idle Manager 开始思考
            if self.idle_manager:
                self.idle_manager.start_thinking()

            # Step 1: RAG 对话
            print("\n" + "=" * 50)
            print("[Step 1/3] RAG 对话 (PsyMind)")
            print("=" * 50)
            print(f"用户输入: {text}")

            rag_result = await self.rag.chat(text, session_id or self.session_id)
            response = rag_result["response"]
            print(f"AI 回复: {response}")

            # Step 2: 语音合成 - 使用 TTS Profile
            print("\n" + "=" * 50)
            print("[Step 2/3] 语音合成")
            print("=" * 50)

            audio_path = tempfile.mktemp(suffix=".wav")

            # Phase 2: 使用 Avatar 的 TTS Profile
            tts_kwargs = {"prefer_gptsovits": use_gptsovits}
            if self._current_tts_profile:
                if self._current_tts_profile.ref_audio_path:
                    tts_kwargs["ref_audio_path"] = self._current_tts_profile.ref_audio_path
                if self._current_tts_profile.prompt_text:
                    tts_kwargs["prompt_text"] = self._current_tts_profile.prompt_text

            audio_path = await self.tts.synthesize_async(
                response,
                audio_path,
                **tts_kwargs
            )

            # 通知 Idle Manager 开始说话
            if self.idle_manager:
                self.idle_manager.start_talking()

            # Step 3: 生成动画
            print("\n" + "=" * 50)
            print("[Step 3/3] 生成数字人动画")
            print("=" * 50)

            video_path = self.live_portrait.generate_from_audio(
                audio_path,
                src_image,
            )

            total_time = time.time() - t0
            print("\n" + "=" * 50)
            print(f"[完成] 总耗时: {total_time:.2f}s")
            print(f"视频输出: {video_path}")
            print("=" * 50)

            # 通知 Idle Manager 回到待机
            if self.idle_manager:
                self.idle_manager.stop_talking()

            return PipelineResult(
                success=True,
                video_path=video_path,
                audio_path=audio_path,
                transcript=text,
                response=response,
                time_cost=total_time,
                details={
                    "session_id": rag_result.get("session_id"),
                    "avatar_id": avatar_id or self._current_avatar_id,
                },
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

            # 通知 Idle Manager 错误状态
            if self.idle_manager:
                self.idle_manager.set_error(str(e))

            return PipelineResult(
                success=False,
                error=str(e),
                time_cost=time.time() - t0,
            )

    def process_text_sync(
        self,
        text: str,
        source_image: Optional[str] = None,
        **kwargs,
    ) -> PipelineResult:
        """同步版本的文本处理接口"""
        return asyncio.run(self.process_text_input(text, source_image, **kwargs))

    def process_voice_sync(
        self,
        audio_input,
        source_image: Optional[str] = None,
        **kwargs,
    ) -> PipelineResult:
        """同步版本的语音处理接口"""
        return asyncio.run(self.process_voice_input(audio_input, source_image, **kwargs))

    def reset_session(self) -> str:
        """重置会话"""
        self.session_id = str(uuid.uuid4())
        return self.rag.reset_session()


def create_orchestrator(
    source_image: Optional[str] = None,
    dashscope_api_key: Optional[str] = None,
    gpt_sovits_url: Optional[str] = None,
    **kwargs,
) -> DigitalHumanOrchestrator:
    """
    工厂函数：创建编排器实例

    Args:
        source_image: 数字人源图像
        dashscope_api_key: 阿里云 API Key
        gpt_sovits_url: GPT-SoVITS API 地址
        **kwargs: 其他配置参数

    Returns:
        DigitalHumanOrchestrator: 编排器实例
    """
    config = DigitalHumanConfig.from_env()

    # 覆盖参数
    if dashscope_api_key:
        config.rag.dashscope_api_key = dashscope_api_key
    if gpt_sovits_url:
        config.gpt_sovits.api_url = gpt_sovits_url
    if source_image:
        config.faster_live_portrait.default_source_image = source_image

    # 其他配置覆盖
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    orchestrator = DigitalHumanOrchestrator(config)

    if source_image:
        orchestrator.set_source_image(source_image)

    return orchestrator

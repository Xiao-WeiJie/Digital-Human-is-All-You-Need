# -*- coding: utf-8 -*-
"""
Digital Human Phase 1 - SenseVoice ASR Adapter

封装 SenseVoice-Small 语音识别功能。
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

# 添加 SenseVoice 目录到路径
SENSEVOICE_DIR = Path(__file__).parent.parent / "SenseVoice"
if str(SENSEVOICE_DIR) not in sys.path:
    sys.path.insert(0, str(SENSEVOICE_DIR))

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from .config import SenseVoiceConfig


class SenseVoiceAdapter:
    """SenseVoice ASR 适配器"""

    def __init__(self, config: Optional[SenseVoiceConfig] = None):
        self.config = config or SenseVoiceConfig()
        self.model = None
        self._initialized = False

    def initialize(self):
        """初始化模型（延迟加载）"""
        if self._initialized:
            return

        print(f"[SenseVoice] 正在初始化模型: {self.config.model_dir}")
        t0 = time.time()

        if self.config.use_vad:
            self.model = AutoModel(
                model=self.config.model_dir,
                trust_remote_code=True,
                remote_code=str(SENSEVOICE_DIR / "model.py"),
                vad_model=self.config.vad_model,
                vad_kwargs={"max_single_segment_time": self.config.max_single_segment_time},
                device=self.config.device,
            )
        else:
            self.model = AutoModel(
                model=self.config.model_dir,
                trust_remote_code=True,
                remote_code=str(SENSEVOICE_DIR / "model.py"),
                device=self.config.device,
            )

        print(f"[SenseVoice] 模型初始化完成，耗时 {time.time() - t0:.2f}s")
        self._initialized = True

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes],
        language: Optional[str] = None,
    ) -> dict:
        """
        语音识别

        Args:
            audio_input: 音频文件路径或音频字节数据
            language: 语言代码 (zh, en, yue, ja, ko, auto)

        Returns:
            dict: {
                "text": str,  # 识别文本
                "raw_text": str,  # 原始文本（包含情感/事件标签）
                "language": str,  # 检测到的语言
                "time_cost": float,  # 处理耗时
            }
        """
        if not self._initialized:
            self.initialize()

        t0 = time.time()
        lang = language or self.config.language

        # 处理字节输入
        if isinstance(audio_input, bytes):
            # 写入临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_input)
                audio_path = f.name
        else:
            audio_path = str(audio_input)

        try:
            # 调用模型
            if self.config.use_vad:
                result = self.model.generate(
                    input=audio_path,
                    cache={},
                    language=lang,
                    use_itn=self.config.use_itn,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )
            else:
                result = self.model.generate(
                    input=audio_path,
                    cache={},
                    language=lang,
                    use_itn=self.config.use_itn,
                    batch_size=64,
                )

            # 提取结果
            raw_text = result[0]["text"] if result else ""
            clean_text = rich_transcription_postprocess(raw_text)

            time_cost = time.time() - t0
            print(f"[SenseVoice] 识别完成，耗时 {time_cost:.2f}s")
            print(f"[SenseVoice] 识别结果: {clean_text}")

            return {
                "text": clean_text,
                "raw_text": raw_text,
                "language": lang,
                "time_cost": time_cost,
            }

        finally:
            # 清理临时文件
            if isinstance(audio_input, bytes) and os.path.exists(audio_path):
                os.unlink(audio_path)

    def transcribe_file(self, audio_path: Union[str, Path]) -> str:
        """
        简单接口：返回识别文本

        Args:
            audio_path: 音频文件路径

        Returns:
            str: 识别的文本
        """
        result = self.transcribe(audio_path)
        return result["text"]

    def __call__(self, audio_input: Union[str, Path, bytes]) -> str:
        """使适配器可调用"""
        return self.transcribe_file(audio_input) if isinstance(audio_input, (str, Path)) else self.transcribe(audio_input)["text"]

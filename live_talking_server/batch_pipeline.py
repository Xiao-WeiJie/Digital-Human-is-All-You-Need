# -*- coding: utf-8 -*-
"""
数字人批量视频生成流水线

默认使用 Kokoro TTS，支持 SenseVoice ASR
"""

import os
import sys
import time
import asyncio
import logging
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# 使用统一的 logger
try:
    from logger import logger
except ImportError:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(handler)

# 导入配置
from digital_human.config import get_default_config, DigitalHumanConfig

# 导入视频生成器
try:
    from batch_video_generator import BatchVideoGenerator, VideoGenerationResult, BatchVideoGeneratorSimple
except ImportError:
    try:
        from live_talking_server.batch_video_generator import BatchVideoGenerator, VideoGenerationResult, BatchVideoGeneratorSimple
    except ImportError:
        from .batch_video_generator import BatchVideoGenerator, VideoGenerationResult, BatchVideoGeneratorSimple


@dataclass
class PipelineResult:
    """流水线处理结果"""
    video_path: str
    video_url: str
    response_text: str
    user_text: str
    total_time: float
    metrics: dict
    audio_duration: float
    success: bool
    error_message: str = ""


class SenseVoiceASR:
    """SenseVoice 语音识别"""

    def __init__(self, config=None):
        self.config = config or get_default_config().sensevoice
        self.model = None

    def _init_model(self):
        if self.model is None:
            try:
                from funasr import AutoModel
                self.model = AutoModel(
                    model=self.config.model_name,
                    model_path=self.config.model_path if os.path.exists(self.config.model_path) else None
                )
                logger.info("[ASR] SenseVoice model loaded")
            except Exception as e:
                logger.error(f"[ASR] Failed to load model: {e}")

    def recognize(self, audio_path: str) -> str:
        """识别音频"""
        self._init_model()

        if self.model is None:
            logger.warning("[ASR] Model not available, returning empty text")
            return ""

        try:
            result = self.model.generate(
                input=audio_path,
                language=self.config.language
            )
            if result and len(result) > 0:
                text = result[0].get("text", "")
                logger.info(f"[ASR] Recognized: {text}")
                return text
        except Exception as e:
            logger.error(f"[ASR] Recognition error: {e}")

        return ""


class GPTSoVITSTTS:
    """GPT-SoVITS TTS 语音合成"""

    def __init__(self, config=None, avatar_id: str = "human_1"):
        self.config = config or get_default_config().gpt_sovits
        self.server_url = self.config.server_url
        self.ref_audio = self.config.ref_audio
        self.ref_text = self.config.ref_text
        self.avatar_id = avatar_id
        self._avatar_config = None

    def _get_ref_audio(self, emotion: str = "default") -> Tuple[str, str]:
        """
        根据数字人和情感获取参考音频

        Returns:
            (ref_audio_path, ref_text)
        """
        try:
            # 尝试从 avatar 配置获取
            if self._avatar_config is None:
                self._avatar_config = get_default_config().avatar

            avatar_info = self._avatar_config.get_avatar(self.avatar_id)
            if avatar_info and avatar_info.tts_config:
                ref_audio, ref_text = avatar_info.get_tts_ref(emotion)
                if ref_audio and ref_text:
                    logger.info(f"[GPT-SoVITS] Using avatar {self.avatar_id} emotion {emotion}: {ref_audio}")
                    return ref_audio, ref_text
        except Exception as e:
            logger.warning(f"[GPT-SoVITS] Failed to get avatar config: {e}")

        # 回退到全局配置
        return self.ref_audio, self.ref_text

    def set_avatar(self, avatar_id: str):
        """切换数字人"""
        self.avatar_id = avatar_id
        self._avatar_config = None  # 重置缓存，下次重新加载

    def generate(self, text: str, output_path: str, emotion: str = "default") -> Tuple[str, float]:
        """
        生成语音

        Args:
            text: 输入文本
            output_path: 输出文件路径
            emotion: 情感标签 (default/happy/sad/calm/question)

        Returns:
            Tuple[str, float]: (音频路径, 音频时长)
        """
        import requests
        import soundfile as sf
        import io
        import wave
        import struct

        # 根据情感获取参考音频
        ref_audio, ref_text = self._get_ref_audio(emotion)

        logger.info(f"[GPT-SoVITS] Starting generation for text: {text[:50]}...")
        logger.info(f"[GPT-SoVITS] Server URL: {self.server_url}")
        logger.info(f"[GPT-SoVITS] Avatar: {self.avatar_id}, Emotion: {emotion}")
        logger.info(f"[GPT-SoVITS] Ref audio: {ref_audio}")
        logger.info(f"[GPT-SoVITS] Ref text: {ref_text}")

        # 检查配置
        if not self.server_url:
            raise RuntimeError("GPT-SoVITS server URL is not configured!")
        if not ref_audio:
            raise RuntimeError("GPT-SoVITS ref_audio is not configured!")

        # 使用非流式模式直接生成 WAV 格式
        req = {
            'text': text,
            'text_lang': 'zh',
            'ref_audio_path': ref_audio,
            'prompt_text': ref_text,
            'prompt_lang': 'zh',
            'media_type': 'wav',  # 直接生成 WAV 格式
            'streaming_mode': False,  # 非流式模式，返回完整 WAV 文件
            'text_split_method': 'cut0',  # 不分割文本，避免分段导致的音频间隙
            'parallel_infer': False,  # 禁用并行推理，确保音频连续性
            'split_bucket': False,  # 禁用分桶处理
            # 情感丰富度参数
            'temperature': 1.1,
            'top_k': 10,
            'top_p': 0.95,
        }

        logger.info(f"[GPT-SoVITS] Sending request to {self.server_url}/tts")

        # 非流式请求，直接获取完整响应
        try:
            response = requests.post(
                f"{self.server_url}/tts",
                json=req,
                timeout=120
            )
            logger.info(f"[GPT-SoVITS] Response status: {response.status_code}")
        except Exception as e:
            logger.error(f"[GPT-SoVITS] Request failed: {e}")
            raise

        if response.status_code != 200:
            logger.error(f"[GPT-SoVITS] Error response: {response.text[:500]}")
            raise RuntimeError(f"GPT-SoVITS error: {response.text}")

        # 直接获取响应内容
        audio_data = response.content
        logger.info(f"[GPT-SoVITS] Received {len(audio_data)} bytes")

        if len(audio_data) == 0:
            raise RuntimeError("GPT-SoVITS returned empty audio")

        # 打印文件头用于调试
        header_hex = audio_data[:16].hex() if len(audio_data) >= 16 else audio_data.hex()
        logger.info(f"[GPT-SoVITS] Audio header: {header_hex}")

        # 检测格式
        header = audio_data[:4]
        if header == b'OggS':
            logger.info("[GPT-SoVITS] Detected OGG format")
        elif header == b'RIFF':
            logger.info("[GPT-SoVITS] Detected WAV format")
            # 修复 WAV 头中的文件大小字段
            audio_data = self._fix_wav_header(audio_data)
        else:
            logger.warning(f"[GPT-SoVITS] Unknown format header: {header.hex()}")

        # 直接保存为文件
        with open(output_path, 'wb') as f:
            f.write(audio_data)

        # 获取时长
        try:
            info = sf.info(output_path)
            logger.info(f"[GPT-SoVITS] Generated audio: {info.duration:.2f}s, format={info.format}, samplerate={info.samplerate}")
            return output_path, info.duration
        except Exception as e:
            logger.error(f"[GPT-SoVITS] Failed to read audio info: {e}")
            raise RuntimeError(f"Invalid audio format from GPT-SoVITS: {e}")

    def _fix_wav_header(self, audio_data: bytes) -> bytes:
        """修复 WAV 文件头中的文件大小字段"""
        import struct

        if len(audio_data) < 44:
            return audio_data

        # WAV 文件结构:
        # RIFF header (4 bytes) + file size (4 bytes) + WAVE (4 bytes)
        # fmt chunk + data chunk

        # 查找 data chunk
        data_pos = audio_data.find(b'data')
        if data_pos == -1:
            logger.warning("[GPT-SoVITS] Cannot find 'data' chunk in WAV")
            return audio_data

        # data chunk 格式: "data" (4 bytes) + data_size (4 bytes) + audio_data
        data_size_offset = data_pos + 4
        actual_data_size = len(audio_data) - (data_size_offset + 4)

        # 修复 RIFF 头中的文件大小 (总大小 - 8)
        riff_size = len(audio_data) - 8
        fixed_data = bytearray(audio_data)
        fixed_data[4:8] = struct.pack('<I', riff_size)

        # 修复 data chunk 中的数据大小
        fixed_data[data_size_offset:data_size_offset+4] = struct.pack('<I', actual_data_size)

        logger.info(f"[GPT-SoVITS] Fixed WAV header: riff_size={riff_size}, data_size={actual_data_size}")

        return bytes(fixed_data)


class EdgeTTSFallback:
    """EdgeTTS 语音合成（默认 TTS）"""

    # 数字人音色映射
    DEFAULT_VOICE_MAP = {
        "human_1": "zh-CN-XiaoxuanNeural",   # 数字人1：晓萱（温柔知性女声）
        "human_2": "zh-CN-YunxiNeural",      # 数字人2：云希（阳光男声）
    }

    def __init__(self, avatar_id: str = "human_1"):
        self.avatar_id = avatar_id
        self.default_voice = "zh-CN-XiaoxuanNeural"
        logger.info(f"[EdgeTTS] Initialized, avatar: {avatar_id}, voice: {self._get_voice()}")

    def set_avatar(self, avatar_id: str):
        """设置当前数字人"""
        self.avatar_id = avatar_id
        logger.info(f"[EdgeTTS] Avatar set to: {avatar_id}, voice: {self._get_voice()}")

    def _get_voice(self) -> str:
        """获取当前数字人对应的音色"""
        voice = self.DEFAULT_VOICE_MAP.get(self.avatar_id, self.default_voice)
        return voice

    async def generate(self, text: str, output_path: str, emotion: str = "default") -> Tuple[str, float]:
        """
        使用 EdgeTTS 生成音频

        Args:
            text: 输入文本
            output_path: 输出文件路径
            emotion: 情感标签（EdgeTTS 暂不支持情感，保留参数兼容）

        Returns:
            Tuple[str, float]: (音频文件路径, 音频时长)
        """
        import edge_tts
        import soundfile as sf
        import io

        voice = self._get_voice()
        logger.info(f"[EdgeTTS] Generating audio with voice: {voice}")

        communicate = edge_tts.Communicate(text, voice)

        # 收集音频数据
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])

        # 保存到文件
        audio_data.seek(0)
        with open(output_path, 'wb') as f:
            f.write(audio_data.read())

        # 获取时长
        audio_data.seek(0)
        info = sf.info(audio_data)
        logger.info(f"[EdgeTTS] Generated: {output_path}, duration: {info.duration:.2f}s")
        return output_path, info.duration


class KokoroTTS:
    """Kokoro-82M 本地 TTS"""

    def __init__(self, avatar_id: str = "human_1"):
        # Kokoro 模型路径
        self.kokoro_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "FasterLivePortrait", "checkpoints", "Kokoro-82M"
        )

        # 默认音色
        self.default_voice = "zf_001"

        # 音色映射
        self.voice_map = {}
        self.avatar_id = avatar_id
        self._load_voice_map()

        # 模型延迟加载
        self._model = None
        self._pipeline = None
        self._voices_loaded = False

        logger.info(f"[KokoroTTS] Initialized, default voice: {self.default_voice}")

    def _load_voice_map(self):
        """从配置文件加载数字人音色映射"""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Human_Choice", "avatar_config.json"
            )
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for avatar_id, info in config.get("avatars", {}).items():
                    kokoro_voice = info.get("kokoro_voice")
                    if kokoro_voice:
                        self.voice_map[avatar_id] = kokoro_voice
                        logger.info(f"[KokoroTTS] Mapped {avatar_id} -> {kokoro_voice}")
        except Exception as e:
            logger.warning(f"[KokoroTTS] Failed to load voice map: {e}")

    def set_avatar(self, avatar_id: str):
        """设置当前数字人"""
        self.avatar_id = avatar_id
        logger.info(f"[KokoroTTS] Avatar set to: {avatar_id}")

    def _get_voice(self) -> str:
        """获取当前数字人对应的音色"""
        voice = self.voice_map.get(self.avatar_id, self.default_voice)
        logger.info(f"[KokoroTTS] Using voice: {voice} for avatar: {self.avatar_id}")
        return voice

    def _init_model(self) -> bool:
        """延迟初始化 Kokoro 模型"""
        if self._model is not None:
            return True

        try:
            import platform
            import subprocess
            system = platform.system()

            if system == "Windows":
                espeak_path = r"C:\Program Files\eSpeak NG"
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(espeak_path, "libespeak-ng.dll")
                os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(espeak_path, "espeak-ng.exe")
            elif system == "Linux":
                # Linux 上检测并设置 espeak-ng 路径
                # 尝试多个可能的路径
                possible_libs = [
                    "/usr/lib/x86_64-linux-gnu/libespeak-ng.so",
                    "/usr/lib/aarch64-linux-gnu/libespeak-ng.so",
                    "/usr/local/lib/libespeak-ng.so",
                    "/usr/lib/libespeak-ng.so",
                ]
                possible_bins = [
                    "/usr/bin/espeak-ng",
                    "/usr/local/bin/espeak-ng",
                ]

                for lib_path in possible_libs:
                    if os.path.exists(lib_path):
                        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
                        logger.info(f"[KokoroTTS] Found espeak library: {lib_path}")
                        break

                for bin_path in possible_bins:
                    if os.path.exists(bin_path):
                        os.environ["PHONEMIZER_ESPEAK_PATH"] = bin_path
                        logger.info(f"[KokoroTTS] Found espeak binary: {bin_path}")
                        break

                # 检查 espeak-ng 是否安装
                try:
                    result = subprocess.run(["which", "espeak-ng"], capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info(f"[KokoroTTS] espeak-ng found at: {result.stdout.strip()}")
                    else:
                        logger.warning("[KokoroTTS] espeak-ng not found, please install: apt-get install espeak-ng")
                except Exception as e:
                    logger.warning(f"[KokoroTTS] Failed to check espeak-ng: {e}")

            # 尝试导入 kokoro
            try:
                from kokoro import KPipeline, KModel
            except ImportError as e:
                logger.error(f"[KokoroTTS] Failed to import kokoro: {e}")
                logger.error("[KokoroTTS] Please install: pip install kokoro>=0.3.0")
                return False

            # 检查模型文件（支持多个版本）
            config_path = os.path.join(self.kokoro_path, "config.json")
            # 优先尝试新版本模型
            model_candidates = [
                "kokoro-v1_1-zh.pth",
                "kokoro-v1_1.pth",
                "kokoro-v1_0.pth",
            ]
            model_path = None
            for candidate in model_candidates:
                candidate_path = os.path.join(self.kokoro_path, candidate)
                if os.path.exists(candidate_path):
                    model_path = candidate_path
                    logger.info(f"[KokoroTTS] Found model: {candidate}")
                    break

            if not os.path.exists(config_path):
                logger.error(f"[KokoroTTS] Config not found: {config_path}")
                return False
            if model_path is None:
                logger.error(f"[KokoroTTS] Model not found, tried: {model_candidates}")
                return False

            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)

            self._model = KModel(config=model_config, model=model_path)

            logger.info("[KokoroTTS] Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"[KokoroTTS] Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def generate(self, text: str, output_path: str) -> Tuple[str, float]:
        """
        生成语音

        Args:
            text: 输入文本
            output_path: 输出文件路径

        Returns:
            (音频路径, 音频时长)
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text")

        # 初始化模型
        if not self._init_model():
            raise RuntimeError("Failed to initialize Kokoro model")

        try:
            from kokoro import KPipeline
            import soundfile as sf
            import resampy

            voice = self._get_voice()

            # 创建 pipeline
            pipeline = KPipeline(lang_code=voice[0], model=self._model)

            # 加载音色
            if not self._voices_loaded:
                voice_dir = os.path.join(self.kokoro_path, "voices")
                for vname in os.listdir(voice_dir):
                    if vname.endswith(".pt"):
                        voice_name = os.path.splitext(vname)[0]
                        pipeline.voices[voice_name] = torch.load(
                            os.path.join(voice_dir, vname),
                            weights_only=True
                        )
                self._voices_loaded = True
                logger.info(f"[KokoroTTS] Loaded {len(pipeline.voices)} voices")

            # 生成音频
            t_start = time.time()
            generator = pipeline(
                text,
                voice=voice,
                speed=1,
                split_pattern=r'\n+'
            )

            # 收集所有音频片段
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)

            if not audios:
                raise RuntimeError("No audio generated")

            # 合并音频
            audio_data = np.concatenate(audios)
            logger.info(f"[KokoroTTS] Generated audio in {time.time() - t_start:.2f}s")

            # 转换采样率 (24kHz -> 16kHz)
            audio_16k = resampy.resample(audio_data, sr_orig=24000, sr_new=16000)

            # 保存文件
            sf.write(output_path, audio_16k, 16000)

            # 获取时长
            duration = len(audio_16k) / 16000
            logger.info(f"[KokoroTTS] Saved to {output_path}, duration: {duration:.2f}s")

            return output_path, duration

        except Exception as e:
            logger.error(f"[KokoroTTS] Generation failed: {e}")
            raise


def preprocess_text_for_tts(text: str) -> str:
    """
    预处理文本以优化 TTS 合成

    只做最小化处理：
    - 移除表情符号（可能导致 TTS 分段异常）
    - 规范化空白字符
    """
    if not text:
        return text

    original_text = text
    logger.info(f"[TTS Preprocess] Input: {repr(text[:100])}")

    # 使用简单直接的方式移除 emoji：逐字符过滤
    # emoji 通常在 Unicode 的 Supplementary Planes (U+10000 及以上)
    result_chars = []
    for char in text:
        code = ord(char)
        # 跳过 emoji 范围
        if 0x1F600 <= code <= 0x1F64F:  # Emoticons
            continue
        if 0x1F300 <= code <= 0x1F5FF:  # Misc Symbols and Pictographs
            continue
        if 0x1F680 <= code <= 0x1F6FF:  # Transport and Map
            continue
        if 0x1F1E0 <= code <= 0x1F1FF:  # Flags
            continue
        if 0x1F900 <= code <= 0x1F9FF:  # Supplemental Symbols and Pictographs
            continue
        if 0x1FA00 <= code <= 0x1FA6F:  # Chess Symbols
            continue
        if 0x1FA70 <= code <= 0x1FAFF:  # Symbols and Pictographs Extended-A
            continue
        if 0x2600 <= code <= 0x26FF:    # Misc symbols
            continue
        if 0x2700 <= code <= 0x27BF:    # Dingbats
            continue
        if code == 0xFE0F:              # Variation Selector-16
            continue
        if code == 0x200D:              # Zero Width Joiner
            continue
        result_chars.append(char)

    text = ''.join(result_chars)
    logger.info(f"[TTS Preprocess] After emoji removal: {repr(text[:100])}")

    # 规范化空白字符：多个换行/空格变成一个空格
    text = re.sub(r'\s+', ' ', text)

    # 移除首尾空格
    text = text.strip()

    # 移除开头的标点符号（如破折号、句号、波浪号等）
    text = re.sub(r'^[—…。！？，、；：\s\-～~]+', '', text)

    # 确保结尾有标点
    if text and text[-1] not in '。！？，、；：.!?～~':
        text += '。'

    # 如果处理后为空，返回原始文本
    if not text or len(text.strip()) == 0:
        logger.warning(f"[TTS Preprocess] Text became empty after processing, using original")
        text = original_text

    logger.info(f"[TTS Preprocess] Final: {repr(text[:100])}")
    logger.info(f"[TTS Preprocess] Length: {len(original_text)} -> {len(text)}")

    return text


class DigitalHumanBatchPipeline:
    """
    数字人批量处理流水线

    流程:
    1. ASR识别（如果是语音输入）
    2. Psychology_Rag 生成回复
    3. GPT-SoVITS TTS 生成音频
    4. BatchVideoGenerator 生成视频
    """

    def __init__(
        self,
        output_dir: str = "/tmp/digital_human_videos",
        dashscope_api_key: str = None,
        gpt_sovits_server: str = None,
        tts_ref_file: str = None,
        tts_ref_text: str = None,
        use_asr: bool = False,
        pipeline=None,
        joyvasa_pipeline=None,
        default_source_image: str = None,
        config: DigitalHumanConfig = None
    ):
        self.config = config or get_default_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        self.use_asr = use_asr

        # 初始化 ASR（可选）
        if use_asr:
            self.asr = SenseVoiceASR(self.config.sensevoice)
        else:
            self.asr = None

        # 初始化 LLM
        self.psy_mind = None
        try:
            # 添加 Psychology_Rag 到 Python 路径
            project_root = Path(__file__).resolve().parent.parent
            psy_rag_path = project_root / "Psychology_Rag"
            if psy_rag_path not in sys.path:
                sys.path.insert(0, str(psy_rag_path))
            from system import PsyMindSystem
            self.psy_mind = PsyMindSystem()
            logger.info("[Pipeline] Psychology_Rag initialized")
        except Exception as e:
            logger.warning(f"[Pipeline] Psychology_Rag init failed: {e}, will use simple LLM")

        # 初始化 TTS（默认使用 EdgeTTS）
        default_avatar_id = self.config.avatar.default_avatar
        self.tts = EdgeTTSFallback(avatar_id=default_avatar_id)
        self.tts_gptsovits = GPTSoVITSTTS(self.config.gpt_sovits, avatar_id=default_avatar_id)
        self.tts_kokoro = None  # Kokoro 延迟加载（需要时才初始化）
        self.current_avatar_id = default_avatar_id
        logger.info(f"[Pipeline] TTS initialized: EdgeTTS (primary) -> GPT-SoVITS (fallback)")

        # 初始化视频生成器
        self.video_generator = BatchVideoGeneratorSimple(
            pipeline=pipeline,
            joyvasa_pipeline=joyvasa_pipeline,
            output_dir=str(self.output_dir),
            emotion_config=self.config.emotion,  # 传递情感配置
            motion_seed=self.config.video.motion_seed  # 传递运动随机种子
        )

        # 默认源图像
        self.default_source_image = default_source_image

        # 记录最后处理的文本
        self.last_user_text = ""
        self.last_response_text = ""

        logger.info(f"[Pipeline] Initialized, output: {self.output_dir}")

    def _set_avatar_for_tts(self, avatar_id: str = None):
        """更新当前 TTS 所使用的数字人"""
        if avatar_id:
            self.current_avatar_id = avatar_id
            self.tts.set_avatar(avatar_id)
            self.tts_gptsovits.set_avatar(avatar_id)

    async def _generate_from_response_text(
        self,
        response_text: str,
        user_text: str,
        source_image: str = None,
        session_id: str = None,
        avatar_id: str = None,
        emotion: str = "default",
        video_emotion: str = "default",
        start_time: float = None,
        metrics: Dict[str, Any] = None
    ) -> PipelineResult:
        """
        使用既定文本直接执行 TTS 与视频生成

        Args:
            response_text: 要送入 TTS 的文本
            user_text: 对外记录的输入文本
            source_image: 源图像路径
            session_id: 会话 ID
            avatar_id: 数字人 ID
            emotion: 情感标签
            video_emotion: 视频驱动情感标签
            start_time: 流程起始时间
            metrics: 累计指标
        """
        start_time = start_time or time.time()
        metrics = {} if metrics is None else metrics
        source_image = source_image or self.default_source_image

        if not source_image:
            raise RuntimeError("No source image configured")

        self._set_avatar_for_tts(avatar_id)

        self.last_user_text = user_text
        self.last_response_text = response_text

        # ========== Step 3: TTS 生成音频 ==========
        t3 = time.time()
        logger.info("[Pipeline] Step 3: TTS generating audio...")
        audio_path, audio_duration = await self._tts_generate(response_text, emotion)
        metrics['tts_time'] = time.time() - t3
        logger.info(f"[Pipeline] TTS: {metrics['tts_time']:.2f}s, audio duration: {audio_duration:.2f}s")

        # ========== Step 4: 生成视频 ==========
        t4 = time.time()
        logger.info("[Pipeline] Step 4: Generating video...")
        video_result = self.video_generator.generate(
            audio_path=audio_path,
            source_image_path=source_image,
            session_id=session_id,
            emotion=video_emotion
        )
        metrics['video_time'] = video_result.total_time
        metrics['joyvasa_time'] = video_result.metrics.get('joyvasa_time', 0)
        metrics['render_time'] = video_result.metrics.get('render_time', 0)
        metrics['ffmpeg_time'] = video_result.metrics.get('ffmpeg_time', 0)
        metrics['motion_post_time'] = video_result.metrics.get('motion_post_time', 0)
        metrics['generate_video_wall_time'] = time.time() - t4
        logger.info(f"[Pipeline] Video: {metrics['video_time']:.2f}s")

        # ========== 清理临时音频 ==========
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        total_time = time.time() - start_time
        metrics['total_time'] = total_time

        if total_time > 30:
            logger.warning(f"[Pipeline] Generation timeout: {total_time:.2f}s > 30s")

        video_filename = os.path.basename(video_result.video_path)
        video_url = f"/videos/{video_filename}"

        logger.info(f"[Pipeline] Completed in {total_time:.2f}s")

        return PipelineResult(
            video_path=video_result.video_path,
            video_url=video_url,
            response_text=response_text,
            user_text=user_text,
            total_time=total_time,
            metrics=metrics,
            audio_duration=audio_duration,
            success=True
        )

    async def process_input(
        self,
        user_text: str = None,
        user_audio_path: str = None,
        source_image: str = None,
        session_id: str = None,
        avatar_id: str = None
    ) -> PipelineResult:
        """
        处理输入，生成视频

        Args:
            user_text: 用户输入文本
            user_audio_path: 用户输入音频路径（需要 ASR）
            source_image: 源图像路径（可选，使用默认）
            session_id: 会话 ID
            avatar_id: 数字人 ID（可选，用于切换参考音频）

        Returns:
            PipelineResult: 处理结果
        """
        start_time = time.time()
        source_image = source_image or self.default_source_image
        metrics = {}

        self._set_avatar_for_tts(avatar_id)

        try:
            # ========== Step 1: ASR 识别（如果是语音输入） ==========
            if user_audio_path and not user_text and self.use_asr:
                t1 = time.time()
                logger.info("[Pipeline] Step 1: ASR recognition...")
                user_text = await self._asr_recognize(user_audio_path)
                metrics['asr_time'] = time.time() - t1
                logger.info(f"[Pipeline] ASR: {metrics['asr_time']:.2f}s, text: {user_text}")

            if not user_text:
                return PipelineResult(
                    video_path="", video_url="",
                    response_text="", user_text="",
                    total_time=0, metrics=metrics,
                    audio_duration=0, success=False,
                    error_message="No input text provided"
                )

            self.last_user_text = user_text

            # ========== Step 2: LLM 生成回复（带情绪上下文） ==========
            t2 = time.time()
            logger.info("[Pipeline] Step 2: LLM generating response...")

            emotion = "default"  # 默认情感
            emotion_context = ""  # 情绪上下文

            # 获取融合情绪上下文
            try:
                from digital_human.emotion_fusion import get_emotion_fusion
                fusion = get_emotion_fusion()
                fused = fusion.fuse(user_text, session_id or "default")
                emotion_context = fused.prompt_context
                if emotion_context:
                    logger.info(f"[Pipeline] Emotion context detected: {fused.primary_emotion}, intensity={fused.intensity:.2f}")
            except Exception as e:
                logger.warning(f"[Pipeline] Failed to get emotion context: {e}")

            if self.psy_mind:
                result = await self.psy_mind.process_message(
                    user_text,
                    session_id or "default",
                    emotion_context  # 传递情绪上下文
                )
                response_text = result.get("response", "")
                emotion = result.get("emotion", "default")
            else:
                response_text = await self._simple_llm_response(user_text)

            metrics['llm_time'] = time.time() - t2
            logger.info(f"[Pipeline] LLM: {metrics['llm_time']:.2f}s, emotion: {emotion}, response length: {len(response_text)}")

            return await self._generate_from_response_text(
                response_text=response_text,
                user_text=user_text,
                source_image=source_image,
                session_id=session_id,
                avatar_id=avatar_id,
                emotion=emotion,
                video_emotion="default",
                start_time=start_time,
                metrics=metrics
            )

        except Exception as e:
            logger.exception(f"[Pipeline] Error: {e}")
            return PipelineResult(
                video_path="", video_url="",
                response_text="", user_text=user_text or "",
                total_time=time.time() - start_time,
                metrics=metrics, audio_duration=0,
                success=False, error_message=str(e)
            )

    async def process_direct_text(
        self,
        tts_text: str,
        source_image: str = None,
        session_id: str = None,
        avatar_id: str = None
    ) -> PipelineResult:
        """
        直接将输入文本送入 TTS，并继续生成视频

        Args:
            tts_text: 直接用于 TTS 的文本
            source_image: 源图像路径
            session_id: 会话 ID
            avatar_id: 数字人 ID
        """
        start_time = time.time()
        metrics = {}
        source_image = source_image or self.default_source_image
        self._set_avatar_for_tts(avatar_id)

        try:
            if not tts_text or not tts_text.strip():
                return PipelineResult(
                    video_path="", video_url="",
                    response_text="", user_text="",
                    total_time=0, metrics=metrics,
                    audio_duration=0, success=False,
                    error_message="No TTS text provided"
                )

            logger.info("[Pipeline] Terminal direct-text mode: skipping ASR and LLM")

            return await self._generate_from_response_text(
                response_text=tts_text.strip(),
                user_text=tts_text.strip(),
                source_image=source_image,
                session_id=session_id,
                avatar_id=avatar_id,
                emotion="default",
                video_emotion="default",
                start_time=start_time,
                metrics=metrics
            )
        except Exception as e:
            logger.exception(f"[Pipeline] Direct text error: {e}")
            return PipelineResult(
                video_path="", video_url="",
                response_text="", user_text=tts_text or "",
                total_time=time.time() - start_time,
                metrics=metrics, audio_duration=0,
                success=False, error_message=str(e)
            )

    async def _asr_recognize(self, audio_path: str) -> str:
        """ASR 语音识别"""
        if self.asr is None:
            return ""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.asr.recognize, audio_path)

    async def _simple_llm_response(self, user_text: str) -> str:
        """简单 LLM 回复（当 Psychology_Rag 不可用时）"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.config.dashscope.api_key,
                base_url=self.config.dashscope.base_url
            )

            response = await client.chat.completions.create(
                model=self.config.dashscope.model_name,
                messages=[
                    {"role": "system", "content": "你是一个友好、专业的心理健康助手。请用简洁、温暖的语言回复用户。"},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=self.config.dashscope.max_tokens,
                temperature=self.config.dashscope.temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"抱歉，我现在无法回复。您说的是：{user_text}"

    async def _tts_generate(self, text: str, emotion: str = "default") -> Tuple[str, float]:
        """
        TTS 生成音频（默认使用 EdgeTTS）

        优先级: EdgeTTS -> GPT-SoVITS

        Args:
            text: 要合成的文本
            emotion: 情感标签 (default/happy/sad/calm/question)

        Returns:
            Tuple[str, float]: (音频文件路径, 音频时长)
        """
        # 生成临时音频文件路径
        audio_path = str(self.temp_dir / f"tts_{int(time.time() * 1000)}.wav")

        # 预处理文本（移除表情符号、规范化换行等）
        processed_text = preprocess_text_for_tts(text)
        logger.info(f"[TTS] Original text: {text}")
        logger.info(f"[TTS] Processed text: {processed_text}")
        logger.info(f"[TTS] Text length: {len(text)} -> {len(processed_text)}")

        # 验证处理后的文本不为空
        if not processed_text or len(processed_text.strip()) == 0:
            logger.error(f"[TTS] Text became empty after preprocessing!")
            # 使用原始文本作为备选
            processed_text = text
            logger.info(f"[TTS] Falling back to original text: {processed_text}")

        logger.info(f"[TTS] Emotion: {emotion}, Avatar: {self.current_avatar_id}")

        # 优先使用 EdgeTTS
        try:
            result = await self.tts.generate(processed_text, audio_path, emotion=emotion)
            logger.info(f"[TTS] EdgeTTS success: path={result[0]}, duration={result[1]:.2f}s")

            # 验证生成的音频文件有效
            if result[1] < 0.1:
                logger.warning(f"[TTS] Generated audio too short: {result[1]:.2f}s, text may be invalid")

            return result
        except Exception as e:
            logger.warning(f"[TTS] EdgeTTS failed: {e}, trying GPT-SoVITS")

            # 回退到 GPT-SoVITS
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.tts_gptsovits.generate(processed_text, audio_path, emotion=emotion)
                )
                logger.info(f"[TTS] GPT-SoVITS success: path={result[0]}, duration={result[1]:.2f}s")
                return result
            except Exception as e2:
                logger.error(f"[TTS] GPT-SoVITS also failed: {e2}")
                raise RuntimeError("All TTS methods failed")

    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.video_generator.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


class BatchPipelineManager:
    """
    批量流水线管理器

    管理多个会话的流水线实例
    """

    def __init__(self):
        self.config = get_default_config()
        self._pipelines: Dict[str, DigitalHumanBatchPipeline] = {}
        self._initialized = False
        self.joyvasa_pipeline = None
        self.default_source_image = None

    def initialize(
        self,
        pipeline=None,
        joyvasa_pipeline=None,
        config=None,
        default_source_image: str = None,
        dashscope_api_key: str = None,
        gpt_sovits_server: str = None,
        tts_ref_file: str = None,
        tts_ref_text: str = None,
        output_dir: str = None
    ):
        """初始化管理器"""

        # 使用传入的配置或创建默认配置
        if config:
            self.config = config

        # 兼容旧版调用：如果传递了单独的参数，覆盖配置
        if dashscope_api_key:
            self.config.dashscope.api_key = dashscope_api_key
        if gpt_sovits_server:
            self.config.gpt_sovits.server_url = gpt_sovits_server
        if tts_ref_file:
            self.config.gpt_sovits.ref_audio = tts_ref_file
        if tts_ref_text:
            self.config.gpt_sovits.ref_text = tts_ref_text
        if output_dir:
            self.config.output_dir = output_dir

        # 保存 pipeline 引用（兼容 app.py）
        self.pipeline = pipeline
        self.joyvasa_pipeline = joyvasa_pipeline or pipeline
        self.default_source_image = default_source_image
        self._initialized = True

        logger.info("[BatchPipelineManager] Initialized")
        logger.info(f"  - GPT-SoVITS: {self.config.gpt_sovits.server_url}")
        logger.info(f"  - Output: {self.config.output_dir}")

    def get_or_create_pipeline(self, session_id: str) -> DigitalHumanBatchPipeline:
        """获取或创建流水线实例"""
        if not self._initialized:
            raise RuntimeError("BatchPipelineManager not initialized")

        if session_id not in self._pipelines:
            self._pipelines[session_id] = DigitalHumanBatchPipeline(
                output_dir=self.config.output_dir,
                config=self.config,
                pipeline=self.pipeline,
                joyvasa_pipeline=self.joyvasa_pipeline,
                default_source_image=self.default_source_image,
                use_asr=self.config.use_asr
            )

        return self._pipelines[session_id]

    def cleanup_session(self, session_id: str):
        """清理会话"""
        if session_id in self._pipelines:
            self._pipelines[session_id].cleanup()
            del self._pipelines[session_id]

    async def process(
        self,
        session_id: str,
        user_text: str = None,
        user_audio_path: str = None,
        source_image: str = None,
        avatar_id: str = None
    ) -> 'PipelineResult':
        """处理输入，生成视频"""
        pipeline = self.get_or_create_pipeline(session_id)
        return await pipeline.process_input(
            user_text=user_text,
            user_audio_path=user_audio_path,
            source_image=source_image,
            session_id=session_id,
            avatar_id=avatar_id
        )

    async def process_direct_text(
        self,
        session_id: str,
        tts_text: str,
        source_image: str = None,
        avatar_id: str = None
    ) -> 'PipelineResult':
        """直接将文本送入 TTS 并生成视频"""
        pipeline = self.get_or_create_pipeline(session_id)
        return await pipeline.process_direct_text(
            tts_text=tts_text,
            source_image=source_image,
            session_id=session_id,
            avatar_id=avatar_id
        )

    def cleanup_all(self):
        """清理所有会话"""
        for session_id in list(self._pipelines.keys()):
            self.cleanup_session(session_id)
        logger.info("[BatchPipelineManager] All sessions cleaned up")


# 全局管理器实例
_batch_pipeline_manager: Optional[BatchPipelineManager] = None


def get_batch_pipeline_manager() -> BatchPipelineManager:
    """获取全局批量流水线管理器"""
    global _batch_pipeline_manager
    if _batch_pipeline_manager is None:
        _batch_pipeline_manager = BatchPipelineManager()
    return _batch_pipeline_manager


# 导出全局实例（兼容 app.py 的导入方式）
batch_pipeline_manager = get_batch_pipeline_manager()

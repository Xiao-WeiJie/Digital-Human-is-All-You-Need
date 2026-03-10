# -*- coding: utf-8 -*-
"""
Digital Human Phase 1 - GPT-SoVITS TTS Adapter

调用 GPT-SoVITS API 进行语音合成。
支持本地 Kokoro TTS 作为备选。
"""

import asyncio
import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import httpx

from .config import GPTSoVITSConfig


class TTSAdapter:
    """TTS 适配器 - 支持 GPT-SoVITS API 和 Kokoro 本地"""

    def __init__(self, config: Optional[GPTSoVITSConfig] = None):
        self.config = config or GPTSoVITSConfig.from_env()
        self._kokoro_available = None
        self._gptsovits_available = None

    def check_gptsovits_available(self) -> bool:
        """检查 GPT-SoVITS API 是否可用"""
        if self._gptsovits_available is not None:
            return self._gptsovits_available

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.config.api_url}/docs", follow_redirects=True)
                if resp.status_code == 200:
                    print(f"[TTS] GPT-SoVITS API 可用: {self.config.api_url}")
                    self._gptsovits_available = True
                    return True
                resp = client.get(f"{self.config.api_url}/")
                self._gptsovits_available = resp.status_code < 500
        except Exception as e:
            print(f"[TTS] GPT-SoVITS API 不可用: {e}")
            self._gptsovits_available = False

        return self._gptsovits_available

    def check_kokoro_available(self) -> bool:
        """检查本地 Kokoro TTS 是否可用"""
        # 如果配置禁用了 Kokoro，直接返回 False
        if not self.config.enable_kokoro_fallback:
            print("[TTS] Kokoro 已在配置中禁用")
            self._kokoro_available = False
            return False

        if self._kokoro_available is not None:
            return self._kokoro_available

        try:
            import platform
            if platform.system() == "Windows":
                espeak_lib = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
                espeak_exe = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
                if not (os.path.exists(espeak_lib) and os.path.exists(espeak_exe)):
                    self._kokoro_available = False
                    return False
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib
                os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_exe

            from kokoro import KPipeline, KModel
            model_path = Path("FasterLivePortrait/checkpoints/Kokoro-82M")
            if not model_path.exists():
                self._kokoro_available = False
                return False

            self._kokoro_available = True
        except ImportError:
            print("[TTS] Kokoro 未安装")
            self._kokoro_available = False
        except Exception as e:
            print(f"[TTS] Kokoro 初始化失败: {e}")
            self._kokoro_available = False

        return self._kokoro_available

    def synthesize_gptsovits(
        self,
        text: str,
        output_path: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> str:
        """使用 GPT-SoVITS API 合成语音"""
        t0 = time.time()

        params = {
            "text": text,
            "text_lang": self.config.text_lang,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
            "text_split_method": self.config.text_split_method,
            "batch_size": self.config.batch_size,
            "speed_factor": self.config.speed_factor,
            "splitting": self.config.splitting,
            "media_type": "wav",
            "seed": -1,
        }

        ref_audio = ref_audio_path or self.config.ref_audio_path
        if ref_audio:
            params["ref_audio_path"] = ref_audio
            params["prompt_text"] = prompt_text or self.config.prompt_text
            params["prompt_lang"] = self.config.prompt_lang
            print(f"[TTS] 使用参考音频: {ref_audio}")
            print(f"[TTS] 参考文本: {params['prompt_text'][:30]}...")

        print(f"[TTS] 调用 GPT-SoVITS API: {self.config.api_url}/tts")
        print(f"[TTS] 文本: {text[:50]}...")

        with httpx.Client(
            timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=1),
        ) as client:
            resp = client.post(f"{self.config.api_url}/tts", json=params, headers={"Connection": "close"})

            if resp.status_code != 200:
                error_detail = resp.text[:200] if resp.text else "Unknown error"
                raise Exception(f"GPT-SoVITS API 返回错误 {resp.status_code}: {error_detail}")

            content_type = resp.headers.get("content-type", "")

            if "audio" in content_type or "application/octet-stream" in content_type:
                audio_data = resp.content
            elif "application/json" in content_type:
                result = resp.json()
                if "audio" in result:
                    audio_data = base64.b64decode(result["audio"])
                elif "wav" in result:
                    audio_data = base64.b64decode(result["wav"])
                else:
                    raise Exception(f"未知的响应格式: {list(result.keys())}")
            else:
                audio_data = resp.content

            if output_path is None:
                output_path = tempfile.mktemp(suffix=".wav")

            # 调试：显示响应信息
            print(f"[TTS] 响应 Content-Type: {content_type}")
            print(f"[TTS] 响应数据大小: {len(audio_data)} bytes")

            # 检查是否是标准 WAV 格式 (RIFF header)
            is_wav = audio_data[:4] == b'RIFF' if len(audio_data) >= 4 else False
            print(f"[TTS] 是否 WAV 格式: {is_wav}")

            # 调试：显示 WAV 头信息
            if is_wav and len(audio_data) >= 44:
                import struct
                riff = audio_data[:4]
                file_size = struct.unpack('<I', audio_data[4:8])[0]
                wave_id = audio_data[8:12]
                fmt_chunk_id = audio_data[12:16]
                fmt_size = struct.unpack('<I', audio_data[16:20])[0]
                audio_format = struct.unpack('<H', audio_data[20:22])[0]
                num_channels = struct.unpack('<H', audio_data[22:24])[0]
                sample_rate = struct.unpack('<I', audio_data[24:28])[0]
                byte_rate = struct.unpack('<I', audio_data[28:32])[0]
                block_align = struct.unpack('<H', audio_data[32:34])[0]
                bits_per_sample = struct.unpack('<H', audio_data[34:36])[0]
                print(f"[TTS] WAV 头: format={audio_format}, channels={num_channels}, "
                      f"rate={sample_rate}, bits={bits_per_sample}")

                # 查找 data chunk
                data_start = 36
                while data_start < len(audio_data) - 8:
                    chunk_id = audio_data[data_start:data_start+4]
                    chunk_size = struct.unpack('<I', audio_data[data_start+4:data_start+8])[0]
                    if chunk_id == b'data':
                        print(f"[TTS] data chunk: offset={data_start}, size={chunk_size}")
                        expected_samples = chunk_size // (num_channels * bits_per_sample // 8)
                        print(f"[TTS] 预期样本数: {expected_samples}, 时长: {expected_samples/sample_rate:.2f}s")
                        break
                    data_start += 8 + chunk_size
                else:
                    print("[TTS] 未找到 data chunk!")

            # 使用 soundfile 重新保存音频，确保格式正确
            import numpy as np
            import soundfile as sf
            import io
            import struct

            try:
                if is_wav:
                    # 解析 WAV 头获取格式信息
                    audio_format = struct.unpack('<H', audio_data[20:22])[0]
                    num_channels = struct.unpack('<H', audio_data[22:24])[0]
                    sample_rate = struct.unpack('<I', audio_data[24:28])[0]
                    bits_per_sample = struct.unpack('<H', audio_data[34:36])[0]

                    # 查找 data chunk
                    data_offset = 36
                    data_size = 0
                    while data_offset < len(audio_data) - 8:
                        chunk_id = audio_data[data_offset:data_offset+4]
                        chunk_size = struct.unpack('<I', audio_data[data_offset+4:data_offset+8])[0]
                        if chunk_id == b'data':
                            data_size = chunk_size
                            data_offset += 8  # 跳过 chunk id 和 size
                            break
                        data_offset += 8 + chunk_size

                    # 如果 data_size 为 0 或 0xFFFFFFFF，计算实际大小
                    if data_size == 0 or data_size == 0xFFFFFFFF:
                        data_size = len(audio_data) - data_offset
                        print(f"[TTS] 流式 WAV，计算数据大小: {data_size} bytes")

                    # 读取 PCM 数据
                    bytes_per_sample = bits_per_sample // 8
                    audio_array = np.frombuffer(
                        audio_data[data_offset:data_offset + data_size],
                        dtype=np.int16 if bits_per_sample == 16 else np.int32
                    ).astype(np.float32) / (32768.0 if bits_per_sample == 16 else 2147483648.0)

                    # 确保是单声道
                    if num_channels > 1:
                        audio_array = audio_array.reshape(-1, num_channels).mean(axis=1)

                    print(f"[TTS] 手动解析: 采样率={sample_rate}, 样本数={len(audio_array)}, 时长={len(audio_array)/sample_rate:.2f}s")

                    # 用 soundfile 重新保存
                    sf.write(output_path, audio_array, sample_rate)
                else:
                    # 如果不是 WAV 格式，假设是 raw PCM 数据
                    print("[TTS] 非 WAV 格式，尝试解析为 raw PCM...")

                    # 尝试不同的采样率
                    for sr in [32000, 24000, 16000, 44100, 22050]:
                        try:
                            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                            duration = len(audio_array) / sr

                            if duration > 1.0:
                                print(f"[TTS] 使用采样率 {sr} Hz, 时长={duration:.2f}s")
                                sf.write(output_path, audio_array, sr)
                                break
                        except Exception as e:
                            print(f"[TTS] 采样率 {sr} 失败: {e}")
                            continue
                    else:
                        print("[TTS] PCM 解析失败，直接保存原始数据")
                        with open(output_path, "wb") as f:
                            f.write(audio_data)

                # 验证输出文件
                verify_array, verify_sr = sf.read(output_path)
                print(f"[TTS] 输出验证: 采样率={verify_sr}, 样本数={len(verify_array)}, 时长={len(verify_array)/verify_sr:.2f}s")

            except Exception as e:
                import traceback
                print(f"[TTS] 处理失败 ({e})，直接保存原始数据")
                traceback.print_exc()
                with open(output_path, "wb") as f:
                    f.write(audio_data)

            print(f"[TTS] GPT-SoVITS 合成完成，耗时 {time.time() - t0:.2f}s")
            return output_path

    def synthesize_kokoro(
        self,
        text: str,
        output_path: Optional[str] = None,
        voice_name: str = "af",
    ) -> str:
        """使用本地 Kokoro TTS 合成语音"""
        import json
        import numpy as np
        import torch
        import soundfile as sf
        from kokoro import KPipeline, KModel

        t0 = time.time()

        model_config_path = Path("FasterLivePortrait/checkpoints/Kokoro-82M/config.json")
        model_weight_path = Path("FasterLivePortrait/checkpoints/Kokoro-82M/kokoro-v1_0.pth")
        voice_path = Path("FasterLivePortrait/checkpoints/Kokoro-82M/voices")

        if not model_config_path.exists():
            raise FileNotFoundError(f"Kokoro 配置文件不存在: {model_config_path}")

        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)

        model = KModel(config=model_config, model=str(model_weight_path))
        pipeline = KPipeline(lang_code=voice_name[0], model=model)

        model.voices = {}
        if voice_path.exists():
            for vname in os.listdir(voice_path):
                pipeline.voices[os.path.splitext(vname)[0]] = torch.load(
                    os.path.join(voice_path, vname), weights_only=True
                )

        generator = pipeline(text, voice=voice_name, speed=1, split_pattern=r'\n+')

        audios = [audio for _, _, audio in generator]
        if not audios:
            raise Exception("Kokoro 未能生成音频")

        audio_data = np.concatenate(audios)

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        sf.write(output_path, audio_data, self.config.output_sample_rate)

        print(f"[TTS] Kokoro 合成完成，耗时 {time.time() - t0:.2f}s")
        return output_path

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        prefer_gptsovits: bool = True,
        **kwargs,
    ) -> str:
        """语音合成（自动选择可用引擎）"""
        print(f"[TTS] 合成文本: {text[:50]}...")

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        if prefer_gptsovits and self.check_gptsovits_available():
            try:
                return self.synthesize_gptsovits(text, output_path, **kwargs)
            except Exception as e:
                print(f"[TTS] GPT-SoVITS 失败，尝试 Kokoro: {e}")

        if self.check_kokoro_available():
            try:
                voice_name = kwargs.get("voice_name", "af")
                return self.synthesize_kokoro(text, output_path, voice_name)
            except Exception as e:
                print(f"[TTS] Kokoro 失败: {e}")
                raise

        raise Exception("没有可用的 TTS 引擎")

    async def synthesize_async(self, text: str, output_path: Optional[str] = None, **kwargs) -> str:
        """异步合成接口"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.synthesize(text, output_path, **kwargs))

    def __call__(self, text: str, **kwargs) -> str:
        """使适配器可调用"""
        return self.synthesize(text, **kwargs)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整语音链路测试

流程:
    用户语音 (通过参数传入)
        ↓
    SenseVoice ASR 识别
        ↓
    RAG/LLM 生成回复
        ↓
    GPT-SoVITS 合成语音
        ↓
    FasterLivePortrait 生成视频

使用方式:
    python test_voice_pipeline.py --audio autodl-tmp/test.wav --source FasterLivePortrait/assets/examples/source/s12.jpg

配置文件: digital_human/config.py
    - RAGConfig.dashscope_api_key: 阿里云 API Key
    - GPTSoVITSConfig.api_url: GPT-SoVITS 服务地址
    - GPTSoVITSConfig.ref_audio_path: 参考音频路径
    - GPTSoVITSConfig.prompt_text: 参考音频对应的文本
"""

import argparse
import os
import sys
import time
import asyncio
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="数字人完整语音链路测试")
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="用户语音输入文件路径 (如: autodl-tmp/test.wav)"
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="数字人源图像路径 (默认使用 config.py 中的配置)"
    )
    parser.add_argument(
        "--session",
        default=None,
        help="会话 ID (可选，用于保持对话上下文)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  数字人完整语音链路测试")
    print("=" * 70)
    print()

    # ============================================================
    # Step 1: 加载配置
    # ============================================================
    print("[Step 1/6] 加载配置...")

    from digital_human import get_config, set_config, DigitalHumanConfig

    config = DigitalHumanConfig.from_env()
    set_config(config)

    # 命令行参数覆盖
    user_audio = args.audio
    source_image = args.source or config.faster_live_portrait.default_source_image
    session_id = args.session

    print(f"  用户音频: {user_audio}")
    print(f"  源图像: {source_image}")
    print(f"  TTS API: {config.gpt_sovits.api_url}")
    print(f"  参考音频: {config.gpt_sovits.ref_audio_path}")
    print(f"  API Key: {'已设置' if config.rag.dashscope_api_key else '未设置'}")
    print()

    # 验证
    if not config.rag.dashscope_api_key:
        print("[X] 错误: 请在 config.py 中设置 RAGConfig.dashscope_api_key")
        return 1

    if not os.path.exists(user_audio):
        print(f"[X] 错误: 用户音频不存在: {user_audio}")
        return 1

    if not source_image or not os.path.exists(source_image):
        print(f"[X] 错误: 源图像不存在: {source_image}")
        return 1

    print("[OK] 配置验证通过")
    print()

    # ============================================================
    # Step 2: SenseVoice ASR
    # ============================================================
    print("=" * 70)
    print("[Step 2/6] SenseVoice ASR 语音识别...")
    print("=" * 70)

    from digital_human import SenseVoiceAdapter

    asr = SenseVoiceAdapter(config.sensevoice)
    asr_result = asr.transcribe(user_audio)

    user_text = asr_result["text"]
    if not user_text.strip():
        print("[X] 错误: 语音识别结果为空")
        return 1

    print(f"  识别文本: {user_text}")
    print(f"  耗时: {asr_result['time_cost']:.2f}s")
    print("[OK] 语音识别完成")
    print()

    # ============================================================
    # Step 3: RAG/LLM
    # ============================================================
    print("=" * 70)
    print("[Step 3/6] RAG/LLM 生成回复...")
    print("=" * 70)

    from digital_human import RAGAdapter

    rag = RAGAdapter(config.rag)
    rag.initialize()

    rag_result = asyncio.run(rag.chat(user_text, session_id))

    ai_response = rag_result["response"]
    if not ai_response:
        print("[X] 错误: LLM 回复为空")
        return 1

    print(f"  用户: {user_text}")
    print(f"  AI: {ai_response}")
    print("[OK] 回复生成完成")
    print()

    # ============================================================
    # Step 4: GPT-SoVITS TTS
    # ============================================================
    print("=" * 70)
    print("[Step 4/6] GPT-SoVITS 语音合成...")
    print("=" * 70)

    from digital_human import TTSAdapter

    tts = TTSAdapter(config.gpt_sovits)

    if not tts.check_gptsovits_available():
        print("[X] 错误: GPT-SoVITS 服务不可用")
        print(f"    请确保服务正在运行: {config.gpt_sovits.api_url}")
        return 1

    print(f"  合成文本: {ai_response[:50]}...")
    t0 = time.time()

    tts_audio_path = tts.synthesize(ai_response)

    print(f"  耗时: {time.time() - t0:.2f}s")
    print(f"  音频: {tts_audio_path}")
    print("[OK] 语音合成完成")
    print()

    # ============================================================
    # Step 5: FasterLivePortrait
    # ============================================================
    print("=" * 70)
    print("[Step 5/6] FasterLivePortrait 生成视频...")
    print("=" * 70)

    from digital_human import LivePortraitAdapter

    lpp = LivePortraitAdapter(config.faster_live_portrait)

    print(f"  源图像: {source_image}")
    print(f"  驱动音频: {tts_audio_path}")
    print("  正在生成视频...")
    t0 = time.time()

    video_path = lpp.generate_from_audio(tts_audio_path, source_image)

    print(f"  耗时: {time.time() - t0:.2f}s")
    print(f"  视频: {video_path}")
    print("[OK] 视频生成完成")
    print()

    # ============================================================
    # Step 6: 完成
    # ============================================================
    print("=" * 70)
    print("[Step 6/6] 测试完成!")
    print("=" * 70)
    print()
    print("结果汇总:")
    print(f"  用户输入: {user_text}")
    print(f"  AI 回复: {ai_response}")
    print(f"  输出视频: {video_path}")
    print()
    print("测试通过!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

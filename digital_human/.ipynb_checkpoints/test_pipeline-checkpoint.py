# -*- coding: utf-8 -*-
"""
Digital Human Phase 1 - Test Script

测试脚本：验证各模块和完整链路。

使用方式:
    # 设置环境变量
    export DASHSCOPE_API_KEY="your-api-key"

    # 运行测试
    python test_pipeline.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_config():
    """测试配置模块"""
    print("\n" + "=" * 60)
    print("测试 1: 配置模块")
    print("=" * 60)

    from digital_human.config import DigitalHumanConfig, get_config

    config = DigitalHumanConfig.from_env()
    print(f"SenseVoice 设备: {config.sensevoice.device}")
    print(f"RAG API Key: {'已设置' if config.rag.dashscope_api_key else '未设置'}")
    print(f"GPT-SoVITS URL: {config.gpt_sovits.api_url}")
    print(f"FLP 配置路径: {config.faster_live_portrait.config_path}")

    return True


def test_sensevoice():
    """测试 SenseVoice ASR"""
    print("\n" + "=" * 60)
    print("测试 2: SenseVoice ASR")
    print("=" * 60)

    from digital_human.sensevoice_adapter import SenseVoiceAdapter

    # 查找示例音频
    example_audio = PROJECT_ROOT / "SenseVoice" / "iic" / "SenseVoiceSmall" / "example" / "zh.mp3"
    if not example_audio.exists():
        # 尝试另一个路径
        example_audio = PROJECT_ROOT / "SenseVoice" / "example" / "zh.mp3"

    if not example_audio.exists():
        print("跳过: 未找到示例音频文件")
        return None

    print(f"使用音频: {example_audio}")

    adapter = SenseVoiceAdapter()
    result = adapter.transcribe(str(example_audio))

    print(f"识别结果: {result['text']}")
    print(f"耗时: {result['time_cost']:.2f}s")

    return result['text'] is not None


def test_rag():
    """测试 RAG 系统"""
    print("\n" + "=" * 60)
    print("测试 3: RAG 对话系统")
    print("=" * 60)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("跳过: DASHSCOPE_API_KEY 未设置")
        return None

    from digital_human.rag_adapter import RAGAdapter
    from digital_human.config import RAGConfig

    config = RAGConfig(dashscope_api_key=api_key)
    adapter = RAGAdapter(config)

    # 同步调用
    response = adapter.chat_sync("你好，请简单介绍一下你自己")
    print(f"回复: {response}")

    return response is not None


def test_tts():
    """测试 TTS"""
    print("\n" + "=" * 60)
    print("测试 4: TTS 语音合成")
    print("=" * 60)

    from digital_human.tts_adapter import TTSAdapter

    adapter = TTSAdapter()

    # 检查可用引擎
    gptsovits = adapter.check_gptsovits_available()
    kokoro = adapter.check_kokoro_available()

    print(f"GPT-SoVITS 可用: {gptsovits}")
    print(f"Kokoro 可用: {kokoro}")

    if not gptsovits and not kokoro:
        print("跳过: 没有可用的 TTS 引擎")
        return None

    # 尝试合成
    try:
        audio_path = adapter.synthesize("你好，这是一个测试。", prefer_gptsovits=gptsovits)
        print(f"音频已保存: {audio_path}")
        return True
    except Exception as e:
        print(f"TTS 测试失败: {e}")
        return False


def test_liveportrait():
    """测试 LivePortrait"""
    print("\n" + "=" * 60)
    print("测试 5: FasterLivePortrait")
    print("=" * 60)

    # 检查源图像
    source_image = PROJECT_ROOT / "FasterLivePortrait" / "assets" / "examples" / "source" / "s12.jpg"
    if not source_image.exists():
        print("跳过: 未找到示例源图像")
        return None

    print(f"使用源图像: {source_image}")

    from digital_human.liveportrait_adapter import LivePortraitAdapter

    adapter = LivePortraitAdapter()

    # 测试源图像准备
    try:
        result = adapter.prepare_source(str(source_image))
        print(f"源图像准备: {'成功' if result else '失败'}")
        return result
    except Exception as e:
        print(f"LivePortrait 测试失败: {e}")
        return False


async def test_orchestrator():
    """测试完整编排器"""
    print("\n" + "=" * 60)
    print("测试 6: 完整编排器（文本输入）")
    print("=" * 60)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("跳过: DASHSCOPE_API_KEY 未设置")
        return None

    # 检查源图像
    source_image = PROJECT_ROOT / "FasterLivePortrait" / "assets" / "examples" / "source" / "s12.jpg"
    if not source_image.exists():
        print("跳过: 未找到示例源图像")
        return None

    from digital_human import create_orchestrator

    orchestrator = create_orchestrator(
        source_image=str(source_image),
        dashscope_api_key=api_key,
    )

    result = await orchestrator.process_text_input(
        "你好，请简单介绍一下你自己",
        use_gptsovits=False,  # 使用 Kokoro 避免依赖外部服务
    )

    print(f"成功: {result.success}")
    print(f"回复: {result.response}")
    print(f"视频: {result.video_path}")
    print(f"耗时: {result.time_cost:.2f}s")

    if result.error:
        print(f"错误: {result.error}")

    return result.success


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("  Digital Human Phase 1 - 测试套件")
    print("=" * 60)

    tests = [
        ("配置模块", test_config),
        ("SenseVoice ASR", test_sensevoice),
        ("RAG 对话", test_rag),
        ("TTS 语音合成", test_tts),
        ("LivePortrait", test_liveportrait),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = result
        except Exception as e:
            print(f"测试 {name} 异常: {e}")
            results[name] = False

    # 异步测试
    try:
        result = asyncio.run(test_orchestrator())
        results["完整编排器"] = result
    except Exception as e:
        print(f"测试 完整编排器 异常: {e}")
        results["完整编排器"] = False

    # 汇总
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)

    for name, result in results.items():
        status = "✓ 通过" if result else ("○ 跳过" if result is None else "✗ 失败")
        print(f"  {name}: {status}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print("=" * 60)
    print(f"  通过: {passed}, 失败: {failed}, 跳过: {skipped}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

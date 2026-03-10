#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Digital Human Phase 1 - CLI Entry Point

命令行入口：用于快速测试和验证数字人闭环。

使用方式：
    # 文本输入模式（调试）
    python -m digital_human.cli --mode text --source path/to/avatar.jpg --input "你好"

    # 语音输入模式
    python -m digital_human.cli --mode voice --source path/to/avatar.jpg --input path/to/audio.wav

    # 交互式模式
    python -m digital_human.cli --mode interactive --source path/to/avatar.jpg

环境变量：
    DASHSCOPE_API_KEY: 阿里云 DashScope API Key
    GPT_SOVITS_API_URL: GPT-SoVITS API 地址（可选）
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="数字人第一阶段闭环 - 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 文本输入（调试链路）
  python -m digital_human.cli --mode text --source avatar.jpg --input "你好，请介绍一下自己"

  # 语音输入（主链路）
  python -m digital_human.cli --mode voice --source avatar.jpg --input user_speech.wav

  # 交互式对话
  python -m digital_human.cli --mode interactive --source avatar.jpg
        """,
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["text", "voice", "interactive"],
        default="text",
        help="运行模式: text(文本输入), voice(语音输入), interactive(交互式)",
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="数字人源图像路径",
    )
    parser.add_argument(
        "--input", "-i",
        help="输入内容（文本或音频��件路径）",
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="输出目录（默认: ./output）",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="阿里云 DashScope API Key（也可通过 DASHSCOPE_API_KEY 环境变量设置）",
    )
    parser.add_argument(
        "--tts-url",
        dest="tts_url",
        help="GPT-SoVITS API 地址（可选，默认使用本地 Kokoro）",
    )
    parser.add_argument(
        "--no-gptsovits",
        action="store_true",
        help="禁用 GPT-SoVITS，强制使用 Kokoro",
    )

    args = parser.parse_args()

    # 检查 API Key
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置 DASHSCOPE_API_KEY 环境变量或使用 --api-key 参数")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导入模块
    print("正在初始化数字人系统...")
    from digital_human import create_orchestrator

    # 创建编排器
    orchestrator = create_orchestrator(
        source_image=args.source,
        dashscope_api_key=api_key,
        gpt_sovits_url=args.tts_url,
    )

    # 设置输出目录
    orchestrator.config.faster_live_portrait.output_dir = str(output_dir)

    # 根据模式运行
    if args.mode == "text":
        run_text_mode(orchestrator, args)
    elif args.mode == "voice":
        run_voice_mode(orchestrator, args)
    elif args.mode == "interactive":
        run_interactive_mode(orchestrator, args)


def run_text_mode(orchestrator, args):
    """文本输入模式"""
    if not args.input:
        print("错误: 文本模式需要 --input 参数")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  文本输入模式")
    print("=" * 60)
    print(f"输入文本: {args.input}")
    print(f"源图像: {args.source}")
    print("=" * 60 + "\n")

    result = orchestrator.process_text_sync(
        args.input,
        use_gptsovits=not args.no_gptsovits,
    )

    print_result(result)


def run_voice_mode(orchestrator, args):
    """语音输入模式"""
    if not args.input:
        print("错误: 语音模式需要 --input 参数（音频文件路径）")
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"错误: 音频文件不存在: {args.input}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  语音输入模式")
    print("=" * 60)
    print(f"输入音频: {args.input}")
    print(f"源图像: {args.source}")
    print("=" * 60 + "\n")

    result = orchestrator.process_voice_sync(args.input)

    print_result(result)


def run_interactive_mode(orchestrator, args):
    """交互式模式"""
    print("\n" + "=" * 60)
    print("  交互式对话模式")
    print("=" * 60)
    print(f"源图像: {args.source}")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'reset' 重置会话")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "退出"):
            print("再见！")
            break

        if user_input.lower() == "reset":
            orchestrator.reset_session()
            print("[会话已重置]\n")
            continue

        result = orchestrator.process_text_sync(
            user_input,
            use_gptsovits=not args.no_gptsovits,
        )

        if result.success:
            print(f"\nAI: {result.response}")
            print(f"[视频: {result.video_path}]\n")
        else:
            print(f"\n[错误: {result.error}]\n")


def print_result(result):
    """打印处理结果"""
    print("\n" + "=" * 60)
    if result.success:
        print("  处理成功!")
        print("=" * 60)
        if result.transcript:
            print(f"识别文本: {result.transcript}")
        if result.response:
            print(f"AI 回复: {result.response}")
        if result.video_path:
            print(f"视频输出: {result.video_path}")
        if result.audio_path:
            print(f"音频输出: {result.audio_path}")
        print(f"总耗时: {result.time_cost:.2f}s")
    else:
        print("  处理失败!")
        print("=" * 60)
        print(f"错误: {result.error}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

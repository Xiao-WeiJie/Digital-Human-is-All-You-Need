#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Digital Human 启动脚本

快速启动实时数字人服务（LiveTalking + FasterLivePortrait）。
配置优先级: 命令行参数 > digital_human/config.py
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from digital_human.config import get_default_config


def check_environment(config=None):
    """检查环境配置"""
    print("=" * 60)
    print("  Digital Human 环境检查")
    print("=" * 60)

    issues = []

    # 检查 DASHSCOPE_API_KEY（优先从配置文件读取）
    api_key = None
    if config and hasattr(config, 'psychology_rag'):
        api_key = config.psychology_rag.dashscope_api_key
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY")

    if not api_key:
        issues.append("❌ DASHSCOPE_API_KEY 未设置")
    else:
        print("✅ DASHSCOPE_API_KEY 已设置")

    # 检查 GPT-SoVITS
    gpt_sovits_url = os.environ.get("GPT_SOVITS_API_URL", "http://127.0.0.1:9880")
    try:
        import requests
        response = requests.get(f"{gpt_sovits_url}/", timeout=5)
        print(f"✅ GPT-SoVITS 服务可访问: {gpt_sovits_url}")
    except:
        print(f"⚠️ GPT-SoVITS 服务不可访问: {gpt_sovits_url} (可选)")

    # 检查 Python 版本
    py_version = sys.version_info
    if py_version >= (3, 8):
        print(f"✅ Python 版本: {py_version.major}.{py_version.minor}")
    else:
        issues.append(f"❌ Python 版本过低: {py_version.major}.{py_version.minor}，需要 >= 3.8")

    # 检查 PyTorch
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA 不可用，将使用 CPU（速度较慢）")
    except ImportError:
        issues.append("❌ PyTorch 未安装")

    # 检查关键模块
    modules = [
        ("aiohttp", "Web 服务"),
        ("aiortc", "WebRTC"),
        ("omegaconf", "配置管理"),
    ]

    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name} 依赖已安装")
        except ImportError:
            issues.append(f"❌ {name} 依赖未安装: {module}")

    print("=" * 60)

    if issues:
        print("\n⚠️ 发现以下问题：")
        for issue in issues:
            print(f"  {issue}")
        print("\n请先解决上述问题后再启动系统。")
        return False

    print("\n✅ 环境检查通过！")
    return True


def find_avatar():
    """查找默认 Avatar"""
    human_choice_dir = PROJECT_ROOT / "Human_Choice"
    if human_choice_dir.exists():
        for human_dir in human_choice_dir.iterdir():
            if human_dir.is_dir():
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    for img in human_dir.glob(ext):
                        return str(img)
    return None


def main():
    # 加载配置文件中的默认值
    config = get_default_config()
    server_config = config.server

    parser = argparse.ArgumentParser(
        description="Digital Human 启动脚本 - 实时数字人交互系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="?",  # 可选参数
        choices=["api", "cli", "check"],
        default=server_config.default_command,
        help=f"启动模式: api(实时数字人服务), cli(命令行交互), check(环境检查)，默认: {server_config.default_command}"
    )

    parser.add_argument(
        "--host",
        type=str,
        default=server_config.host,
        help=f"服务监听地址，默认: {server_config.host}"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=server_config.port,
        help=f"服务监听端口，默认: {server_config.port}"
    )

    parser.add_argument(
        "-i", "--source",
        type=str,
        default=config.default_avatar_path,
        help="源图像（Avatar）路径"
    )

    args = parser.parse_args()

    if args.command == "check":
        check_environment(config)
        return

    # 检查环境
    if server_config.auto_check_env and not check_environment(config):
        sys.exit(1)

    if args.command == "api":
        print(f"\n启动实时数字人服务: http://{args.host}:{args.port}")
        print(f"前端页面: http://localhost:{args.port}/webrtcapi.html")
        print(f"Dashboard: http://localhost:{args.port}/dashboard.html")
        print("\n按 Ctrl+C 停止服务\n")

        # 使用 LiveTalking 服务器（支持 WebRTC 实时传输）
        from live_talking_server.app import run_server
        run_server(host=args.host, port=args.port)

    elif args.command == "cli":
        # 查找 Avatar
        avatar = args.source or find_avatar()
        if not avatar:
            print("❌ 未找到 Avatar，请使用 -i 参数指定")
            sys.exit(1)

        print(f"\n使用 Avatar: {avatar}")
        print("启动命令行交互模式...\n")

        from digital_human.cli import main as cli_main
        # 重构 sys.argv 以传递给 CLI
        sys.argv = ["cli.py", "--mode", "interactive", "--source", avatar]
        cli_main()


if __name__ == "__main__":
    main()

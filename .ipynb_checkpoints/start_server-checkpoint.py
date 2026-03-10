#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Digital Human Phase 2 - Start Server

启动数字人后端服务器。

用法:
    python start_server.py                    # 默认启动
    python start_server.py --port 8080        # 指定端口
    python start_server.py --reload           # 开发模式
    python start_server.py --host 0.0.0.0     # 允许外部访问
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from digital_human.config import get_config, ServerConfig


def check_environment():
    """检查环境配置"""
    config = get_config()
    server_config = config.server

    print("=" * 60)
    print("  Digital Human Phase 2 - 环境检查")
    print("=" * 60)

    # 检查 Python 版本
    py_version = sys.version_info
    print(f"Python 版本: {py_version.major}.{py_version.minor}.{py_version.micro}")

    if py_version < (3, 8):
        print("错误: 需要 Python 3.8 或更高版本")
        sys.exit(1)

    # 检查环境变量
    dashscope_key = config.rag.dashscope_api_key
    if dashscope_key:
        print(f"DASHSCOPE_API_KEY: 已设置 ({dashscope_key[:8]}...)")
    else:
        print("警告: DASHSCOPE_API_KEY 未设置")
        print("       请设置环境变量: export DASHSCOPE_API_KEY=your-key")

    gptsovits_url = config.gpt_sovits.api_url
    if gptsovits_url:
        print(f"GPT_SOVITS_API_URL: {gptsovits_url}")
    else:
        print("GPT_SOVITS_API_URL: 未设置 (将使用默认 Kokoro TTS)")

    # 检查 Avatar 资源
    avatar_dir = server_config.avatar_dir
    if avatar_dir.exists():
        avatars = list(avatar_dir.glob("Human_*"))
        print(f"Avatar 资源目录: {avatar_dir}")
        print(f"发现 {len(avatars)} 个数字人: {[a.name for a in avatars]}")
    else:
        print(f"警告: Avatar 资源目录不存在: {avatar_dir}")

    # 检查前端
    frontend_dir = server_config.frontend_dir
    if frontend_dir.exists():
        print(f"前端目录: {frontend_dir}")
    else:
        print(f"警告: 前端目录不存在: {frontend_dir}")

    print("=" * 60)


def check_dependencies():
    """检查依赖"""
    config = get_config()

    print("\n检查依赖包...")

    required = [
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
    ]

    optional = [
        ("funasr", "SenseVoice ASR"),
        ("torch", "PyTorch"),
        ("omegaconf", "OmegaConf"),
    ]

    # 只有启用 Kokoro 时才检查
    if config.gpt_sovits.enable_kokoro_fallback:
        optional.append(("kokoro", "Kokoro TTS"))
    else:
        print("  [跳过] Kokoro TTS (已在配置中禁用)")

    missing_required = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [缺失] {pkg}")
            missing_required.append(pkg)

    for pkg, desc in optional:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg} ({desc})")
        except ImportError:
            print(f"  [可选缺失] {pkg} ({desc})")
        except Exception as e:
            # 可选包导入时有其他错误（如版本不兼容），只打印警告
            print(f"  [警告] {pkg} ({desc}) - {type(e).__name__}: {e}")

    if missing_required:
        print(f"\n请安装缺失的依赖: pip install {' '.join(missing_required)}")
        sys.exit(1)

    print()


def main():
    config = get_config()
    server_config = config.server

    parser = argparse.ArgumentParser(description="Digital Human Phase 2 Server")
    parser.add_argument("--host", default=server_config.host, help=f"服务器地址 (default: {server_config.host})")
    parser.add_argument("--port", type=int, default=server_config.port, help=f"端口号 (default: {server_config.port})")
    parser.add_argument("--reload", action="store_true", default=server_config.reload, help="开发模式（自动重载）")
    parser.add_argument("--skip-check", action="store_true", default=server_config.skip_check, help="跳过环境检查")

    args = parser.parse_args()

    # 环境检查
    if not args.skip_check:
        check_environment()
        check_dependencies()

    # 启动服务器
    print(f"\n启动服务器: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print(f"前端页面: http://{args.host}:{args.port}/")
    print()

    import uvicorn

    uvicorn.run(
        "digital_human.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

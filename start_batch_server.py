#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数字人批量视频生成服务启动脚本

启动方式:
    python start_batch_server.py
    python start_batch_server.py --port 8010
    python start_batch_server.py --host 0.0.0.0 --port 8010

访问地址:
    - 批量视频模式: http://localhost:8010/batch_video.html
    - WebRTC实时模式: http://localhost:8010/webrtcapi.html
"""

import os
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "live_talking_server"))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Digital Human Batch Video Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8010, help='Server port')
    args = parser.parse_args()

    print("=" * 60)
    print("  数字人批量视频生成服务")
    print("  Digital Human Batch Video Generation Server")
    print("=" * 60)
    print()
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print()
    print("  访问地址:")
    print(f"    - 批量视频模式: http://localhost:{args.port}/batch_video.html")
    print(f"    - WebRTC实时模式: http://localhost:{args.port}/webrtcapi.html")
    print()
    print("  环境要求:")
    print("    - DASHSCOPE_API_KEY (用于LLM)")
    print("    - FFmpeg (需在PATH中)")
    print()
    print("=" * 60)
    print()

    # 启动服务
    from live_talking_server.app import run_server
    run_server(host=args.host, port=args.port)


if __name__ == '__main__':
    main()

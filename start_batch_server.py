#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数字人批量视频生成服务启动脚本

启动方式:
    python start_batch_server.py
    python start_batch_server.py --port 8010
    python start_batch_server.py --host 0.0.0.0 --port 8010

访问地址:
    - 门户首页: http://localhost:8010/
    - 颐和缘页面: http://localhost:8010/yiheyuan.html
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
    from digital_human.config import get_default_config

    parser = argparse.ArgumentParser(description="Digital Human Batch Video Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8010, help='Server port')
    parser.add_argument('--save-videos', action='store_true',
                        help='保存生成的视频到 output 目录（默认不保存，前端播放后自动删除）')
    parser.add_argument('--motion-seed', type=int, default=10,
                        help='JoyVASA运动生成随机种子（固定种子可使相同输入生成相同表情，默认不固定）')
    parser.add_argument('--mock-mode', action='store_true',
                        help='启用 Mock 测试模式（使用固定回复和预设视频）')
    parser.add_argument('--terminal-tts-mode', action='store_true',
                        help='启用终端直输 TTS 模式（直接输入文本生成视频，不启动 Web 服务）')
    args = parser.parse_args()
    config = get_default_config()
    effective_motion_seed = args.motion_seed if args.motion_seed is not None else config.video.motion_seed

    print("=" * 60)
    if args.terminal_tts_mode:
        print("  数字人终端直输 TTS 模式")
        print("  Digital Human Terminal Direct-Text TTS Mode")
    else:
        print("  数字人批量视频生成服务")
        print("  Digital Human Batch Video Generation Server")
    print("=" * 60)
    print()
    print(f"  运动种子: {effective_motion_seed if effective_motion_seed is not None else '不固定（随机）'}")
    if args.terminal_tts_mode:
        print(f"  终端模式: 已启用")
        print(f"  默认数字人: {config.terminal_tts_avatar}")
        print(f"  输出目录: {config.output_dir}")
        print("  Web 服务: 不启动")
    else:
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"  保存视频: {'是' if args.save_videos else '否'}")
        if args.mock_mode:
            print(f"  Mock 模式: 已启用")
        print()
        print("  访问地址:")
        print(f"    - 门户首页: http://localhost:{args.port}/")
        print(f"    - 颐和缘页面: http://localhost:{args.port}/yiheyuan.html")
        print(f"    - 批量视频模式: http://localhost:{args.port}/batch_video.html")
        print(f"    - WebRTC实时模式: http://localhost:{args.port}/webrtcapi.html")
        print()
        print("  环境要求:")
        print("    - DASHSCOPE_API_KEY (用于LLM)")
        print("    - FFmpeg (需在PATH中)")
    print()
    print("=" * 60)
    print()

    if args.terminal_tts_mode:
        from live_talking_server.app import run_terminal_tts_mode
        run_terminal_tts_mode(
            host=args.host,
            port=args.port,
            motion_seed=args.motion_seed,
            avatar_id=config.terminal_tts_avatar
        )
    else:
        from live_talking_server.app import run_server
        run_server(
            host=args.host,
            port=args.port,
            save_videos=args.save_videos,
            motion_seed=args.motion_seed,
            mock_mode=args.mock_mode
        )


if __name__ == '__main__':
    main()

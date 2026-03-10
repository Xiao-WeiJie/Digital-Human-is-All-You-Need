#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PsyMind — 心理健康科普 RAG 对话系统

入口文件：启动交互式命令行对话。

使用方式:
  1. 设置环境变量: export DASHSCOPE_API_KEY="sk-your-key"
  2. 运行: python main.py

输入 'quit' 或 'exit' 退出 | 输入 'debug' 查看调试信息
"""

import uuid
import asyncio
from system import PsyMindSystem


async def main():
    """主入口：初始化系统并启动对话循环。"""

    # ── 欢迎界面 ──
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║          🌿 PsyMind — 心理健康科普对话系统                  ║")
    print("║                                                            ║")
    print("║   本系统仅用于心理健康科普教育与学术研究，                   ║")
    print("║   【不涉及任何临床诊断或治疗行为】。                        ║")
    print("║                                                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── 初始化系统 ──
    system = PsyMindSystem()

    # ── 创建对话 Session ──
    session_id = str(uuid.uuid4())
    print(f"\n  [系统] 新对话会话已创建 (Session: {session_id[:8]}...)")
    print(f"  [系统] 输入 'quit' 退出 | 'debug' 调试 | 'rebuild' 重建知识库 | 'sessions' 查看历史会话\n")

    # ── 对话循环 ──
    turn_count = 0
    while True:
        try:
            user_input = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  小暖: 再见！照顾好自己，随时欢迎回来聊天 🌟\n")
            break

        if not user_input:
            continue

        # 退出指令
        if user_input.lower() in ("quit", "exit", "退出", "再见"):
            print("\n  小暖: 很高兴和你聊天！记得照顾好自己，有需要随时来找我 🌟\n")
            break

        # 调试指令
        if user_input.lower() == "debug":
            info = system.memory.get_debug_info(session_id)
            stats = system.retriever.get_stats()
            print(f"\n  [调试] 当前轮次: {turn_count}")
            print(f"  [调试] 历史消息数: {info['message_count']}")
            print(f"  [调试] 当前摘要: {info['summary']}")
            print(f"  [调试] 用户名: {info['user_name']}")
            print(f"  [调试] Chroma 科普库: {stats.get('factual_kb_count', 0)} 块")
            print(f"  [调试] Chroma 护栏库: {stats.get('guardrail_kb_count', 0)} 块")
            print(f"  [调试] SQLite 共情库: {stats.get('empathy_examples', 0)} 条\n")
            continue

        # 重建知识库指令
        if user_input.lower() == "rebuild":
            print("\n  ⏳ 正在重建全部知识库（清空 → 重新入库）...")
            system.retriever.rebuild()
            print("  ✅ 重建完成\n")
            continue

        # 查看历史会话
        if user_input.lower() == "sessions":
            sessions = system.session_store.list_sessions()
            if sessions:
                print(f"\n  [历史会话] 共 {len(sessions)} 个:")
                for s in sessions:
                    sid = s['session_id'][:8]
                    name = s.get('user_name', '朋友')
                    updated = s.get('updated_at', '')[:16]
                    summary = s.get('summary', '') or '暂无摘要'
                    print(f"    • {sid}... ({name}) [{updated}] {summary[:40]}")
            else:
                print("\n  [历史会话] 暂无历史")
            print()
            continue

        # 处理消息
        turn_count += 1
        print()
        try:
            response = await system.process_message(user_input, session_id)
            print(f"  🌿 小暖: {response}")
        except Exception as e:
            print(f"  [系统错误] {type(e).__name__}: {e}")
            print("  小暖: 抱歉，我这边出了点小问题。你可以再说一次吗？")
        print()


if __name__ == "__main__":
    asyncio.run(main())
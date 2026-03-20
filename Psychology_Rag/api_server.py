"""
Psychology_Rag HTTP API 服务

提供心理健康对话服务的 HTTP 接口

Usage:
    python api_server.py --port 8002
"""

import os
import sys
import argparse
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import PsyMindSystem


# ────────────────────────────────────────────
# 请求/响应模型
# ────────────────────────────────────────────

class ChatRequest(BaseModel):
    """对话请求"""
    text: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """对话响应"""
    code: int = 0
    msg: str = "ok"
    data: Optional[dict] = None


# ────────────────────────────────────────────
# FastAPI 应用
# ────────────────────────────────────────────

app = FastAPI(
    title="Psychology RAG API",
    description="心理健康科普对话服务",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局系统实例
system: Optional[PsyMindSystem] = None


@app.on_event("startup")
async def startup():
    """应用启动时初始化系统"""
    global system
    print("=" * 60)
    print("  初始化 Psychology RAG 服务...")
    print("=" * 60)
    system = PsyMindSystem()
    print("=" * 60)
    print("  服务初始化完成!")
    print("=" * 60)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口

    Args:
        request: 包含 text 和 session_id

    Returns:
        response: AI回复和情感标签
    """
    if system is None:
        raise HTTPException(status_code=503, detail="系统未初始化")

    try:
        result = await system.process_message(request.text, request.session_id)

        return ChatResponse(
            data={
                "response": result["response"],
                "emotion": result.get("emotion", "neutral")
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ChatResponse(
            code=500,
            msg=str(e),
            data=None
        )


@app.post("/api/v1/chat/simple")
async def chat_simple(request: ChatRequest):
    """
    简单对话接口（只返回回复文本）

    Args:
        request: 包含 text 和 session_id

    Returns:
        response: AI回复文本
    """
    if system is None:
        raise HTTPException(status_code=503, detail="系统未初始化")

    try:
        response = await system.process_message_simple(request.text, request.session_id)
        return {"response": response}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok" if system is not None else "initializing",
        "service": "psychology_rag"
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Psychology RAG API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /api/v1/chat",
            "chat_simple": "POST /api/v1/chat/simple",
            "health": "GET /health"
        }
    }


def run_server(host: str = "0.0.0.0", port: int = 8002):
    """启动服务器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psychology RAG API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8002, help="服务器端口")

    args = parser.parse_args()

    print(f"启动 Psychology RAG 服务于 {args.host}:{args.port}")
    run_server(args.host, args.port)

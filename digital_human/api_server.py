# -*- coding: utf-8 -*-
"""
Digital Human Phase 2 - FastAPI Server

提供 REST API 和 WebSocket 接口，用于集成到其他系统。

API 端点:
- POST /text: 文本输入，返回视频
- POST /voice: 语音输入，返回视频
- GET /health: 健康检查
- POST /session/reset: 重置会话
- GET /avatars: 获取数字人列表
- POST /avatar/set: 设置当前数字人
- GET /idle/{avatar_id}: 获取待机视频
- WS /ws: WebSocket 实时通信

启动方式:
    uvicorn digital_human.api_server:app --host 0.0.0.0 --port 8000

环境变量:
    DASHSCOPE_API_KEY: 阿里云 API Key
    GPT_SOVITS_API_URL: GPT-SoVITS API 地址（可选）
    DEFAULT_SOURCE_IMAGE: 默认源图像路径（可选）
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect, Header
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 创建应用
app = FastAPI(
    title="Digital Human API",
    description="数字人第二阶段闭环 API",
    version="0.2.0",
)

# CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
orchestrator = None
avatar_manager = None
idle_manager = None
ws_handler = None
jobs = {}  # 任务状态存储


# ============================================================
# 数据模型
# ============================================================

class TextRequest(BaseModel):
    """文本输入请求"""
    text: str
    avatar_id: Optional[str] = None
    source_image: Optional[str] = None
    session_id: Optional[str] = None
    use_gptsovits: bool = True


class VoiceRequest(BaseModel):
    """语音输入请求（用于异步任务）"""
    avatar_id: Optional[str] = None
    source_image: Optional[str] = None
    session_id: Optional[str] = None


class JobStatus(BaseModel):
    """任务状态"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    result: Optional[dict] = None
    error: Optional[str] = None


class AvatarInfo(BaseModel):
    """数字人信息"""
    avatar_id: str
    name: str
    description: str = ""
    source_image: str


class SetAvatarRequest(BaseModel):
    """设置数字人请求"""
    avatar_id: str


# ============================================================
# 初始化
# ============================================================

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global avatar_manager, idle_manager, ws_handler

    print("=" * 60)
    print("  Digital Human API Phase 2 启动中...")
    print("=" * 60)

    # 检查必要的环境变量
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: DASHSCOPE_API_KEY 未设置，API 将无法正常工作")

    # 初始化 Avatar 管理器
    try:
        from digital_human.avatar_config import get_avatar_manager
        avatar_manager = get_avatar_manager()
        print(f"[启动] Avatar 管理器已初始化，发现 {len(avatar_manager.list_avatars())} 个数字人")
    except Exception as e:
        print(f"[启动] Avatar 管理器初始化失败: {e}")

    # 初始化待机管理器
    try:
        from digital_human.idle_manager import IdleStateManager
        idle_manager = IdleStateManager()
        print("[启动] 待机状态管理器已初始化")
    except Exception as e:
        print(f"[启动] 待机状态管理器初始化失败: {e}")

    # 初始化 WebSocket 处理器
    try:
        from digital_human.websocket_server import DigitalHumanWebSocketHandler
        ws_handler = DigitalHumanWebSocketHandler(
            avatar_manager=avatar_manager,
            idle_manager=idle_manager,
        )
        print("[启动] WebSocket 处理器已初始化")
    except Exception as e:
        print(f"[启动] WebSocket 处理器初始化失败: {e}")

    print("=" * 60)
    print("API 服务已启动，等待首次请求初始化模型...")
    print("=" * 60)


def get_orchestrator():
    """获取或创建 orchestrator 实例"""
    global orchestrator

    if orchestrator is None:
        from digital_human import create_orchestrator

        # 使用当前 avatar 的源图像
        default_source = None
        if avatar_manager:
            current = avatar_manager.get_current_avatar()
            if current:
                default_source = current.source_image

        if not default_source:
            default_source = os.environ.get("DEFAULT_SOURCE_IMAGE")

        orchestrator = create_orchestrator(
            source_image=default_source,
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
            gpt_sovits_url=os.environ.get("GPT_SOVITS_API_URL"),
        )

        # 设置 WebSocket 处理器的 orchestrator
        if ws_handler:
            ws_handler.orchestrator = orchestrator
            ws_handler.liveportrait = orchestrator.live_portrait if orchestrator else None

        # 设置待机管理器的 liveportrait
        if idle_manager and orchestrator:
            idle_manager.liveportrait = orchestrator.live_portrait

    return orchestrator


# ============================================================
# 静态文件和前端
# ============================================================

# 挂载前端静态文件
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
        <html>
            <head><title>Digital Human API</title></head>
            <body>
                <h1>Digital Human API Phase 2</h1>
                <p>前端文件未找到，请访问 <a href="/docs">/docs</a> 查看 API 文档</p>
            </body>
        </html>
    """)


# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
async def health_check():
    """健康检查"""
    avatars_count = len(avatar_manager.list_avatars()) if avatar_manager else 0
    return {
        "status": "healthy",
        "version": "0.2.0",
        "avatars_count": avatars_count,
    }


# ============================================================
# Avatar 管理 API
# ============================================================

@app.get("/avatars")
async def list_avatars():
    """获取所有数字人列表"""
    if not avatar_manager:
        raise HTTPException(status_code=500, detail="Avatar 管理器未初始化")

    avatars = []
    for avatar_id, avatar in avatar_manager.get_all_avatars().items():
        avatars.append({
            "avatar_id": avatar.avatar_id,
            "name": avatar.name,
            "description": avatar.description,
            "source_image": avatar.source_image,
            "has_idle_video": bool(avatar.idle_video and os.path.exists(avatar.idle_video)),
        })

    return {"avatars": avatars, "current": avatar_manager.get_current_avatar().avatar_id if avatar_manager.get_current_avatar() else None}


@app.get("/avatars/current")
async def get_current_avatar():
    """获取当前数字人"""
    if not avatar_manager:
        raise HTTPException(status_code=500, detail="Avatar 管理器未初始化")

    current = avatar_manager.get_current_avatar()
    if not current:
        raise HTTPException(status_code=404, detail="未设置当前数字人")

    return {
        "avatar_id": current.avatar_id,
        "name": current.name,
        "description": current.description,
        "source_image": current.source_image,
    }


@app.post("/avatar/set")
async def set_avatar(request: SetAvatarRequest):
    """设置当前数字人"""
    if not avatar_manager:
        raise HTTPException(status_code=500, detail="Avatar 管理器未初始化")

    avatar = avatar_manager.get_avatar(request.avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail=f"未找到数字人: {request.avatar_id}")

    avatar_manager.set_current_avatar(request.avatar_id)

    # 更新 orchestrator 的源图像
    if orchestrator:
        orchestrator.set_source_image(avatar.source_image)

    return {
        "success": True,
        "avatar_id": avatar.avatar_id,
        "name": avatar.name,
        "source_image": avatar.source_image,
    }


@app.get("/avatar/{avatar_id}")
async def get_avatar(avatar_id: str):
    """获取指定数字人信息"""
    if not avatar_manager:
        raise HTTPException(status_code=500, detail="Avatar 管理器未初始化")

    avatar = avatar_manager.get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail=f"未找到数字人: {avatar_id}")

    return {
        "avatar_id": avatar.avatar_id,
        "name": avatar.name,
        "description": avatar.description,
        "source_image": avatar.source_image,
        "tts_profile": avatar.tts_profile.to_dict() if avatar.tts_profile else None,
    }


# ============================================================
# 待机状态 API
# ============================================================

@app.get("/idle/{avatar_id}")
async def get_idle_video(avatar_id: str):
    """获取待机视频（优先返回 Human_Choice/Human_X/idle.mp4）"""
    if not avatar_manager:
        raise HTTPException(status_code=500, detail="Avatar 管理器未初始化")

    avatar = avatar_manager.get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail=f"未找到数字人: {avatar_id}")

    # 优先返回预置待机视频
    if avatar.idle_video and os.path.exists(avatar.idle_video):
        ext = Path(avatar.idle_video).suffix.lower()
        media_type = "video/webm" if ext == ".webm" else "video/mp4"
        return FileResponse(
            avatar.idle_video,
            media_type=media_type,
            filename=f"idle_{avatar_id}{ext}",
        )

    # 再尝试 idle_manager 缓存（运行时生成的）
    if idle_manager:
        video_path = idle_manager.get_idle_video(avatar_id)
        if video_path and os.path.exists(video_path):
            return FileResponse(
                video_path,
                media_type="video/mp4",
                filename=f"idle_{avatar_id}.mp4",
            )

    raise HTTPException(
        status_code=404,
        detail=f"待机视频不可用，请在 digital_human/Human_Choice/{avatar_id.replace('human_', 'Human_')}/idle.mp4 放置待机视频",
    )


@app.get("/state")
async def get_current_state():
    """获取当前状态"""
    if not idle_manager:
        return {"state": "unknown"}

    return idle_manager.to_dict()


# ============================================================
# 文本处理 API
# ============================================================

@app.post("/text")
async def process_text(
    text: str = Form(...),
    avatar_id: Optional[str] = Form(None),
    source_image: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    use_gptsovits: bool = Form(True),
):
    """
    文本输入处理

    Args:
        text: 输入文本
        avatar_id: 数字人 ID（可选）
        source_image: 源图像文件（可选，优先使用 avatar_id）
        session_id: 会话 ID（可选）
        use_gptsovits: 是否使用 GPT-SoVITS（默认 True）

    Returns:
        视频文件或 JSON 结果
    """
    orch = get_orchestrator()

    # 确定源图像
    source_path = None

    if avatar_id and avatar_manager:
        avatar = avatar_manager.get_avatar(avatar_id)
        if avatar:
            source_path = avatar.source_image
            # 更新当前 avatar
            avatar_manager.set_current_avatar(avatar_id)
            orch.set_source_image(source_path)

    if not source_path and source_image and source_image.filename:
        # 保存上传的图像
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, source_image.filename)
        with open(source_path, "wb") as f:
            f.write(await source_image.read())

    try:
        # 更新状态
        if idle_manager:
            idle_manager.start_thinking()

        result = await orch.process_text_input(
            text,
            source_image=source_path,
            session_id=session_id,
            use_gptsovits=use_gptsovits,
        )

        if not result.success:
            if idle_manager:
                idle_manager.set_error(result.error)
            raise HTTPException(status_code=500, detail=result.error)

        # 完成后回到待机
        if idle_manager:
            idle_manager.stop_talking()

        # 返回视频文件
        return FileResponse(
            result.video_path,
            media_type="video/mp4",
            filename=f"output_{uuid.uuid4().hex[:8]}.mp4",
        )

    except HTTPException:
        raise
    except Exception as e:
        if idle_manager:
            idle_manager.set_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text/json")
async def process_text_json(request: TextRequest):
    """
    文本输入处理（JSON 接口，返回元数据而非文件）

    Args:
        request: 文本请求

    Returns:
        JSON 结果（包含视频路径、音频路径、识别文本等）
    """
    orch = get_orchestrator()

    # 确定源图像
    source_path = request.source_image

    if request.avatar_id and avatar_manager:
        avatar = avatar_manager.get_avatar(request.avatar_id)
        if avatar:
            source_path = avatar.source_image
            avatar_manager.set_current_avatar(request.avatar_id)
            orch.set_source_image(source_path)

    try:
        if idle_manager:
            idle_manager.start_thinking()

        result = await orch.process_text_input(
            request.text,
            source_image=source_path,
            session_id=request.session_id,
            use_gptsovits=request.use_gptsovits,
        )

        if idle_manager:
            if result.success:
                idle_manager.stop_talking()
            else:
                idle_manager.set_error(result.error)

        return JSONResponse({
            "success": result.success,
            "video_path": result.video_path,
            "audio_path": result.audio_path,
            "transcript": result.transcript,
            "response": result.response,
            "time_cost": result.time_cost,
            "error": result.error,
        })

    except Exception as e:
        if idle_manager:
            idle_manager.set_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 语音处理 API
# ============================================================

@app.post("/voice")
async def process_voice(
    audio: UploadFile = File(...),
    avatar_id: Optional[str] = Form(None),
    source_image: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
):
    """
    语音输入处理

    Args:
        audio: 音频文件
        avatar_id: 数字人 ID（可选）
        source_image: 源图像文件（可选）
        session_id: 会话 ID（可选）

    Returns:
        视频文件
    """
    orch = get_orchestrator()

    # 保存上传的音频
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, audio.filename or "input.wav")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # 确定源图像
    source_path = None

    if avatar_id and avatar_manager:
        avatar = avatar_manager.get_avatar(avatar_id)
        if avatar:
            source_path = avatar.source_image
            avatar_manager.set_current_avatar(avatar_id)
            orch.set_source_image(source_path)

    if not source_path and source_image and source_image.filename:
        source_path = os.path.join(temp_dir, source_image.filename)
        with open(source_path, "wb") as f:
            f.write(await source_image.read())

    try:
        if idle_manager:
            idle_manager.start_listening()

        result = await orch.process_voice_input(
            audio_path,
            source_image=source_path,
            session_id=session_id,
        )

        if not result.success:
            if idle_manager:
                idle_manager.set_error(result.error)
            raise HTTPException(status_code=500, detail=result.error)

        if idle_manager:
            idle_manager.stop_talking()

        # 返回视频文件
        return FileResponse(
            result.video_path,
            media_type="video/mp4",
            filename=f"output_{uuid.uuid4().hex[:8]}.mp4",
        )

    except HTTPException:
        raise
    except Exception as e:
        if idle_manager:
            idle_manager.set_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 会话管理 API
# ============================================================

@app.post("/session/reset")
async def reset_session():
    """重置会话"""
    orch = get_orchestrator()
    new_session_id = orch.reset_session()
    return {"session_id": new_session_id}


@app.post("/source/set")
async def set_source_image(
    source_image: UploadFile = File(...),
):
    """
    设置默认源图像

    Args:
        source_image: 源图像文件
    """
    orch = get_orchestrator()

    # 保存图像
    save_dir = Path("./uploads")
    save_dir.mkdir(exist_ok=True)

    image_path = save_dir / f"source_{uuid.uuid4().hex[:8]}{Path(source_image.filename).suffix}"
    with open(image_path, "wb") as f:
        f.write(await source_image.read())

    orch.set_source_image(str(image_path))

    return {"source_image": str(image_path)}


# ============================================================
# WebSocket 端点
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 实时通信端点"""
    if not ws_handler:
        await websocket.accept()
        await websocket.send_json({"type": "error", "data": {"error": "WebSocket 处理器未初始化"}})
        await websocket.close()
        return

    # 确保 orchestrator 初始化
    get_orchestrator()

    # 使用 WS 处理器处理连接
    await ws_handler.handle_connection(websocket)


# ============================================================
# 文件服务 API
# ============================================================

from fastapi.responses import StreamingResponse
import re

@app.get("/video/{path:path}")
async def get_video(path: str, range: Optional[str] = Header(None)):
    """
    获取视频文件（支持 Range 请求）

    Range 请求允许浏览器只请求视频的部分内容，这对于视频流播放至关重要。
    如果不支持 Range 请求，浏览器可能无法正确播放长视频。
    """
    # 解码路径
    import urllib.parse
    decoded_path = urllib.parse.unquote(path)

    # 移除开头的斜杠（如果有双斜杠的情况）
    if decoded_path.startswith("/"):
        decoded_path = decoded_path[1:] if len(decoded_path) > 1 else decoded_path

    # 处理绝对路径（以 /root 或盘符开头的路径）
    # 原始路径可能是 /root/autodl-tmp/...，需要保留
    if path.startswith("/") or path.startswith("%2F"):
        # 这是绝对路径，直接使用解码后的路径
        video_path = "/" + decoded_path if not decoded_path.startswith("/") else decoded_path
    else:
        video_path = decoded_path

    if not os.path.exists(video_path):
        print(f"[Video] 文件不存在: {video_path}")
        raise HTTPException(status_code=404, detail=f"文件不存在: {video_path}")

    file_size = os.path.getsize(video_path)
    file_ext = Path(video_path).suffix.lower()

    # 根据扩展名设置 MIME 类型
    mime_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
    }
    media_type = mime_types.get(file_ext, "video/mp4")

    # 如果没有 Range 请求，返回完整文件
    if not range:
        return FileResponse(
            video_path,
            media_type=media_type,
        )

    # 解析 Range 请求头
    # 格式: bytes=start-end 或 bytes=start-
    range_match = re.match(r"bytes=(\d+)-(\d*)", range)
    if not range_match:
        return FileResponse(
            video_path,
            media_type=media_type,
        )

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

    # 确保范围有效
    if start >= file_size:
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")

    end = min(end, file_size - 1)
    chunk_size = end - start + 1

    # 定义生成器函数来流式传输文件块
    def iterfile():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                read_size = min(8192 * 1024, remaining)  # 8MB chunks
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    # 返回部分内容响应
    from fastapi import Response
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
    }

    return StreamingResponse(
        iterfile(),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )


@app.get("/image/{path:path}")
async def get_image(path: str):
    """获取图像文件"""
    import urllib.parse
    decoded_path = urllib.parse.unquote(path)

    if not os.path.exists(decoded_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    # 判断文件类型
    ext = Path(decoded_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")

    return FileResponse(decoded_path, media_type=media_type)


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Digital Human API Server")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--reload", action="store_true", help="开发模式（自动重载）")

    args = parser.parse_args()

    uvicorn.run(
        "digital_human.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

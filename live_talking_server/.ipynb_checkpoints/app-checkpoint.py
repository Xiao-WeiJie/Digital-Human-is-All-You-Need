###############################################################################
#  LiveTalking Server for Digital Human
#  整合 LiveTalking 的 WebRTC 实时传输与 FasterLivePortrait 数字人生成
#  新增: 方案B 批量视频生成 API
###############################################################################

import os
import sys
import json
import asyncio
import random
import argparse
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch.multiprocessing as mp
from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

# 导入 LiveTalking 模块
from webrtc import HumanPlayer
from basereal import BaseReal
from logger import logger

# 导入统一配置
from digital_human.config import (
    get_default_config,
    DEFAULT_PATHS,
    DigitalHumanConfig
)

# 全局状态
nerfreals: Dict[int, BaseReal] = {}
pcs = set()
opt = None
model = None
avatar = None

# 批量生成相关
batch_pipeline_manager = None
video_output_dir = None
app_config: DigitalHumanConfig = None

# Mock 模式相关
mock_mode_enabled = False
MOCK_RESPONSES = [
    "你好呀，我是小暖，是你的心理陪护助手。就像一个愿意倾听的朋友，也懂一些心理学的知识。你可以和我聊聊你的想法和感受，我会一直在这里陪着你。最近过得怎么样呀？",
    "听到你这样说，我明白你现在可能正被一些懊恼和自责的情绪包围着。每个人都会有状态不佳的时候，这很正常，就像天气会有阴晴圆缺一样。你愿意跟我聊聊具体发生了什么吗？有时候把心里的想法说出来，会让自己感觉轻松一些。",
    "听到你这样说，我真的很心疼。不小心摔碎碗这样的小事，却让你感到这么难过，说明你对自己有很高的期待，也一定是个很细心的人。其实每个人都会有手滑的时候，这并不代表你没用，只是生活中的小意外罢了。\n你知道吗？心理学上有个概念叫'全有或全无思维'，就是当我们把事情看成非黑即白，一旦出现一点差错就觉得整个世界都崩塌了。但其实，生活是有很多灰色地带的，一次小失误并不会定义你的价值。\n你愿意和我聊聊，当时摔碎碗时具体是什么感觉吗？",
    "听起来那个碗对你来说真的很特别呢。喜欢的东西突然不见了，确实会让人感到惋惜。你愿意和我聊聊它为什么那么特别吗？"
]

WEB_DIR = Path(__file__).parent / "web"


def randN(N) -> int:
    """生成长度为 N 的随机数"""
    min_val = pow(10, N - 1)
    max_val = pow(10, N)
    return random.randint(min_val, max_val - 1)


def build_nerfreal(sessionid: int) -> BaseReal:
    """构建 FasterLivePortrait 实例"""
    opt.sessionid = sessionid
    from fasterlivereal import FasterLivePortraitReal
    nerfreal = FasterLivePortraitReal(opt, model, avatar)
    return nerfreal


# ==================== API 路由 ====================

async def offer(request):
    """WebRTC offer 处理"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = randN(6)
    nerfreals[sessionid] = None
    logger.info('sessionid=%d, session num=%d', sessionid, len(nerfreals))

    # 在线程池中构建 nerfreal
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    # 配置 ICE 服务器
    ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            if sessionid in nerfreals:
                del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            if sessionid in nerfreals:
                del nerfreals[sessionid]

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # 配置视频编码
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "sessionid": sessionid
        }),
    )


async def human(request):
    """处理文本/语音消息"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)

        # 检查会话是否存在
        if sessionid not in nerfreals or nerfreals[sessionid] is None:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "请先点击'开始'按钮建立连接"}),
            )

        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()

        if params['type'] == 'echo':
            nerfreals[sessionid].put_msg_txt(params['text'])
        elif params['type'] == 'chat':
            # 使用 LLM 生成回复
            from llm import llm_response
            asyncio.get_event_loop().run_in_executor(
                None, llm_response, params['text'], nerfreals[sessionid]
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"}),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)

        # 检查会话是否存在
        if sessionid not in nerfreals or nerfreals[sessionid] is None:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "会话不存在，请先建立连接"}),
            )

        nerfreals[sessionid].flush_talk()

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"}),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


async def humanaudio(request):
    """处理上传的音频"""
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        if sessionid not in nerfreals or nerfreals[sessionid] is None:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "Session not found"}),
            )

        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"}),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


async def is_speaking(request):
    """检查是否在说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)

        if sessionid not in nerfreals or nerfreals[sessionid] is None:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "data": False}),
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": 0,
                "data": nerfreals[sessionid].is_speaking()
            }),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


async def list_avatars(request):
    """列出所有可用的数字人形象"""
    global app_config

    try:
        avatars = []
        for avatar_info in app_config.avatar.list_avatars():
            avatar_data = {
                "id": avatar_info.id,
                "name": avatar_info.name,
                "description": avatar_info.description,
                "source_image": avatar_info.get_source_image_url()
            }
            if avatar_info.idle_video:
                avatar_data["idle_video"] = avatar_info.get_idle_video_url()
            avatars.append(avatar_data)

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": 0,
                "data": {
                    "avatars": avatars,
                    "default": app_config.avatar.default_avatar
                }
            }),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


async def switch_avatar(request):
    """切换数字人形象"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        avatar_id = params.get('avatar_id', '')

        if not avatar_id:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "avatar_id is required"}),
            )

        if sessionid not in nerfreals or nerfreals[sessionid] is None:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "Session not found"}),
            )

        success = nerfreals[sessionid].switch_avatar(avatar_id)

        if success:
            logger.info(f"Switched to avatar: {avatar_id} for session: {sessionid}")
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"avatar_id": avatar_id}}),
            )
        else:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"Failed to switch to avatar: {avatar_id}"}),
            )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


async def on_shutdown(app):
    """关闭时清理"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    # 清理批量流水线
    global batch_pipeline_manager
    if batch_pipeline_manager:
        batch_pipeline_manager.cleanup_all()


# ==================== 情绪识别 API ====================

async def detect_emotion(request):
    """
    单帧情绪检测

    请求方式: POST (application/json)
    参数:
        - image: Base64 编码的图像（支持 data:image/xxx;base64, 前缀）
        - session_id: 会话 ID（可选，默认 "default"）

    返回:
        {
            "code": 0,
            "data": {
                "dominant_emotion": "happy",
                "emotion_scores": {"happy": 85.5, "sad": 5.2, ...},
                "sad_score": 5.2,
                "risk_level": "normal",
                "confidence": 85.5,
                "timestamp": 1234567890.123
            }
        }
    """
    try:
        params = await request.json()
        base64_image = params.get("image", "")
        session_id = params.get("session_id", "default")

        if not base64_image:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "image parameter is required"}),
            )

        # 获取情绪适配器
        from digital_human import get_emotion_adapter
        adapter = get_emotion_adapter()

        if not adapter.enabled:
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": 0,
                    "data": {
                        "dominant_emotion": "neutral",
                        "emotion_scores": {},
                        "sad_score": 0.0,
                        "risk_level": "normal",
                        "confidence": 0.0,
                        "timestamp": time.time(),
                        "disabled": True,
                    }
                }),
            )

        # 执行检测
        result = adapter.detect_from_base64(base64_image, session_id)

        if result is None:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "Emotion detection failed"}),
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "data": result.to_dict()}),
        )

    except Exception as e:
        logger.exception(f"[EmotionDetect] Error: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


async def get_emotion_context(request):
    """
    获取情绪上下文

    请求方式: GET
    参数:
        - session_id: 会话 ID（URL 参数）

    返回:
        {
            "code": 0,
            "data": {
                "has_data": true,
                "dominant_emotions": {"happy": 5, "neutral": 3},
                "avg_sad_score": 15.5,
                "latest_risk_level": "normal",
                "trend": "stable",
                "sample_count": 8,
                "prompt_context": "[用户情绪状态]\n当前主要情绪: happy\n"
            }
        }
    """
    try:
        session_id = request.query.get("session_id", "default")

        from digital_human import get_emotion_adapter
        adapter = get_emotion_adapter()

        summary = adapter.get_context(session_id)
        prompt_context = adapter.get_prompt_context(session_id)

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": 0,
                "data": {
                    **summary,
                    "prompt_context": prompt_context,
                }
            }),
        )

    except Exception as e:
        logger.exception(f"[EmotionContext] Error: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


# ==================== Mock 模式 API ====================

async def get_mock_config(request):
    """
    获取 Mock 模式配置

    返回:
        {
            "code": 0,
            "data": {
                "enabled": true,
                "responses": [...],
                "videos": ["/mock_videos/1.mp4", ...]
            }
        }
    """
    global mock_mode_enabled

    if not mock_mode_enabled:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "data": {"enabled": False}}),
        )

    mock_videos = [f"/mock_videos/{i}.mp4" for i in range(1, 5)]

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "code": 0,
            "data": {
                "enabled": True,
                "responses": MOCK_RESPONSES,
                "videos": mock_videos
            }
        }),
    )


async def serve_mock_video(request):
    """提供 Mock 视频文件服务"""
    filename = request.match_info.get('filename', '')

    if not filename or '..' in filename or '/' in filename or '\\' in filename:
        return web.Response(status=400, text="Invalid filename")

    mock_video_dir = PROJECT_ROOT / "output"
    video_path = mock_video_dir / filename

    if not video_path.exists():
        return web.Response(status=404, text=f"Mock video not found: {filename}")

    return web.FileResponse(video_path, headers={
        'Content-Type': 'video/mp4',
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=3600'
    })


# ==================== 批量视频生成 API ====================

async def batch_generate(request):
    """
    批量生成数字人回复视频（方案B）

    请求方式: POST (multipart/form-data)
    参数:
        - text: 用户输入文本（必填，或提供 audio）
        - audio: 用户语音文件（可选，与 text 二选一）
        - source_image: 数字人ID（默认: human_1）
        - session_id: 会话ID（可选）

    返回:
        {
            "code": 0,
            "msg": "ok",
            "data": {
                "video_url": "/videos/xxx.mp4",
                "response_text": "数字人回复内容",
                "user_text": "用户输入",
                "total_time": 20.5,
                "metrics": {...},
                "audio_duration": 8.0
            }
        }
    """
    global batch_pipeline_manager, video_output_dir, app_config

    if batch_pipeline_manager is None:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "Batch pipeline not initialized"}),
        )

    start_time = time.time()

    try:
        # 解析请求
        reader = await request.multipart()

        text = None
        audio_path = None
        source_image_id = app_config.avatar.default_avatar
        session_id = f"batch_{int(time.time() * 1000)}"

        async for field in reader:
            if field.name == "text":
                text = (await field.read()).decode('utf-8')
            elif field.name == "audio":
                # 保存音频文件
                audio_data = await field.read()
                audio_path = tempfile.mktemp(suffix='.wav')
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            elif field.name == "source_image":
                source_image_id = (await field.read()).decode('utf-8')
            elif field.name == "session_id":
                session_id = (await field.read()).decode('utf-8')

        # 验证输入
        if not text and not audio_path:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "请提供文本或语音输入"}),
            )

        # 获取源图像路径
        avatar_info = app_config.avatar.get_avatar(source_image_id)
        if not avatar_info:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"数字人形象不存在: {source_image_id}"}),
            )

        source_path = avatar_info.get_full_source_path()
        if not source_path or not os.path.exists(source_path):
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"数字人源图像不存在: {source_image_id}"}),
            )

        logger.info(f"[BatchGenerate] Processing: text={text[:20] if text else 'audio'}, avatar={source_image_id}")

        # 调用批量流水线
        result = await batch_pipeline_manager.process(
            session_id=session_id,
            user_text=text,
            user_audio_path=audio_path,
            source_image=source_path,
            avatar_id=source_image_id  # 传递 avatar_id，确保 TTS 使用正确的参考音频
        )

        # 清理临时音频
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

        if not result.success:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": result.error_message}),
            )

        logger.info(f"[BatchGenerate] Completed in {result.total_time:.2f}s")

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": 0,
                "msg": "ok",
                "data": {
                    "video_url": result.video_url,
                    "response_text": result.response_text,
                    "user_text": result.user_text,
                    "total_time": result.total_time,
                    "metrics": result.metrics,
                    "audio_duration": result.audio_duration
                }
            }),
        )

    except Exception as e:
        logger.exception(f"[BatchGenerate] Error: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


async def serve_video(request):
    """提供视频文件服务"""
    filename = request.match_info.get('filename', '')

    if not filename:
        return web.Response(status=400, text="Filename required")

    # 安全检查：防止路径遍历
    if '..' in filename or '/' in filename or '\\' in filename:
        return web.Response(status=400, text="Invalid filename")

    global video_output_dir
    if video_output_dir is None:
        return web.Response(status=404, text="Video directory not configured")

    video_path = Path(video_output_dir) / filename

    if not video_path.exists():
        return web.Response(status=404, text="Video not found")

    return web.FileResponse(
        video_path,
        headers={
            'Content-Type': 'video/mp4',
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'no-cache'
        }
    )


async def delete_video(request):
    """
    删除已播放完成的视频文件

    请求方式: DELETE
    参数: filename (URL路径参数)

    返回:
        {"code": 0, "msg": "ok"} 或错误信息
    """
    global app_config

    # 如果配置了保存视频，则跳过删除
    if app_config and app_config.save_generated_videos:
        logger.info(f"[VideoCleanup] Skipped deletion (save mode): {request.match_info.get('filename', '')}")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "Video saved (deletion skipped)"}),
        )

    filename = request.match_info.get('filename', '')

    if not filename:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "Filename required"}),
        )

    # 安全检查：防止路径遍历
    if '..' in filename or '/' in filename or '\\' in filename:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "Invalid filename"}),
        )

    global video_output_dir
    if video_output_dir is None:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "Video directory not configured"}),
        )

    video_path = Path(video_output_dir) / filename

    # 确保只删除 output 目录下的文件
    try:
        video_path.resolve().relative_to(Path(video_output_dir).resolve())
    except ValueError:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "Invalid path"}),
        )

    if not video_path.exists():
        # 文件已不存在，视为成功
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "Video already deleted"}),
        )

    try:
        os.remove(video_path)
        logger.info(f"[VideoCleanup] Deleted video: {filename}")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"}),
        )
    except Exception as e:
        logger.error(f"[VideoCleanup] Failed to delete {filename}: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(e)}),
        )


# ==================== 主函数 ====================

def _serve_web_page(filename: str) -> web.FileResponse:
    """Serve a page from the bundled web directory."""
    page_path = WEB_DIR / filename
    if not page_path.exists():
        raise web.HTTPNotFound(text=f"Page not found: {filename}")
    return web.FileResponse(page_path)


async def serve_root(request):
    raise web.HTTPFound("/xinyu_complete.html")


async def serve_portal_page(request):
    return _serve_web_page("xinyu_complete.html")


async def serve_brief_page(request):
    return _serve_web_page("brief.html")


async def serve_batch_video_page(request):
    return _serve_web_page("batch_video.html")


async def serve_legacy_dashboard_page(request):
    raise web.HTTPFound("/xinyu_complete.html")


async def serve_legacy_webrtc_page(request):
    raise web.HTTPFound("/batch_video.html")


def load_models_and_avatar(opt):
    """加载模型和 Avatar"""
    global model, avatar

    logger.info("Loading FasterLivePortrait models...")
    from fasterlivereal import load_model, load_avatar, warm_up

    model = load_model()
    avatar = load_avatar(opt.avatar_id, model[0])
    warm_up(opt.batch_size, model[0], avatar)

    logger.info("Models and avatar loaded successfully")


@web.middleware
async def html_no_cache_middleware(request, handler):
    response = await handler(request)
    if request.path.endswith('.html') or request.path == '/':
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


def create_app(opt):
    """创建 aiohttp 应用"""
    appasync = web.Application(client_max_size=1024**2 * 100, middlewares=[html_no_cache_middleware])
    appasync.on_shutdown.append(on_shutdown)

    # WebRTC API 路由
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/list_avatars", list_avatars)
    appasync.router.add_post("/switch_avatar", switch_avatar)

    # 批量视频生成 API（方案B）
    appasync.router.add_post("/api/v1/generate", batch_generate)
    appasync.router.add_get("/videos/{filename}", serve_video)
    appasync.router.add_delete("/api/v1/video/{filename}", delete_video)

    # 情绪识别 API
    appasync.router.add_post("/api/v1/emotion/detect", detect_emotion)
    appasync.router.add_get("/api/v1/emotion/context", get_emotion_context)

    # Mock 模式 API
    appasync.router.add_get("/api/v1/mock/config", get_mock_config)
    appasync.router.add_get("/mock_videos/{filename}", serve_mock_video)

    # 静态文件服务
    web_dir = Path(__file__).parent / "web"
    appasync.router.add_get("/", serve_root)
    appasync.router.add_get("/xinyu_complete.html", serve_portal_page)
    appasync.router.add_get("/brief.html", serve_brief_page)
    appasync.router.add_get("/batch_video.html", serve_batch_video_page)
    appasync.router.add_get("/dashboard.html", serve_legacy_dashboard_page)
    appasync.router.add_get("/webrtcapi.html", serve_legacy_webrtc_page)
    appasync.router.add_static('/', path=str(web_dir))

    # Human_Choice 目录
    human_choice_path = DEFAULT_PATHS["human_choice"]
    if Path(human_choice_path).exists():
        appasync.router.add_static('/human_choice', path=human_choice_path)

    # CORS 配置
    cors = aiohttp_cors.setup(appasync, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(appasync.router.routes()):
        cors.add(route)

    return appasync


def _build_runtime_opt(config: DigitalHumanConfig, port: int):
    """构建兼容旧代码的运行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=config.video.fps)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)
    parser.add_argument('--W', type=int, default=config.video.width)
    parser.add_argument('--H', type=int, default=config.video.height)
    parser.add_argument('--avatar_id', type=str, default=config.avatar.default_avatar)
    parser.add_argument('--batch_size', type=int, default=config.video.batch_size)
    parser.add_argument('--customvideo_config', type=str, default='')
    parser.add_argument('--tts', type=str, default='kokoro')
    parser.add_argument('--REF_FILE', type=str, default=config.gpt_sovits.ref_audio)
    parser.add_argument('--REF_TEXT', type=str, default=config.gpt_sovits.ref_text)
    parser.add_argument('--TTS_SERVER', type=str, default=config.gpt_sovits.server_url)
    parser.add_argument('--model', type=str, default='fasterliveportrait')
    parser.add_argument('--transport', type=str, default='webrtc')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=port)

    args = parser.parse_args([])
    args.listenport = port
    args.sessionid = 0
    args.customopt = []
    return args


def initialize_batch_runtime(
    host: str = "0.0.0.0",
    port: int = 8010,
    save_videos: bool = False,
    motion_seed: int = None,
    mock_mode: bool = False,
    terminal_tts_mode: bool = False,
    terminal_tts_avatar: str = None
):
    """初始化批量生成运行时资源"""
    global opt, batch_pipeline_manager, video_output_dir, app_config, mock_mode_enabled

    mock_mode_enabled = mock_mode

    mp.set_start_method('spawn', force=True)

    app_config = get_default_config()
    app_config.host = host
    app_config.port = port
    app_config.save_generated_videos = save_videos
    app_config.terminal_tts_mode = terminal_tts_mode
    if terminal_tts_avatar:
        app_config.terminal_tts_avatar = terminal_tts_avatar
    if motion_seed is not None:
        app_config.video.motion_seed = motion_seed

    opt = _build_runtime_opt(app_config, port)

    load_models_and_avatar(opt)

    logger.info("Initializing batch pipeline...")
    from batch_pipeline import batch_pipeline_manager as _batch_manager
    batch_pipeline_manager = _batch_manager

    video_output_dir = app_config.output_dir
    os.makedirs(video_output_dir, exist_ok=True)

    default_avatar = app_config.avatar.get_default_avatar()
    default_source = default_avatar.get_full_source_path() if default_avatar else None

    batch_pipeline_manager.initialize(
        pipeline=model[0],
        joyvasa_pipeline=model[1],
        config=app_config,
        default_source_image=default_source
    )

    logger.info("Batch pipeline initialized")

    # 打印视频保存配置
    if app_config.save_generated_videos:
        logger.info(f"[Config] 视频保存模式: 已启用，视频将保存到 {video_output_dir}")
    else:
        logger.info("[Config] 视频保存模式: 未启用，视频播放后自动删除")

    return {
        "default_avatar": default_avatar,
        "default_source": default_source,
        "video_output_dir": video_output_dir,
        "config": app_config
    }


def _resolve_avatar_source_path(avatar_id: str) -> str:
    """解析数字人源图像路径"""
    global app_config

    avatar_info = app_config.avatar.get_avatar(avatar_id) if app_config else None
    if avatar_info is None:
        raise RuntimeError(f"数字人形象不存在: {avatar_id}")

    source_path = avatar_info.get_full_source_path()
    if not source_path or not os.path.exists(source_path):
        raise RuntimeError(f"数字人源图像不存在: {avatar_id}")

    return source_path


def run_server(host: str = "0.0.0.0", port: int = 8010, save_videos: bool = False, motion_seed: int = None, mock_mode: bool = False, **kwargs):
    """运行服务器"""
    initialize_batch_runtime(
        host=host,
        port=port,
        save_videos=save_videos,
        motion_seed=motion_seed,
        mock_mode=mock_mode,
        terminal_tts_mode=False
    )

    # 创建应用
    appasync = create_app(opt)

    logger.info(f'Starting server at http://{host}:{port}/')
    logger.info(f'WebRTC 实时模式: http://{host}:{port}/webrtcapi.html')
    logger.info(f'批量视频模式: http://{host}:{port}/batch_video.html')
    logger.info(f'Dashboard: http://{host}:{port}/dashboard.html')

    # 运行服务器
    web.run_app(appasync, host=host, port=port)


def run_terminal_tts_mode(
    host: str = "0.0.0.0",
    port: int = 8010,
    motion_seed: int = None,
    avatar_id: str = "human_1"
):
    """运行终端直输 TTS 模式"""
    global batch_pipeline_manager, video_output_dir

    initialize_batch_runtime(
        host=host,
        port=port,
        save_videos=True,
        motion_seed=motion_seed,
        mock_mode=False,
        terminal_tts_mode=True,
        terminal_tts_avatar=avatar_id
    )

    source_image = _resolve_avatar_source_path(avatar_id)

    print("=" * 60)
    print("  终端直输 TTS 合成模式")
    print("  Terminal Direct-Text TTS Mode")
    print("=" * 60)
    print()
    print("  当前模式不会启动 Web 服务。")
    print(f"  默认数字人: {avatar_id}")
    print(f"  输出目录: {video_output_dir}")
    print(f"  运动种子: {motion_seed if motion_seed is not None else '不固定（随机）'}")
    print()
    print("  使用说明:")
    print("    - 直接输入要合成的文本并回车")
    print("    - 输入 exit / quit / q 退出")
    print()
    print("=" * 60)
    print()

    counter = 0

    try:
        while True:
            try:
                user_input = input("请输入要合成的文本（exit / quit / q 退出）：").strip()
            except EOFError:
                print("\n检测到输入结束，终端模式退出。")
                break

            if not user_input:
                print("输入为空，请重新输入。")
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                print("终端模式退出。")
                break

            counter += 1
            session_id = f"terminal_{int(time.time() * 1000)}_{counter}"
            print(f"[{counter}] 开始生成...")

            try:
                result = asyncio.run(
                    batch_pipeline_manager.process_direct_text(
                        session_id=session_id,
                        tts_text=user_input,
                        source_image=source_image,
                        avatar_id=avatar_id
                    )
                )

                if result.success:
                    print(f"[{counter}] 生成成功")
                    print(f"文本: {result.response_text}")
                    print(f"视频: {result.video_path}")
                    print(f"音频时长: {result.audio_duration:.2f}s")
                    print(f"总耗时: {result.total_time:.2f}s")
                else:
                    print(f"[{counter}] 生成失败: {result.error_message}")
            finally:
                if batch_pipeline_manager is not None:
                    batch_pipeline_manager.cleanup_session(session_id)
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，终端模式退出。")
    finally:
        if batch_pipeline_manager is not None:
            batch_pipeline_manager.cleanup_all()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8010)
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)

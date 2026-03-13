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
            source_image=source_path
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


# ==================== 主函数 ====================

def load_models_and_avatar(opt):
    """加载模型和 Avatar"""
    global model, avatar

    logger.info("Loading FasterLivePortrait models...")
    from fasterlivereal import load_model, load_avatar, warm_up

    model = load_model()
    avatar = load_avatar(opt.avatar_id, model[0])
    warm_up(opt.batch_size, model[0], avatar)

    logger.info("Models and avatar loaded successfully")


def create_app(opt):
    """创建 aiohttp 应用"""
    appasync = web.Application(client_max_size=1024**2 * 100)
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

    # 静态文件服务
    web_dir = Path(__file__).parent / "web"
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


def run_server(host: str = "0.0.0.0", port: int = 8010, **kwargs):
    """运行服务器"""
    global opt, batch_pipeline_manager, video_output_dir, app_config

    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)

    # 加载统一配置
    app_config = get_default_config()
    app_config.host = host
    app_config.port = port

    # 创建默认参数（兼容旧代码）
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=app_config.video.fps)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)
    parser.add_argument('--W', type=int, default=app_config.video.width)
    parser.add_argument('--H', type=int, default=app_config.video.height)
    parser.add_argument('--avatar_id', type=str, default=app_config.avatar.default_avatar)
    parser.add_argument('--batch_size', type=int, default=app_config.video.batch_size)
    parser.add_argument('--customvideo_config', type=str, default='')
    parser.add_argument('--tts', type=str, default='gpt-sovits')  # 默认使用 GPT-SoVITS
    parser.add_argument('--REF_FILE', type=str, default=app_config.gpt_sovits.ref_audio)
    parser.add_argument('--REF_TEXT', type=str, default=app_config.gpt_sovits.ref_text)
    parser.add_argument('--TTS_SERVER', type=str, default=app_config.gpt_sovits.server_url)
    parser.add_argument('--model', type=str, default='fasterliveportrait')
    parser.add_argument('--transport', type=str, default='webrtc')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=port)

    # 解析参数
    args = parser.parse_args([])
    args.listenport = port

    # 设置 session 相关属性
    args.sessionid = 0
    args.customopt = []

    opt = args

    # 加载模型和 Avatar
    load_models_and_avatar(opt)

    # ========== 初始化批量流水线 ==========
    logger.info("Initializing batch pipeline...")
    from batch_pipeline import batch_pipeline_manager as _batch_manager
    batch_pipeline_manager = _batch_manager

    # 设置视频输出目录
    video_output_dir = app_config.output_dir
    os.makedirs(video_output_dir, exist_ok=True)

    # 获取默认源图像路径
    default_avatar = app_config.avatar.get_default_avatar()
    default_source = default_avatar.get_full_source_path() if default_avatar else None

    # 初始化批量流水线管理器
    batch_pipeline_manager.initialize(
        pipeline=model[0],
        joyvasa_pipeline=model[1],
        config=app_config
    )

    logger.info("Batch pipeline initialized")

    # 创建应用
    appasync = create_app(opt)

    logger.info(f'Starting server at http://{host}:{port}/')
    logger.info(f'WebRTC 实时模式: http://{host}:{port}/webrtcapi.html')
    logger.info(f'批量视频模式: http://{host}:{port}/batch_video.html')
    logger.info(f'Dashboard: http://{host}:{port}/dashboard.html')

    # 运行服务器
    web.run_app(appasync, host=host, port=port)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8010)
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)

# -*- coding: utf-8 -*-
"""
Digital Human Phase 2 - WebSocket Server

提供实时通信能力，用于：
- 状态推送
- 流式输出
- 语音/文本交互

消息格式:
{
    "type": "state_update" | "asr_result" | "llm_response" | "tts_progress" | "video_ready" | "error",
    "data": { ... },
    "timestamp": float
}
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable

from fastapi import WebSocket, WebSocketDisconnect


class MessageType(Enum):
    """WebSocket 消息类型"""
    # 客户端 -> 服务器
    TEXT_INPUT = "text_input"
    VOICE_INPUT = "voice_input"
    SWITCH_AVATAR = "switch_avatar"
    RESET_SESSION = "reset_session"
    PING = "ping"

    # 服务器 -> 客户端
    STATE_UPDATE = "state_update"
    ASR_RESULT = "asr_result"
    LLM_RESPONSE = "llm_response"
    LLM_STREAM = "llm_stream"
    TTS_PROGRESS = "tts_progress"
    VIDEO_PROGRESS = "video_progress"
    VIDEO_READY = "video_ready"
    AVATAR_LIST = "avatar_list"
    AVATAR_SWITCHED = "avatar_switched"
    ERROR = "error"
    PONG = "pong"


@dataclass
class WSMessage:
    """WebSocket 消息"""
    type: str
    data: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "WSMessage":
        data = json.loads(json_str)
        return cls(
            type=data["type"],
            data=data.get("data", {}),
            timestamp=data.get("timestamp"),
        )


class ConnectionManager:
    """
    WebSocket 连接管理器

    管理所有客户端连接和消息广播。
    """

    def __init__(self):
        # 活跃连接: client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        # 客户端会话: client_id -> session_id
        self.client_sessions: Dict[str, str] = {}
        # 客户端当前 avatar: client_id -> avatar_id
        self.client_avatars: Dict[str, str] = {}
        # 锁
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """
        接受新连接

        Args:
            websocket: WebSocket 连接
            client_id: 客户端 ID（可选，自动生成）

        Returns:
            客户端 ID
        """
        await websocket.accept()

        if client_id is None:
            client_id = str(uuid.uuid4())[:8]

        async with self._lock:
            self.active_connections[client_id] = websocket
            self.client_sessions[client_id] = str(uuid.uuid4())

        print(f"[WS] 客户端连接: {client_id}, 当前连接数: {len(self.active_connections)}")
        return client_id

    async def disconnect(self, client_id: str):
        """断开连接"""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            if client_id in self.client_sessions:
                del self.client_sessions[client_id]
            if client_id in self.client_avatars:
                del self.client_avatars[client_id]

        print(f"[WS] 客户端断开: {client_id}, 当前连接数: {len(self.active_connections)}")

    async def send_message(self, client_id: str, message: WSMessage):
        """发送消息给指定客户端"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message.to_json())
            except Exception as e:
                print(f"[WS] 发送消息失败: {e}")
                await self.disconnect(client_id)

    async def broadcast(self, message: WSMessage, exclude: Set[str] = None):
        """广播消息给所有客户端"""
        if exclude is None:
            exclude = set()

        disconnected = []
        for client_id, websocket in self.active_connections.items():
            if client_id in exclude:
                continue
            try:
                await websocket.send_text(message.to_json())
            except Exception:
                disconnected.append(client_id)

        for client_id in disconnected:
            await self.disconnect(client_id)

    def get_session_id(self, client_id: str) -> Optional[str]:
        """获取客户端会话 ID"""
        return self.client_sessions.get(client_id)

    def get_avatar_id(self, client_id: str) -> Optional[str]:
        """获取客户端当前 avatar"""
        return self.client_avatars.get(client_id)

    def set_avatar_id(self, client_id: str, avatar_id: str):
        """设置客户端 avatar"""
        self.client_avatars[client_id] = avatar_id


class DigitalHumanWebSocketHandler:
    """
    数字人 WebSocket 处理器

    处理 WebSocket 消息并调用数字人系统。
    """

    def __init__(
        self,
        orchestrator=None,
        avatar_manager=None,
        idle_manager=None,
    ):
        self.orchestrator = orchestrator
        self.avatar_manager = avatar_manager
        self.idle_manager = idle_manager
        self.connection_manager = ConnectionManager()

        # 处理中的任务
        self._processing_tasks: Dict[str, asyncio.Task] = {}

    async def handle_connection(self, websocket: WebSocket):
        """处理 WebSocket 连接"""
        client_id = await self.connection_manager.connect(websocket)

        try:
            # 发送初始信息
            await self._send_initial_info(client_id)

            # 消息循环
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=300.0  # 5 分钟超时
                    )
                    message = WSMessage.from_json(data)
                    await self._handle_message(client_id, message)

                except asyncio.TimeoutError:
                    # 发送心跳检测
                    await self.connection_manager.send_message(
                        client_id,
                        WSMessage(MessageType.PONG.value, {})
                    )

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WS] 连接错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 取消处理中的任务
            if client_id in self._processing_tasks:
                self._processing_tasks[client_id].cancel()
                del self._processing_tasks[client_id]

            await self.connection_manager.disconnect(client_id)

    async def _send_initial_info(self, client_id: str):
        """发送初始信息"""
        # 发送 avatar 列表
        if self.avatar_manager:
            avatars = []
            for avatar_id, avatar in self.avatar_manager.get_all_avatars().items():
                avatars.append({
                    "avatar_id": avatar.avatar_id,
                    "name": avatar.name,
                    "description": avatar.description,
                })

            await self.connection_manager.send_message(
                client_id,
                WSMessage(MessageType.AVATAR_LIST.value, {"avatars": avatars})
            )

            # 设置默认 avatar
            current = self.avatar_manager.get_current_avatar()
            if current:
                self.connection_manager.set_avatar_id(client_id, current.avatar_id)
                await self.connection_manager.send_message(
                    client_id,
                    WSMessage(MessageType.AVATAR_SWITCHED.value, {
                        "avatar_id": current.avatar_id,
                        "name": current.name,
                    })
                )

        # 发送初始状态
        if self.idle_manager:
            await self.connection_manager.send_message(
                client_id,
                WSMessage(MessageType.STATE_UPDATE.value, self.idle_manager.to_dict())
            )

    async def _handle_message(self, client_id: str, message: WSMessage):
        """处理消息"""
        msg_type = message.type
        data = message.data

        try:
            if msg_type == MessageType.TEXT_INPUT.value:
                await self._handle_text_input(client_id, data)

            elif msg_type == MessageType.VOICE_INPUT.value:
                await self._handle_voice_input(client_id, data)

            elif msg_type == MessageType.SWITCH_AVATAR.value:
                await self._handle_switch_avatar(client_id, data)

            elif msg_type == MessageType.RESET_SESSION.value:
                await self._handle_reset_session(client_id)

            elif msg_type == MessageType.PING.value:
                await self.connection_manager.send_message(
                    client_id,
                    WSMessage(MessageType.PONG.value, {})
                )

            else:
                print(f"[WS] 未知消息类型: {msg_type}")

        except Exception as e:
            await self._send_error(client_id, str(e))

    async def _handle_text_input(self, client_id: str, data: dict):
        """处理文本输入"""
        text = data.get("text", "")
        if not text:
            await self._send_error(client_id, "文本不能为空")
            return

        # 获取当前 avatar
        avatar_id = self.connection_manager.get_avatar_id(client_id)
        if not avatar_id or not self.avatar_manager:
            await self._send_error(client_id, "未选择数字人")
            return

        avatar = self.avatar_manager.get_avatar(avatar_id)
        if not avatar:
            await self._send_error(client_id, f"未找到数字人: {avatar_id}")
            return

        # 更新状态
        if self.idle_manager:
            self.idle_manager.start_thinking()
            await self._broadcast_state(client_id)

        # 获取会话 ID
        session_id = self.connection_manager.get_session_id(client_id)

        # 创建处理任务
        task = asyncio.create_task(
            self._process_text_request(client_id, text, avatar, session_id)
        )
        self._processing_tasks[client_id] = task

        try:
            await task
        finally:
            if client_id in self._processing_tasks:
                del self._processing_tasks[client_id]

    async def _process_text_request(
        self,
        client_id: str,
        text: str,
        avatar,
        session_id: str,
    ):
        """处理文本请求"""
        try:
            if not self.orchestrator:
                await self._send_error(client_id, "系统未初始化")
                return

            # Step 1: RAG 对话
            await self._send_state_update(client_id, "thinking", "正在思考...")

            rag_result = await self.orchestrator.rag.chat(text, session_id)
            response = rag_result["response"]

            # 发送 LLM 响应
            await self.connection_manager.send_message(
                client_id,
                WSMessage(MessageType.LLM_RESPONSE.value, {
                    "response": response,
                    "user_input": text,
                })
            )

            # Step 2: TTS 合成
            await self._send_state_update(client_id, "talking", "正在合成语音...")

            audio_path = tempfile.mktemp(suffix=".wav")
            tts_profile = avatar.tts_profile

            # 使用 profile 参数
            audio_path = await self.orchestrator.tts.synthesize_async(
                response,
                audio_path,
                prefer_gptsovits=True,
                ref_audio_path=tts_profile.ref_audio_path,
                prompt_text=tts_profile.prompt_text,
            )

            await self.connection_manager.send_message(
                client_id,
                WSMessage(MessageType.TTS_PROGRESS.value, {
                    "status": "completed",
                    "audio_path": audio_path,
                })
            )

            # Step 3: 视频生成
            await self._send_state_update(client_id, "talking", "正在生成视频...")

            video_path = self.orchestrator.live_portrait.generate_from_audio(
                audio_path,
                avatar.source_image,
            )

            await self.connection_manager.send_message(
                client_id,
                WSMessage(MessageType.VIDEO_READY.value, {
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "response": response,
                })
            )

            # 完成后回到待机
            if self.idle_manager:
                self.idle_manager.stop_talking()
                await self._broadcast_state(client_id)

        except Exception as e:
            print(f"[WS] 处理请求错误: {e}")
            import traceback
            traceback.print_exc()
            await self._send_error(client_id, str(e))
            if self.idle_manager:
                self.idle_manager.set_error(str(e))
                await self._broadcast_state(client_id)

    async def _handle_voice_input(self, client_id: str, data: dict):
        """处理语音输入"""
        # 语音数据可能是 base64 编码
        audio_data = data.get("audio_data")
        if not audio_data:
            await self._send_error(client_id, "无音频数据")
            return

        import base64

        try:
            # 解码音频数据
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data

            # 保存临时文件
            audio_path = tempfile.mktemp(suffix=".wav")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # 获取当前 avatar
            avatar_id = self.connection_manager.get_avatar_id(client_id)
            if not avatar_id or not self.avatar_manager:
                await self._send_error(client_id, "未选择数字人")
                return

            avatar = self.avatar_manager.get_avatar(avatar_id)
            if not avatar:
                await self._send_error(client_id, f"未找到数字人: {avatar_id}")
                return

            # 更新状态
            if self.idle_manager:
                self.idle_manager.start_listening()
                await self._broadcast_state(client_id)

            # ASR
            await self._send_state_update(client_id, "listening", "正在识别语音...")

            asr_result = self.orchestrator.asr.transcribe(audio_path)
            transcript = asr_result["text"]

            # 发送 ASR 结果
            await self.connection_manager.send_message(
                client_id,
                WSMessage(MessageType.ASR_RESULT.value, {
                    "transcript": transcript,
                    "language": asr_result.get("language", "auto"),
                })
            )

            # 继续处理（类似文本输入）
            session_id = self.connection_manager.get_session_id(client_id)

            task = asyncio.create_task(
                self._process_text_request(client_id, transcript, avatar, session_id)
            )
            self._processing_tasks[client_id] = task

            try:
                await task
            finally:
                if client_id in self._processing_tasks:
                    del self._processing_tasks[client_id]

        except Exception as e:
            print(f"[WS] 语音处理错误: {e}")
            import traceback
            traceback.print_exc()
            await self._send_error(client_id, f"语音处理失败: {e}")

    async def _handle_switch_avatar(self, client_id: str, data: dict):
        """处理切换 avatar"""
        avatar_id = data.get("avatar_id")
        if not avatar_id:
            await self._send_error(client_id, "未指定数字人")
            return

        if not self.avatar_manager:
            await self._send_error(client_id, "Avatar 管理器未初始化")
            return

        avatar = self.avatar_manager.get_avatar(avatar_id)
        if not avatar:
            await self._send_error(client_id, f"未找到数字人: {avatar_id}")
            return

        # 切换 avatar
        self.avatar_manager.set_current_avatar(avatar_id)
        self.connection_manager.set_avatar_id(client_id, avatar_id)

        # 发送切换通知
        await self.connection_manager.send_message(
            client_id,
            WSMessage(MessageType.AVATAR_SWITCHED.value, {
                "avatar_id": avatar.avatar_id,
                "name": avatar.name,
                "source_image": avatar.source_image,
            })
        )

        print(f"[WS] 客户端 {client_id} 切换到数字人: {avatar_id}")

    async def _handle_reset_session(self, client_id: str):
        """处理重置会话"""
        new_session_id = str(uuid.uuid4())
        self.connection_manager.client_sessions[client_id] = new_session_id

        if self.orchestrator:
            self.orchestrator.reset_session()

        await self.connection_manager.send_message(
            client_id,
            WSMessage(MessageType.STATE_UPDATE.value, {
                "session_reset": True,
                "new_session_id": new_session_id,
            })
        )

    async def _send_state_update(self, client_id: str, state: str, message: str = ""):
        """发送状态更新"""
        await self.connection_manager.send_message(
            client_id,
            WSMessage(MessageType.STATE_UPDATE.value, {
                "state": state,
                "message": message,
            })
        )

    async def _broadcast_state(self, client_id: str):
        """广播状态更新"""
        if self.idle_manager:
            await self.connection_manager.send_message(
                client_id,
                WSMessage(MessageType.STATE_UPDATE.value, self.idle_manager.to_dict())
            )

    async def _send_error(self, client_id: str, error: str):
        """发送错误消息"""
        await self.connection_manager.send_message(
            client_id,
            WSMessage(MessageType.ERROR.value, {"error": error})
        )


# WebSocket 路由端点
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点（需要在 FastAPI 中注册）"""
    # 这些需要在 api_server.py 中设置
    from . import get_avatar_manager
    from .idle_manager import IdleStateManager

    handler = DigitalHumanWebSocketHandler(
        avatar_manager=get_avatar_manager(),
        idle_manager=IdleStateManager(),
    )

    await handler.handle_connection(websocket)

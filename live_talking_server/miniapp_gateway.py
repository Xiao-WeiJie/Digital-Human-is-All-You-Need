
import asyncio
import hashlib
import json
import logging
import math
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiohttp
from aiohttp import web

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "runtime_data"
STATE_FILE = STATE_DIR / "miniapp_gateway_state.json"
BAIDU_MAP_AK = os.getenv("BAIDU_MAP_AK", "guuwsocgqcGclZXRqmZlIcNxIL1ahnXh").strip()
BAIDU_BROWSER_AK = os.getenv("BAIDU_BROWSER_AK", "pjmgjPxmSH17UTVX9JVZjOPuboMHhblk").strip()
TICKET_BACKEND_URL = os.getenv("TICKET_BACKEND_URL", "http://127.0.0.1:3101").rstrip("/")
AUTH_TOKEN_TTL = 30 * 24 * 3600
RESET_CODE_TTL = 10 * 60
PBKDF2_ITERATIONS = 120000
STATE_LOCK = asyncio.Lock()
_STATE_CACHE: Optional[Dict[str, Any]] = None

DEFAULT_PREFERENCES = {
    "alertPush": True,
    "healthDigest": True,
    "quietMode": False,
    "mode": "family",
}

DEMO_RIDE_POINTS = [
    {
        "id": "pickup_yiheyuan_gate",
        "name": "颐和苑小区南门",
        "address": "颐和苑小区南门口",
        "lat": 29.56301,
        "lng": 106.55156,
        "tag": "上车更方便",
    },
    {
        "id": "pickup_community_center",
        "name": "颐和苑社区服务站",
        "address": "社区服务站门前临停点",
        "lat": 29.56412,
        "lng": 106.54988,
        "tag": "推荐",
    },
    {
        "id": "pickup_hospital",
        "name": "和康医院门诊部",
        "address": "和康医院门诊部正门",
        "lat": 29.56732,
        "lng": 106.55368,
        "tag": "医院",
    },
    {
        "id": "pickup_station",
        "name": "社区公交站",
        "address": "颐和苑公交站旁",
        "lat": 29.56546,
        "lng": 106.54792,
        "tag": "公交接驳",
    },
]

STATUS_TEXT_MAP = {
    "draft": "草稿中",
    "parsing_intent": "解析需求中",
    "awaiting_slot_clarification": "待补充出行信息",
    "awaiting_intent_confirmation": "待确认订票意图",
    "queued": "已进入查询队列",
    "searching": "正在查询车次",
    "search_results_ready": "车次候选已生成",
    "awaiting_candidate_confirmation": "待确认车次",
    "login_required": "需要登录 12306",
    "awaiting_human_verification": "等待人工验证",
    "selecting_train": "正在尝试锁定车次",
    "awaiting_final_submit_confirmation": "待提交前确认",
    "submitting_order": "正在提交订单",
    "awaiting_payment": "待支付",
    "completed": "已完成",
    "failed": "执行失败",
    "cancelled": "已取消",
    "expired": "已过期",
}

EVENT_TITLE_MAP = {
    "task_created": "任务已创建",
    "intent_parsed": "意图解析完成",
    "slot_clarification_requested": "需要补充信息",
    "intent_confirmed": "意图已确认",
    "search_started": "开始查询",
    "browser_progress": "执行进度",
    "search_results_ready": "候选车次已生成",
    "candidate_confirmed": "车次已确认",
    "login_required": "需要登录",
    "human_verification_required": "等待人工验证",
    "human_step_completed": "人工步骤完成",
    "final_submit_confirmed": "已确认提交",
    "order_submitted": "订单已提交",
    "payment_pending": "等待支付",
    "task_completed": "任务已完成",
    "task_failed": "任务失败",
    "task_cancelled": "任务已取消",
}


def _ok(data: Optional[Dict[str, Any]] = None, msg: str = "ok", status: int = 200) -> web.Response:
    return web.json_response({"code": 0, "msg": msg, "data": data or {}}, status=status)


def _error(msg: str, code: int = -1, status: int = 200, data: Optional[Dict[str, Any]] = None) -> web.Response:
    payload: Dict[str, Any] = {"code": code, "msg": msg}
    if data is not None:
        payload["data"] = data
    return web.json_response(payload, status=status)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_preferences(value: Any) -> Dict[str, Any]:
    preferences = dict(DEFAULT_PREFERENCES)
    if isinstance(value, dict):
        preferences.update(value)
    preferences["mode"] = "senior" if preferences.get("mode") == "senior" else "family"
    preferences["alertPush"] = bool(preferences.get("alertPush", True))
    preferences["healthDigest"] = bool(preferences.get("healthDigest", True))
    preferences["quietMode"] = bool(preferences.get("quietMode", False))
    return preferences


def _ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _default_state() -> Dict[str, Any]:
    return {
        "users": [],
        "sessions": {},
        "reset_codes": {},
    }


def _load_state() -> Dict[str, Any]:
    global _STATE_CACHE
    if _STATE_CACHE is not None:
        return _STATE_CACHE
    _ensure_state_dir()
    if not STATE_FILE.exists():
        _STATE_CACHE = _default_state()
        _save_state(_STATE_CACHE)
        return _STATE_CACHE
    try:
        _STATE_CACHE = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("Failed to read miniapp gateway state, resetting to defaults")
        _STATE_CACHE = _default_state()
        _save_state(_STATE_CACHE)
    return _STATE_CACHE


def _save_state(state: Dict[str, Any]) -> None:
    global _STATE_CACHE
    _ensure_state_dir()
    temp_path = STATE_FILE.with_suffix(".tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(STATE_FILE)
    _STATE_CACHE = state


def _purge_expired_state(state: Dict[str, Any]) -> bool:
    changed = False
    now = int(time.time())

    sessions = state.get("sessions", {})
    expired_sessions = [token for token, info in sessions.items() if int(info.get("expiresAt", 0)) <= now]
    for token in expired_sessions:
        sessions.pop(token, None)
        changed = True

    reset_codes = state.get("reset_codes", {})
    expired_codes = [key for key, info in reset_codes.items() if int(info.get("expiresAt", 0)) <= now]
    for key in expired_codes:
        reset_codes.pop(key, None)
        changed = True

    return changed


def _hash_password(password: str, salt_hex: Optional[str] = None) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(PBKDF2_ITERATIONS, salt.hex(), digest.hex())


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iteration_text, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iteration_text),
        )
        return secrets.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def _sanitize_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user.get("id", ""),
        "username": user.get("username", ""),
        "phone": user.get("phone", ""),
        "nickname": user.get("nickname") or user.get("username", ""),
        "preferences": _normalize_preferences(user.get("preferences")),
    }


def _find_user_by_id(state: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
    for user in state.get("users", []):
        if user.get("id") == user_id:
            return user
    return None


def _find_user_by_phone(state: Dict[str, Any], phone: str) -> Optional[Dict[str, Any]]:
    phone = _normalize_text(phone)
    for user in state.get("users", []):
        if _normalize_text(user.get("phone")) == phone:
            return user
    return None


def _find_user_by_username(state: Dict[str, Any], username: str) -> Optional[Dict[str, Any]]:
    username = _normalize_text(username)
    for user in state.get("users", []):
        if _normalize_text(user.get("username")) == username:
            return user
    return None


def _find_user_by_account(state: Dict[str, Any], account: str) -> Optional[Dict[str, Any]]:
    account = _normalize_text(account)
    if not account:
        return None
    return _find_user_by_phone(state, account) or _find_user_by_username(state, account)


def _issue_token(state: Dict[str, Any], user_id: str) -> str:
    token = "tok_{}".format(secrets.token_urlsafe(24))
    state.setdefault("sessions", {})[token] = {
        "userId": user_id,
        "issuedAt": int(time.time()),
        "expiresAt": int(time.time()) + AUTH_TOKEN_TTL,
    }
    return token


def _extract_bearer_token(request: web.Request) -> str:
    raw_value = request.headers.get("Authorization", "")
    if raw_value.startswith("Bearer "):
        return raw_value[7:].strip()
    return ""


async def _get_authenticated_user(request: web.Request) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    token = _extract_bearer_token(request)
    if not token:
        return None, None, ""

    async with STATE_LOCK:
        state = _load_state()
        changed = _purge_expired_state(state)
        session = state.get("sessions", {}).get(token)
        if not session:
            if changed:
                _save_state(state)
            return None, None, token
        user = _find_user_by_id(state, session.get("userId", ""))
        if not user:
            state.get("sessions", {}).pop(token, None)
            _save_state(state)
            return None, None, token
        if changed:
            _save_state(state)
        return state, user, token


def _ticket_user_id_from_request(request: web.Request) -> str:
    token = _extract_bearer_token(request)
    if token:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:20]
        return "miniapp_{}".format(digest)
    forwarded_for = _normalize_text(request.headers.get("X-Forwarded-For") or request.remote or "guest")
    return "miniapp_{}".format(hashlib.sha1(forwarded_for.encode("utf-8")).hexdigest()[:20])


def _require_account_fields(phone: str, username: str, password: str) -> Optional[web.Response]:
    if not phone or not username or not password:
        return _error("手机号、用户名和密码均为必填项")
    if len(password) < 6:
        return _error("密码长度至少 6 位")
    return None


def _format_distance_text(distance_meters: float) -> str:
    if distance_meters >= 1000:
        return "{:.1f} 公里".format(distance_meters / 1000.0)
    return "{} 米".format(int(distance_meters))


def _format_duration_text(duration_seconds: float) -> str:
    minutes = max(1, int(round(duration_seconds / 60.0)))
    if minutes >= 60:
        hours = minutes // 60
        remain = minutes % 60
        if remain:
            return "{} 小时 {} 分钟".format(hours, remain)
        return "{} 小时".format(hours)
    return "{} 分钟".format(minutes)


def _estimate_price_text(distance_meters: float, duration_seconds: float) -> str:
    kilometers = max(distance_meters / 1000.0, 1.5)
    minutes = max(duration_seconds / 60.0, 5.0)
    amount = 13.0 + kilometers * 2.6 + minutes * 0.35
    return "约 ¥{:.0f}".format(amount)


def _haversine_distance_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _normalize_ride_option(item: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    return {
        "id": item.get("id") or item.get("uid") or "ride-option-{}".format(index + 1),
        "name": item.get("name") or item.get("title") or item.get("poi_name") or item.get("address") or "候选地点{}".format(index + 1),
        "address": item.get("address") or item.get("addr") or item.get("poi_address") or item.get("district") or "",
        "lat": float(item.get("lat") or item.get("latitude") or item.get("location", {}).get("lat") or 0),
        "lng": float(item.get("lng") or item.get("longitude") or item.get("location", {}).get("lng") or 0),
        "tag": item.get("tag") or item.get("type") or "",
    }


def _demo_ride_points(keyword: str) -> List[Dict[str, Any]]:
    keyword = _normalize_text(keyword)
    if not keyword:
        return DEMO_RIDE_POINTS[:3]
    result: List[Dict[str, Any]] = []
    for item in DEMO_RIDE_POINTS:
        haystack = "{} {} {}".format(item["name"], item["address"], item.get("tag", ""))
        if keyword in haystack:
            result.append(item)
    return result or DEMO_RIDE_POINTS[:3]


async def _baidu_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError("Baidu API returned {}".format(response.status))
            try:
                return json.loads(text)
            except json.JSONDecodeError as error:
                raise RuntimeError("Baidu API response parse failed") from error


async def _search_places(keyword: str) -> Tuple[List[Dict[str, Any]], str]:
    keyword = _normalize_text(keyword)
    if not BAIDU_MAP_AK:
        return list(_demo_ride_points(keyword)), "mock"

    try:
        payload = await _baidu_get_json(
            "https://api.map.baidu.com/place/v2/suggestion",
            {
                "query": keyword or "医院",
                "region": "全国",
                "city_limit": "false",
                "output": "json",
                "ak": BAIDU_MAP_AK,
            },
        )
        if int(payload.get("status", 1)) != 0:
            raise RuntimeError(payload.get("message") or "Baidu place suggestion failed")
        result = [_normalize_ride_option(item, index) for index, item in enumerate(payload.get("result", []) or [])]
        return (result or list(_demo_ride_points(keyword))), "baidu"
    except Exception:
        LOGGER.exception("Ride place search failed, using demo fallback")
        return list(_demo_ride_points(keyword)), "fallback"


async def _estimate_route(pickup: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    pickup_lat = float(pickup.get("lat") or pickup.get("latitude") or 0)
    pickup_lng = float(pickup.get("lng") or pickup.get("longitude") or 0)
    destination_lat = float(destination.get("lat") or destination.get("latitude") or 0)
    destination_lng = float(destination.get("lng") or destination.get("longitude") or 0)

    if BAIDU_MAP_AK and pickup_lat and pickup_lng and destination_lat and destination_lng:
        try:
            payload = await _baidu_get_json(
                "https://api.map.baidu.com/directionlite/v1/driving",
                {
                    "origin": "{},{}".format(pickup_lat, pickup_lng),
                    "destination": "{},{}".format(destination_lat, destination_lng),
                    "ak": BAIDU_MAP_AK,
                    "output": "json",
                },
            )
            if int(payload.get("status", 1)) == 0:
                routes = ((payload.get("result") or {}).get("routes") or [])
                if routes:
                    route = routes[0]
                    distance_meters = float(route.get("distance") or 0)
                    duration_seconds = float(route.get("duration") or 0)
                    return {
                        "provider": "baidu",
                        "distanceText": _format_distance_text(distance_meters),
                        "durationText": _format_duration_text(duration_seconds),
                        "priceText": _estimate_price_text(distance_meters, duration_seconds),
                        "vehicleText": "舒适型优先",
                        "route": {
                            "distance_meters": distance_meters,
                            "duration_seconds": duration_seconds,
                            "distance_text": _format_distance_text(distance_meters),
                            "duration_text": _format_duration_text(duration_seconds),
                        },
                    }
        except Exception:
            LOGGER.exception("Ride route estimation via Baidu failed, using fallback")

    if pickup_lat and pickup_lng and destination_lat and destination_lng:
        distance_meters = _haversine_distance_meters(pickup_lat, pickup_lng, destination_lat, destination_lng) * 1.18
    else:
        distance_meters = 5200.0
    duration_seconds = max(600.0, distance_meters / 9.5)
    return {
        "provider": "fallback",
        "distanceText": _format_distance_text(distance_meters),
        "durationText": _format_duration_text(duration_seconds),
        "priceText": _estimate_price_text(distance_meters, duration_seconds),
        "vehicleText": "舒适型优先",
        "route": {
            "distance_meters": distance_meters,
            "duration_seconds": duration_seconds,
            "distance_text": _format_distance_text(distance_meters),
            "duration_text": _format_duration_text(duration_seconds),
        },
    }


def _status_text(status: str) -> str:
    return STATUS_TEXT_MAP.get(status, status or "处理中")


def _pick_ticket_seat(candidate: Dict[str, Any]) -> str:
    seat_inventory = candidate.get("seatInventory") or candidate.get("seat_inventory") or []
    preferred_statuses = {"有", "数字", "候补"}
    for item in seat_inventory:
        if str(item.get("normalizedStatus") or item.get("status") or "") in preferred_statuses:
            seat_type = _normalize_text(item.get("seatType") or item.get("seat_type"))
            if seat_type:
                return seat_type
    return _normalize_text(candidate.get("seatType")) or "二等座"


def _map_ticket_candidate(item: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    train_no = item.get("trainNo") or item.get("train_no") or item.get("stationTrainCode") or "车次{}".format(index + 1)
    origin = item.get("originStation") or item.get("fromStation") or item.get("from_station_name") or ""
    destination = item.get("destinationStation") or item.get("toStation") or item.get("to_station_name") or ""
    departure_time = item.get("departureTime") or item.get("depart_time") or item.get("start_time") or ""
    arrival_time = item.get("arrivalTime") or item.get("arrive_time") or item.get("arrive_time_text") or ""
    duration_text = item.get("durationText") or item.get("duration") or item.get("lishi") or ""
    seat_inventory = item.get("seatInventory") or []
    seat_summary = "、".join(
        "{}{}".format(entry.get("seatType", ""), entry.get("rawValue", ""))
        for entry in seat_inventory[:3]
        if entry.get("seatType")
    )
    description = "{} - {} {} {}".format(departure_time, arrival_time, duration_text, seat_summary).strip()
    return {
        "id": item.get("id") or train_no,
        "title": "{} {} -> {}".format(train_no, origin, destination).strip(),
        "desc": description,
        "raw": item,
    }


def _map_ticket_event(item: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    event_type = _normalize_text(item.get("type"))
    return {
        "id": item.get("id") or "evt-{}".format(index + 1),
        "title": EVENT_TITLE_MAP.get(event_type, _status_text(_normalize_text(item.get("status"))) or "任务事件"),
        "event": event_type,
        "desc": _normalize_text(item.get("message")) or _status_text(_normalize_text(item.get("status"))),
        "time": item.get("createdAt") or item.get("created_at") or "",
        "status": item.get("status") or "",
    }


def _extract_ticket_payload(response_data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(response_data, dict) and isinstance(response_data.get("data"), dict):
        return response_data["data"]
    return response_data if isinstance(response_data, dict) else {}


def _map_ticket_snapshot(response_data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _extract_ticket_payload(response_data)
    task = payload.get("task") or {}
    status = _normalize_text(task.get("status") or payload.get("status"))
    candidates = [_map_ticket_candidate(item, index) for index, item in enumerate(payload.get("candidates") or [])]
    recent_events = payload.get("recentEvents") or payload.get("events") or []
    pending_human_action = payload.get("pendingHumanAction")
    return {
        "taskId": task.get("id") or payload.get("taskId") or "",
        "status": status,
        "statusText": _status_text(status),
        "candidates": candidates,
        "events": [_map_ticket_event(item, index) for index, item in enumerate(recent_events)],
        "pendingHumanAction": pending_human_action,
        "selectedCandidateId": task.get("selectedCandidateId") or "",
    }


def _ticket_status(response_data: Dict[str, Any]) -> str:
    payload = _extract_ticket_payload(response_data)
    task = payload.get("task") or {}
    return _normalize_text(task.get("status") or payload.get("status"))


async def _ticket_request(method: str, path: str, user_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=25)
    headers = {"x-user-id": user_id}
    kwargs: Dict[str, Any] = {"headers": headers, "timeout": timeout}
    if payload is not None:
        kwargs["json"] = payload

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method.upper(), "{}{}".format(TICKET_BACKEND_URL, path), **kwargs) as response:
                text = await response.text()
                try:
                    body = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    body = {"error": text}
                if response.status >= 400:
                    message = (((body.get("error") or {}).get("message")) if isinstance(body.get("error"), dict) else None) or body.get("message") or body.get("error") or "订票服务请求失败"
                    raise RuntimeError(str(message))
                return body
    except aiohttp.ClientError as error:
        raise RuntimeError("订票服务暂未启动") from error


async def _fetch_ticket_detail(user_id: str, task_id: str) -> Dict[str, Any]:
    return await _ticket_request("GET", "/api/tasks/{}".format(task_id), user_id)


async def _wait_for_ticket_progress(user_id: str, task_id: str, timeout_seconds: float = 18.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    stable_statuses = {
        "awaiting_slot_clarification",
        "awaiting_candidate_confirmation",
        "search_results_ready",
        "awaiting_human_verification",
        "awaiting_final_submit_confirmation",
        "awaiting_payment",
        "completed",
        "failed",
        "cancelled",
        "expired",
    }
    last_response: Dict[str, Any] = {}

    while True:
        last_response = await _fetch_ticket_detail(user_id, task_id)
        snapshot = _map_ticket_snapshot(last_response)
        status = snapshot.get("status", "")
        if snapshot.get("candidates") or status in stable_statuses:
            return last_response
        if time.time() >= deadline:
            return last_response
        await asyncio.sleep(1.0)


async def register_account(request: web.Request) -> web.Response:
    payload = await request.json()
    phone = _normalize_text(payload.get("phone"))
    username = _normalize_text(payload.get("username"))
    password = _normalize_text(payload.get("password"))
    error = _require_account_fields(phone, username, password)
    if error:
        return error

    async with STATE_LOCK:
        state = _load_state()
        _purge_expired_state(state)
        if _find_user_by_phone(state, phone):
            return _error("手机号已注册")
        if _find_user_by_username(state, username):
            return _error("用户名已存在")

        now = int(time.time())
        user = {
            "id": "usr_{}".format(uuid4().hex[:16]),
            "phone": phone,
            "username": username,
            "nickname": username,
            "passwordHash": _hash_password(password),
            "preferences": _normalize_preferences(payload.get("preferences")),
            "createdAt": now,
            "updatedAt": now,
        }
        state.setdefault("users", []).append(user)
        token = _issue_token(state, user["id"])
        _save_state(state)

    return _ok({"token": token, "user": _sanitize_user(user)})


async def login_account(request: web.Request) -> web.Response:
    payload = await request.json()
    account = _normalize_text(payload.get("account") or payload.get("phone") or payload.get("username"))
    password = _normalize_text(payload.get("password"))
    if not account or not password:
        return _error("请输入账号和密码")

    async with STATE_LOCK:
        state = _load_state()
        changed = _purge_expired_state(state)
        user = _find_user_by_account(state, account)
        if not user or not _verify_password(password, _normalize_text(user.get("passwordHash"))):
            if changed:
                _save_state(state)
            return _error("账号或密码错误")
        token = _issue_token(state, user["id"])
        _save_state(state)

    return _ok({"token": token, "user": _sanitize_user(user)})


async def logout_account(request: web.Request) -> web.Response:
    token = _extract_bearer_token(request)
    if not token:
        return _ok({"loggedOut": True})

    async with STATE_LOCK:
        state = _load_state()
        _purge_expired_state(state)
        state.get("sessions", {}).pop(token, None)
        _save_state(state)

    return _ok({"loggedOut": True})


async def get_current_account(request: web.Request) -> web.Response:
    _state, user, token = await _get_authenticated_user(request)
    if not user:
        return _error("未登录或登录已失效", code=40101, status=401)
    return _ok({"token": token, "user": _sanitize_user(user)})


async def request_password_reset(request: web.Request) -> web.Response:
    payload = await request.json()
    account = _normalize_text(payload.get("account"))
    if not account:
        return _error("请填写手机号或用户名")

    async with STATE_LOCK:
        state = _load_state()
        _purge_expired_state(state)
        user = _find_user_by_account(state, account)
        if not user:
            return _error("未找到对应账号")

        reset_code = "{:06d}".format(secrets.randbelow(1000000))
        now = int(time.time())
        state.setdefault("reset_codes", {})[account] = {
            "userId": user["id"],
            "code": reset_code,
            "expiresAt": now + RESET_CODE_TTL,
            "createdAt": now,
        }
        _save_state(state)
        LOGGER.warning("[MiniAppResetCode] account=%s code=%s", account, reset_code)

    return _ok({"resetCode": reset_code, "expiresIn": RESET_CODE_TTL})


async def confirm_password_reset(request: web.Request) -> web.Response:
    payload = await request.json()
    account = _normalize_text(payload.get("account"))
    reset_code = _normalize_text(payload.get("resetCode"))
    password = _normalize_text(payload.get("password"))
    if not account or not reset_code or not password:
        return _error("请填写完整的重置参数")
    if len(password) < 6:
        return _error("密码长度至少 6 位")

    async with STATE_LOCK:
        state = _load_state()
        _purge_expired_state(state)
        reset_info = state.get("reset_codes", {}).get(account)
        if not reset_info or _normalize_text(reset_info.get("code")) != reset_code:
            return _error("重置码无效或已过期")
        user = _find_user_by_id(state, _normalize_text(reset_info.get("userId")))
        if not user:
            return _error("未找到对应账号")

        user["passwordHash"] = _hash_password(password)
        user["updatedAt"] = int(time.time())
        state.get("reset_codes", {}).pop(account, None)
        _save_state(state)

    return _ok({"reset": True})


async def update_user_preferences(request: web.Request) -> web.Response:
    payload = await request.json()
    async with STATE_LOCK:
        state = _load_state()
        _purge_expired_state(state)
        token = _extract_bearer_token(request)
        session = state.get("sessions", {}).get(token)
        if not session:
            return _error("未登录或登录已失效", code=40101, status=401)
        user = _find_user_by_id(state, _normalize_text(session.get("userId")))
        if not user:
            state.get("sessions", {}).pop(token, None)
            _save_state(state)
            return _error("未登录或登录已失效", code=40101, status=401)

        user["preferences"] = _normalize_preferences({**_normalize_preferences(user.get("preferences")), **(payload or {})})
        user["updatedAt"] = int(time.time())
        _save_state(state)

    return _ok({"preferences": user["preferences"], "user": _sanitize_user(user)})


async def suggest_ride_pickup(request: web.Request) -> web.Response:
    keyword = _normalize_text(request.query.get("keyword"))
    items, provider = await _search_places(keyword)
    return _ok({"items": items, "provider": provider})


async def search_ride_destination(request: web.Request) -> web.Response:
    keyword = _normalize_text(request.query.get("keyword"))
    items, provider = await _search_places(keyword)
    return _ok({"items": items, "provider": provider})


async def estimate_ride(request: web.Request) -> web.Response:
    payload = await request.json()
    pickup = payload.get("pickup") or {}
    destination = payload.get("destination") or {}
    if not pickup or not destination:
        return _error("请先确认上车点和目的地")

    estimate = await _estimate_route(pickup, destination)
    return _ok(estimate)


async def create_ride_preview(request: web.Request) -> web.Response:
    payload = await request.json()
    pickup = payload.get("pickup") or {}
    destination = payload.get("destination") or {}
    estimate = payload.get("estimate") or {}
    summary = "已为您生成叫车预览，从“{}”前往“{}”，{}，{}。".format(
        pickup.get("name") or "当前上车点",
        destination.get("name") or "目的地",
        estimate.get("distanceText") or estimate.get("distance") or "路程已计算",
        estimate.get("priceText") or estimate.get("price") or "费用已估算",
    )
    return _ok(
        {
            "summary": summary,
            "driverText": "司机待分配，优先舒适型",
            "arrivalText": "预计 3 分钟到达上车点",
        }
    )


async def create_ticket_task(request: web.Request) -> web.Response:
    payload = await request.json()
    demand = _normalize_text(payload.get("demand") or payload.get("rawText"))
    if not demand:
        return _error("请先描述订票需求")

    user_id = _ticket_user_id_from_request(request)
    create_response = await _ticket_request(
        "POST",
        "/api/assistant/tasks",
        user_id,
        {
            "taskType": "book_train_ticket",
            "rawText": demand,
            "idempotencyKey": "create_{}".format(uuid4().hex),
        },
    )
    create_data = _extract_ticket_payload(create_response)
    task_id = _normalize_text(create_data.get("taskId"))
    status = _normalize_text(create_data.get("status"))

    if not task_id:
        return _error("订票服务未返回任务编号")
    if status == "awaiting_slot_clarification":
        missing_slots = create_data.get("missingSlots") or []
        detail = "、".join(missing_slots) if missing_slots else "出发地、目的地、出行日期"
        return _error("请在需求中补充：{}".format(detail))
    if status == "awaiting_intent_confirmation":
        await _ticket_request(
            "POST",
            "/api/tasks/{}/confirm-intent".format(task_id),
            user_id,
            {
                "idempotencyKey": "intent_{}".format(uuid4().hex),
                "confirmed": True,
            },
        )

    detail_response = await _wait_for_ticket_progress(user_id, task_id)
    return _ok(_map_ticket_snapshot(detail_response), status=201)


async def get_ticket_task(request: web.Request) -> web.Response:
    task_id = _normalize_text(request.match_info.get("taskId"))
    user_id = _ticket_user_id_from_request(request)
    detail_response = await _fetch_ticket_detail(user_id, task_id)
    return _ok(_map_ticket_snapshot(detail_response))


async def get_ticket_task_events(request: web.Request) -> web.Response:
    task_id = _normalize_text(request.match_info.get("taskId"))
    user_id = _ticket_user_id_from_request(request)
    event_response = await _ticket_request("GET", "/api/tasks/{}/events".format(task_id), user_id)
    payload = _extract_ticket_payload(event_response)
    events = [_map_ticket_event(item, index) for index, item in enumerate(payload.get("events") or [])]
    return _ok({"taskId": task_id, "events": events, "nextSequenceId": payload.get("nextSequenceId") or 0})


async def ticket_task_action(request: web.Request) -> web.Response:
    task_id = _normalize_text(request.match_info.get("taskId"))
    action = _normalize_text(request.match_info.get("action"))
    payload = await request.json()
    user_id = _ticket_user_id_from_request(request)

    if action == "confirm-candidate":
        candidate = payload.get("candidate") or {}
        candidate_train_no = _normalize_text(candidate.get("trainNo") or candidate.get("train_no") or candidate.get("stationTrainCode") or candidate.get("code"))
        if not candidate_train_no:
            return _error("未识别到候选车次")
        seat_type = _normalize_text(payload.get("seatType")) or _pick_ticket_seat(candidate)
        allow_waitlist = bool(payload.get("allowWaitlist", True))
        await _ticket_request(
            "POST",
            "/api/tasks/{}/confirm-candidate".format(task_id),
            user_id,
            {
                "idempotencyKey": "candidate_{}".format(uuid4().hex),
                "candidateTrainNo": candidate_train_no,
                "seatType": seat_type,
                "allowWaitlist": allow_waitlist,
            },
        )
        detail_response = await _wait_for_ticket_progress(user_id, task_id, timeout_seconds=12.0)
        return _ok(_map_ticket_snapshot(detail_response))

    if action == "submit-preview":
        detail_response = await _wait_for_ticket_progress(user_id, task_id, timeout_seconds=12.0)
        status = _ticket_status(detail_response)
        snapshot = _map_ticket_snapshot(detail_response)

        if status == "awaiting_human_verification":
            return _error("当前任务需要在 12306 页面完成扫码登录或验证码验证后再继续", status=409)

        if status == "awaiting_final_submit_confirmation":
            pending_human_action = snapshot.get("pendingHumanAction") or {}
            checkpoint_id = _normalize_text(pending_human_action.get("checkpointId")) or "cp_{}_final_submit".format(task_id)
            await _ticket_request(
                "POST",
                "/api/tasks/{}/confirm-submit".format(task_id),
                user_id,
                {
                    "idempotencyKey": "submit_{}".format(uuid4().hex),
                    "checkpointId": checkpoint_id,
                    "confirmed": True,
                },
            )
            detail_response = await _wait_for_ticket_progress(user_id, task_id, timeout_seconds=12.0)
            status = _ticket_status(detail_response)
            snapshot = _map_ticket_snapshot(detail_response)

        if status == "awaiting_human_verification":
            return _error("当前任务需要在 12306 页面完成扫码登录或验证码验证后再继续", status=409)
        if status in {"failed", "cancelled", "expired"}:
            return _error("当前任务无法继续，状态：{}".format(snapshot.get("statusText") or status), status=409)
        return _ok(snapshot)

    if action == "cancel":
        await _ticket_request(
            "POST",
            "/api/tasks/{}/cancel".format(task_id),
            user_id,
            {
                "idempotencyKey": "cancel_{}".format(uuid4().hex),
                "reason": "cancelled_by_miniapp_user",
            },
        )
        detail_response = await _fetch_ticket_detail(user_id, task_id)
        return _ok(_map_ticket_snapshot(detail_response))

    return _error("暂不支持的订票动作: {}".format(action), status=404)


def register_miniapp_gateway_routes(app: web.Application) -> None:
    app.router.add_post("/api/auth/register", register_account)
    app.router.add_post("/api/auth/login", login_account)
    app.router.add_post("/api/auth/logout", logout_account)
    app.router.add_get("/api/auth/me", get_current_account)
    app.router.add_post("/api/auth/password/reset/request", request_password_reset)
    app.router.add_post("/api/auth/password/reset/confirm", confirm_password_reset)
    app.router.add_put("/api/user/preferences", update_user_preferences)

    app.router.add_get("/api/agent/ride/suggest-pickup", suggest_ride_pickup)
    app.router.add_get("/api/agent/ride/search-destination", search_ride_destination)
    app.router.add_post("/api/agent/ride/estimate", estimate_ride)
    app.router.add_post("/api/agent/ride/create-preview", create_ride_preview)

    app.router.add_post("/api/agent/ticket/tasks", create_ticket_task)
    app.router.add_get("/api/agent/ticket/tasks/{taskId}", get_ticket_task)
    app.router.add_get("/api/agent/ticket/tasks/{taskId}/events", get_ticket_task_events)
    app.router.add_post("/api/agent/ticket/tasks/{taskId}/actions/{action}", ticket_task_action)

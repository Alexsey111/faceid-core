# app/api/routes/liveness_challenge.py — active challenge liveness (online access control).
#
# Эндпоинты:
#   POST /liveness/challenge/init    — JWT-auth, выдаёт challenge_id + ws_token + actions.
#   WS   /liveness/challenge/stream  — ws_token auth (query), бинарные JPEG-кадры →
#                                      result + liveness_token.
#
# Поток: init → клиент стримит кадры (≤ LIVENESS_WS_MAX_FRAMES, окно LIVENESS_CHALLENGE_TTL_S)
# → движок verify_challenge_stream → result + liveness_token (single-use) → /verify.
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.api.deps import require_auth
from app.api._helpers import decode_image_bytes
from app.core.config import settings
from app.infrastructure.redis_client import redis_client
from app.ml.liveness.challenge import (
    FrameObservation,
    observe_frame,
    sample_actions,
    verify_challenge_stream,
)
from app.services.liveness_token import issue_liveness_token

logger = logging.getLogger("liveness.challenge.route")

router = APIRouter(prefix="/liveness/challenge", tags=["liveness-challenge"])

_CHKEY = "lvchk:"  # Redis key состояния challenge
# Бёрст-лимит параллельных WS-стримов (анти CPU-перегруз на dev; production —
# согласно dev-vs-prod-hardware не сужаем архитектуру, но guard оставляем).
_WS_SEM = asyncio.Semaphore(max(1, settings.LIVENESS_WS_MAX_CONCURRENT))

# (detector, landmarker, passive_checker) — lazy singleton (модели @lru_cache в runtime)
_ML: tuple | None = None


def _get_ml() -> tuple:
    global _ML
    if _ML is None:
        from app.ml.detection.retinaface_detector import RetinaFaceDetector
        from app.ml.runtime import get_landmarker_106, get_liveness_checker

        _ML = (
            RetinaFaceDetector(det_size=settings.RETINA_DET_SIZE_SMALL),
            get_landmarker_106(),
            get_liveness_checker(),
        )
    return _ML


def _iso(ttl_s: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_s)).isoformat()


@router.post("/init")
async def init_challenge(_auth: dict = Depends(require_auth)):
    """Выдать challenge: случайные действия + ws_token (привязан к challenge_id)."""
    if not settings.LIVENESS_ENABLED or not settings.LIVENESS_ACTIVE_ENABLED:
        raise _exc(503, "Active liveness is disabled")

    cid = str(uuid.uuid4())
    ws_token = uuid.uuid4().hex
    actions = sample_actions()
    state = {"actions": actions, "ws_token": ws_token, "used": False, "streaming": False}
    redis_client.setex(_CHKEY + cid, json.dumps(state), ttl=settings.LIVENESS_CHALLENGE_TTL_S)
    logger.info("challenge_init challenge_id=%s actions=%s", cid, actions)
    return {
        "challenge_id": cid,
        "actions": actions,
        "ws_token": ws_token,
        "ws_url": f"/api/v1/liveness/challenge/stream?challenge_id={cid}&ws_token={ws_token}",
        "expires_at": _iso(settings.LIVENESS_CHALLENGE_TTL_S),
    }


def _exc(code: int, detail: str):
    from fastapi import HTTPException
    return HTTPException(status_code=code, detail=detail)


@router.websocket("/stream")
async def stream(
    ws: WebSocket,
    challenge_id: str = Query(...),
    ws_token: str = Query(...),
):
    """Реалтайм-стрим кадров → verify → liveness_token. Single-use challenge."""
    # 1) feature-flag
    if not settings.LIVENESS_ENABLED or not settings.LIVENESS_ACTIVE_ENABLED:
        await ws.close(code=4503, reason="active liveness disabled")
        return

    # 2) валидация challenge (Redis)
    raw = redis_client.get(_CHKEY + challenge_id)
    if raw is None:
        await ws.close(code=4410, reason="challenge expired or unknown")
        return
    try:
        st = json.loads(raw)
    except (ValueError, TypeError):
        await ws.close(code=4400, reason="bad challenge state")
        return
    if st.get("ws_token") != ws_token:
        await ws.close(code=4401, reason="bad ws_token")
        return
    if st.get("used"):
        await ws.close(code=4410, reason="challenge already used")
        return
    if st.get("streaming"):
        await ws.close(code=4409, reason="challenge already streaming")
        return

    # 3) бёрст-лимит (reject-if-busy, не блокируем gate)
    try:
        await asyncio.wait_for(_WS_SEM.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        await ws.close(code=4503, reason="server busy")
        return

    # 4) пометить streaming (anti-concurrent) — внутри sem, до accept
    redis_client.setex(
        _CHKEY + challenge_id,
        json.dumps({**st, "streaming": True}),
        ttl=settings.LIVENESS_CHALLENGE_TTL_S,
    )

    try:
        await ws.accept()
        await ws.send_json({
            "type": "challenge",
            "actions": st["actions"],
            "deadline_ms": int(settings.LIVENESS_CHALLENGE_TTL_S * 1000),
        })

        det, landmarker, passive = _get_ml()
        obs: list[FrameObservation] = []
        t0 = time.monotonic()
        deadline = t0 + settings.LIVENESS_CHALLENGE_TTL_S
        max_frames = settings.LIVENESS_WS_MAX_FRAMES

        try:
            while len(obs) < max_frames and time.monotonic() < deadline:
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                if msg.get("bytes"):
                    try:
                        img = decode_image_bytes(msg["bytes"])
                    except ValueError:
                        # невалидный кадр → пропускаем (стрим продолжается)
                        continue
                    ob = await asyncio.to_thread(
                        observe_frame, img, det, landmarker, passive, len(obs), t0
                    )
                    if ob is None:
                        # лица нет на кадре — пропускаем (стрим продолжается)
                        continue
                    obs.append(ob)
                elif msg.get("text"):
                    try:
                        cmd = json.loads(msg["text"]).get("cmd")
                    except (ValueError, TypeError):
                        cmd = None
                    if cmd == "done":
                        break  # клиент завершил действия → финальный вердикт
                    if cmd == "cancel":
                        await ws.send_json({"type": "cancelled"})
                        break
        except (asyncio.TimeoutError, WebSocketDisconnect):
            pass

        result = verify_challenge_stream(obs, st["actions"])
        token = issue_liveness_token(challenge_id) if result.is_live else None

        # 5) mark used (single-use anti-replay) — даже при провале (повторно не дать)
        redis_client.setex(
            _CHKEY + challenge_id,
            json.dumps({**st, "used": True, "streaming": False}),
            ttl=settings.LIVENESS_CHALLENGE_TTL_S,
        )

        await ws.send_json({
            "type": "result",
            **result.to_dict(),
            "liveness_token": token,
            "spoofing_indicators": {
                "consistency": "ok" if result.consistency_ok else "fail",
                "reason": result.reason,
            },
        })
        logger.info(
            "challenge_result challenge_id=%s is_live=%s conf=%.3f frames=%d reason=%s",
            challenge_id, result.is_live, result.confidence, result.n_frames, result.reason,
        )
    finally:
        _WS_SEM.release()
        try:
            await ws.close()
        except Exception:
            pass
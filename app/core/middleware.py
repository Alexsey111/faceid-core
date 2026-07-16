import uuid
from fastapi import Request

from app.api._helpers import extract_client_ip
from app.core.logger import reset_trace_id, set_trace_id
from app.core.request_context import reset_client_ip, set_client_ip


def resolve_endpoint(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or request.url.path


async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Сквозной trace_id для корреляции всех логов в рамках HTTP-запроса.
    token = set_trace_id(request_id)
    # Сквозной клиентский IP для audit-логов verification_logs (audit E2,
    # ТЗ-схема требует request_ip). Доехает в background-логирование через
    # копирование context в asyncio.create_task.
    client_ip = extract_client_ip(request)
    ip_token = set_client_ip(client_ip)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        reset_trace_id(token)
        reset_client_ip(ip_token)

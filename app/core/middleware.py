import uuid
from fastapi import Request

from app.core.logger import reset_trace_id, set_trace_id


def resolve_endpoint(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or request.url.path


async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Сквозной trace_id для корреляции всех логов в рамках HTTP-запроса.
    token = set_trace_id(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        reset_trace_id(token)

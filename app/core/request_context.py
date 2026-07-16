# app/core/request_context.py — сквозной клиентский IP для audit-логов.
#
# ContextVar несёт IP клиента через весь стек вызовов (route → service → repo),
# не загромождая сигнатуры. asyncio.create_task копирует context → IP доезжает
# в background-логирование (_persist_verification_log_background).
#
# Для worker-пути (отдельный процесс) IP кладётся в job-payload и
# восстанавливается в contextvar через set_client_ip в начале process_batch.
#
# IP извлекается из X-Real-IP / X-Forwarded-For (nginx api_lb ставит оба,
# infrastructure/nginx/api_lb.conf:66-67), fallback request.client.host.

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

_client_ip_var: ContextVar[Optional[str]] = ContextVar("client_ip", default=None)


def set_client_ip(ip: Optional[str]) -> Token[Optional[str]]:
    """Установить клиентский IP в текущий context. Возвращает token для reset."""
    return _client_ip_var.set(ip)


def get_client_ip() -> Optional[str]:
    """Текущий клиентский IP из context (None если вне HTTP-запроса/worker)."""
    return _client_ip_var.get()


def reset_client_ip(token: Token[Optional[str]]) -> None:
    """Сбросить IP (всегда в finally middleware/worker, чтобы не утёк в следующий запрос)."""
    _client_ip_var.reset(token)
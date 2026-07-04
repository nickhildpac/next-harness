import logging
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response
from pythonjsonlogger import json
from starlette.middleware.base import BaseHTTPMiddleware

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    formatter = json.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logging.getLogger("app.request").info(
                "request_complete",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": locals().get("response").status_code
                    if "response" in locals()
                    else 500,
                    "latency_ms": latency_ms,
                },
            )
            request_id_ctx.reset(token)

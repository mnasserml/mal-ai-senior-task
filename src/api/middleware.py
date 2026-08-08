import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from src.infrastructure.observability.tracing import init_trace, emit_trace_log

class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        session_id = request.headers.get("x-session-id", "default_session")

        # Initialize ContextVar trace for this request execution stack
        init_trace(request_id=request_id, session_id=session_id)

        start_time = time.perf_counter()
        response = await call_next(request)
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Attach request_id header
        response.headers["X-Request-ID"] = request_id

        # Emit structured JSON trace log
        emit_trace_log(total_latency_ms=total_latency_ms)

        return response

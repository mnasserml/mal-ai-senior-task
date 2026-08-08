import os
import sys
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from typing import Optional
from src.domain.models import TraceMetrics

# Logger setup
logger = logging.getLogger("mal_trace_logger")
logger.setLevel(logging.INFO)

# 1. Console Stream Handler (stdout - active for FastAPI app, quiet during pytest)
if "PYTEST_CURRENT_TEST" not in os.environ and not any(type(h) is logging.StreamHandler for h in logger.handlers):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

# 2. Rotating File Handler (logs/app_trace.log)
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler(
        "logs/app_trace.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB limit per log file
        backupCount=5,              # Keep 5 historical log backups
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

# ContextVar for trace isolation across async requests
_current_trace: ContextVar[Optional[TraceMetrics]] = ContextVar("current_trace", default=None)
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")

def init_trace(request_id: str, session_id: str) -> TraceMetrics:
    metrics = TraceMetrics()
    _current_trace.set(metrics)
    _request_id_var.set(request_id)
    _session_id_var.set(session_id)
    return metrics

def get_current_trace() -> Optional[TraceMetrics]:
    return _current_trace.get()

def emit_trace_log(total_latency_ms: float):
    metrics = get_current_trace()
    if not metrics:
        return

    metrics.total_latency_ms = round(total_latency_ms, 2)
    step_latencies = {
        "pii_redaction": round(metrics.pii_redaction_latency_ms, 2),
        "vector_retrieval": round(metrics.retrieval_latency_ms, 2),
        "scope_guard": round(metrics.scope_guard_latency_ms, 2),
        "llm_generation": round(metrics.llm_latency_ms, 2),
    }
    if metrics.eval_latency_ms > 0:
        step_latencies["llm_judge_eval"] = round(metrics.eval_latency_ms, 2)

    trace_payload = {
        "event": "request_trace",
        "request_id": _request_id_var.get(),
        "session_id": _session_id_var.get(),
        "total_latency_ms": metrics.total_latency_ms,
        "step_latencies_ms": step_latencies,
        "retrieval": {
            "retrieved_chunk_ids": metrics.retrieved_chunk_ids,
            "context_relevance_score": round(metrics.context_relevance_score, 4),
        },
        "llm": {
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "total_tokens": metrics.total_tokens,
        },
        "security": {
            "pii_redacted_count": metrics.pii_redacted_count,
            "refused": metrics.refused,
        }
    }

    if metrics.is_grounded is not None:
        trace_payload["quality"] = {
            "is_grounded": metrics.is_grounded,
            "is_relevant": metrics.is_relevant,
            "groundedness_score": round(metrics.groundedness_score, 2) if metrics.groundedness_score is not None else None,
            "relevance_score": round(metrics.relevance_score, 2) if metrics.relevance_score is not None else None,
            "overall_score": round(metrics.overall_score, 2) if metrics.overall_score is not None else None,
        }

    if metrics.steps:
        trace_payload["steps"] = metrics.steps.model_dump(exclude_none=True)

    if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
        # Under pytest, save trace logs to file (logs/app_trace.log) without polluting stdout
        for h in logger.handlers:
            if isinstance(h, RotatingFileHandler):
                h.emit(logging.LogRecord("mal_trace_logger", logging.INFO, "", 0, json.dumps(trace_payload, ensure_ascii=False), (), None))
        return

    logger.info(json.dumps(trace_payload, ensure_ascii=False))

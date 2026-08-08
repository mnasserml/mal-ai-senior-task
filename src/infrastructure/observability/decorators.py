import time
from typing import List, Optional
from src.domain.ports import (
    IPIIRedactor,
    IScopeGuard,
    IVectorStore,
    ILLMProvider,
    ISessionStore
)
from src.domain.models import (
    ChatMessage,
    DocumentChunk,
    RetrievalResult,
    PIIRedactionResult,
    LLMResponse,
    PIIStepTrace,
    VectorRetrievalStepTrace,
    VectorRetrievalChunk,
    ScopeGuardStepTrace
)
from src.infrastructure.observability.tracing import get_current_trace

class ObservablePIIAdapter(IPIIRedactor):
    def __init__(self, inner: IPIIRedactor, engine_name: Optional[str] = None):
        self.inner = inner
        self.engine_name = engine_name or getattr(inner, "engine_name", None) or ("aajil" if "Aajil" in inner.__class__.__name__ else "presidio")

    def redact(self, text: str, language: str = "en") -> PIIRedactionResult:
        start = time.perf_counter()
        result = self.inner.redact(text, language)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        trace = get_current_trace()
        if trace:
            trace.pii_redaction_latency_ms += elapsed_ms
            trace.pii_redacted_count += result.redacted_count
            trace.steps.step_1_pii_redaction = PIIStepTrace(
                latency_ms=round(elapsed_ms, 2),
                engine=self.engine_name,
                redacted_text=result.redacted_text,
                redacted_count=result.redacted_count,
                detected_entities_count=len(result.detected_entities)
            )

        return result

class ObservableScopeAdapter(IScopeGuard):
    def __init__(self, inner: IScopeGuard):
        self.inner = inner

    def is_in_scope(self, text: str, retrieval_result: Optional[RetrievalResult] = None) -> tuple[bool, Optional[str]]:
        start = time.perf_counter()
        in_scope, refusal = self.inner.is_in_scope(text, retrieval_result)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        trace = get_current_trace()
        if trace:
            trace.scope_guard_latency_ms += elapsed_ms
            if not in_scope:
                trace.refused = True
            
            existing = trace.steps.step_3_scope_guard or ScopeGuardStepTrace()
            existing.latency_ms = round(elapsed_ms, 2)
            existing.is_in_scope = in_scope
            existing.refusal_message = refusal
            trace.steps.step_3_scope_guard = existing

        return in_scope, refusal

class ObservableVectorAdapter(IVectorStore):
    def __init__(self, inner: IVectorStore):
        self.inner = inner

    def add_documents(self, documents: List[DocumentChunk]) -> None:
        self.inner.add_documents(documents)

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        start = time.perf_counter()
        result = self.inner.search(query, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        trace = get_current_trace()
        if trace:
            trace.retrieval_latency_ms += elapsed_ms
            trace.retrieved_chunk_ids = [chunk.chunk_id for chunk in result.chunks]
            trace.context_relevance_score = result.average_relevance_score
            
            chunks_snippet = [
                VectorRetrievalChunk(
                    chunk_id=c.chunk_id,
                    doc_name=c.doc_name,
                    content_snippet=c.content[:150] + ("..." if len(c.content) > 150 else "")
                )
                for c in result.chunks
            ]
            trace.steps.step_2_vector_retrieval = VectorRetrievalStepTrace(
                latency_ms=round(elapsed_ms, 2),
                average_relevance_score=round(result.average_relevance_score, 4),
                retrieved_chunks=chunks_snippet
            )

        return result

class ObservableLLMAdapter(ILLMProvider):
    def __init__(self, inner: ILLMProvider):
        self.inner = inner

    def generate_response(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk],
        system_prompt: str
    ) -> LLMResponse:
        start = time.perf_counter()
        response = self.inner.generate_response(messages, context_chunks, system_prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        trace = get_current_trace()
        if trace:
            trace.llm_latency_ms += elapsed_ms

        return response

class ObservableSessionAdapter(ISessionStore):
    def __init__(self, inner: ISessionStore):
        self.inner = inner

    def get_history(self, session_id: str, limit: int = 5) -> List[ChatMessage]:
        return self.inner.get_history(session_id, limit)

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        self.inner.add_message(session_id, message)

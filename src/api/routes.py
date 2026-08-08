from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from dependency_injector.wiring import Provide, inject

from src.domain.models import ChatRequest, ChatResponse
from src.application.chat_use_case import ChatUseCase
from src.infrastructure.container import Container
from src.domain.ports import IPIIRedactor, IScopeGuard, IVectorStore, ILLMProvider
from src.config import Settings
from src.infrastructure.observability.tracing import _session_id_var

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
@inject
def chat_endpoint(
    request: ChatRequest,
    chat_use_case: ChatUseCase = Depends(Provide[Container.chat_use_case])
) -> ChatResponse:
    try:
        # Bind session ID to trace context
        _session_id_var.set(request.session_id)
        response = chat_use_case.execute(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )

@router.get("/health", status_code=status.HTTP_200_OK)
@inject
def health_endpoint(
    settings: Settings = Depends(Provide[Container.settings]),
    pii_redactor: IPIIRedactor = Depends(Provide[Container.observable_pii]),
    scope_guard: IScopeGuard = Depends(Provide[Container.observable_scope]),
    vector_store: IVectorStore = Depends(Provide[Container.observable_vector]),
    llm_provider: ILLMProvider = Depends(Provide[Container.observable_llm])
) -> dict:
    components = {}
    overall_status = "healthy"

    # 1. PII Redactor Check
    try:
        test_pii = pii_redactor.redact("Test 784-1990-1234567-1", language="en")
        pii_status = "ok" if "784-1990-1234567-1" not in test_pii.redacted_text else "degraded"
    except Exception as e:
        pii_status = f"error: {str(e)}"
        overall_status = "degraded"

    components["pii_redactor"] = {
        "status": pii_status,
        "engine": getattr(settings.pii, "engine", "aajil"),
        "supported_languages": getattr(settings.pii, "supported_languages", ["en", "ar"])
    }

    # 2. Vector Store & Retrieval Check
    try:
        raw_vec = vector_store.inner if hasattr(vector_store, 'inner') else vector_store
        chunks_indexed = len(raw_vec.documents) if hasattr(raw_vec, 'documents') else 0
        retrieval_res = vector_store.search("Murabaha", top_k=1)
        vec_status = "ok" if chunks_indexed > 0 and retrieval_res is not None else "degraded"
    except Exception as e:
        vec_status = f"error: {str(e)}"
        chunks_indexed = 0
        overall_status = "degraded"

    components["vector_store"] = {
        "status": vec_status,
        "provider": settings.vector_store.provider,
        "embedding_model": settings.vector_store.embedding_model,
        "chunks_indexed": chunks_indexed
    }

    # 3. Scope Guardrails Check
    try:
        is_in, _ = scope_guard.is_in_scope("What is Murabaha?")
        guard_status = "ok" if is_in else "degraded"
    except Exception as e:
        guard_status = f"error: {str(e)}"
        overall_status = "degraded"

    components["guardrails"] = {
        "status": guard_status,
        "enabled": settings.guardrails.enabled,
        "classifier_type": settings.guardrails.classifier_type
    }

    # 4. LLM Provider Check
    try:
        raw_llm = llm_provider.inner if hasattr(llm_provider, 'inner') else llm_provider
        provider_name = raw_llm.__class__.__name__ if hasattr(raw_llm, '__class__') else "unknown"
        client_active = getattr(raw_llm, '_client', None) is not None
        llm_status = "ok"
    except Exception as e:
        llm_status = f"error: {str(e)}"
        provider_name = "error"
        client_active = False
        overall_status = "degraded"

    components["llm_provider"] = {
        "status": llm_status,
        "provider_class": provider_name,
        "configured_provider": settings.llm.provider,
        "model_name": settings.llm.model_name,
        "client_initialized": client_active
    }

    return {
        "status": overall_status,
        "service": settings.app.name,
        "version": settings.app.version,
        "environment": settings.server.env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components
    }


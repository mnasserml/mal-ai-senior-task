from dependency_injector import containers, providers

from src.config import load_settings
from src.infrastructure.pii.presidio_adapter import PresidioPIIAdapter
from src.infrastructure.pii.aajil_adapter import AajilPIIAdapter
from src.infrastructure.guardrails.scope_guard import KeywordScopeGuardAdapter, LLMScopeGuardAdapter
from src.infrastructure.session.memory_store import InMemorySessionStore
from src.infrastructure.vector_store.chroma_adapter import ChromaVectorAdapter
from src.infrastructure.llm.openai_adapter import OpenAIAdapter
from src.infrastructure.llm.gemini_adapter import GeminiAdapter

from src.infrastructure.observability.decorators import (
    ObservablePIIAdapter,
    ObservableScopeAdapter,
    ObservableSessionAdapter,
    ObservableVectorAdapter,
    ObservableLLMAdapter
)
from src.application.chat_use_case import ChatUseCase
from src.infrastructure.evaluation.llm_judge import LLMGroundingJudge

from src.infrastructure.vector_store.embedding_adapters import SentenceTransformerEmbeddingAdapter, GoogleGeminiEmbeddingAdapter

class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["src.api.routes"])

    # Load Settings (single source of truth)
    settings = providers.Singleton(load_settings)

    # Dynamic PII Provider selection (Aajil / Presidio)
    def _create_pii_adapter(settings_obj):
        engine_type = getattr(settings_obj.pii, "engine", "aajil").lower()
        if engine_type == "aajil":
            return AajilPIIAdapter()
        return PresidioPIIAdapter(ner_model=settings_obj.pii.ner_model)

    # Base PII Adapter
    base_pii_adapter = providers.Singleton(
        _create_pii_adapter,
        settings_obj=settings
    )

    base_session_store = providers.Singleton(
        InMemorySessionStore,
        max_history_turns=settings.provided.session.max_history_turns
    )

    # Dynamic Embedding Provider selection (SentenceTransformer / Google Gemini)
    def _create_embedding_provider(settings_obj):
        provider = (settings_obj.vector_store.embedding_provider or "sentence_transformer").lower()
        if "google" in provider or "gemini" in provider:
            return GoogleGeminiEmbeddingAdapter(
                model_name=settings_obj.vector_store.embedding_model or "gemini-embedding-001",
                api_key=settings_obj.gemini_api_key
            )
        return SentenceTransformerEmbeddingAdapter(
            model_name=settings_obj.vector_store.embedding_model
        )

    base_embedding_provider = providers.Singleton(
        _create_embedding_provider,
        settings_obj=settings
    )

    base_vector_store = providers.Singleton(
        ChromaVectorAdapter,
        collection_name=settings.provided.vector_store.collection_name,
        enable_chroma_download=settings.provided.vector_store.enable_chroma_download,
        embedding_provider=base_embedding_provider
    )

    # Dynamic LLM Provider selection (OpenAI / Gemini)
    def _create_llm(settings_obj):
        provider = (settings_obj.llm.provider or "openai").lower()
        if "gemini" in provider or "google" in provider:
            return GeminiAdapter(
                model_name=settings_obj.llm.model_name,
                api_key=settings_obj.gemini_api_key,
                temperature=settings_obj.llm.temperature,
                max_tokens=settings_obj.llm.max_tokens,
                enable_reasoning=settings_obj.llm.enable_reasoning
            )
        return OpenAIAdapter(
            model_name=settings_obj.llm.model_name,
            api_key=settings_obj.openai_api_key,
            base_url=settings_obj.llm.base_url,
            temperature=settings_obj.llm.temperature,
            max_tokens=settings_obj.llm.max_tokens,
            enable_reasoning=settings_obj.llm.enable_reasoning
        )

    base_llm_provider = providers.Singleton(
        _create_llm,
        settings_obj=settings
    )

    # LLM Grounding Judge Evaluator (prompt from config/prompts.yaml)
    def _create_eval_judge(settings_obj, llm_provider):
        return LLMGroundingJudge(
            llm_provider=llm_provider,
            judge_prompt=settings_obj.prompts.llm_judge_system_prompt
        )

    base_eval_judge = providers.Singleton(
        _create_eval_judge,
        settings_obj=settings,
        llm_provider=base_llm_provider
    )

    # Dynamic Scope Guard Selection (Keyword vs LLM Adapter based on config)
    def _create_scope_guard(settings_obj, llm_provider):
        classifier_type = (settings_obj.guardrails.classifier_type or "keyword").lower()
        if "llm" in classifier_type:
            return LLMScopeGuardAdapter(
                llm_provider=llm_provider,
                classifier_prompt=settings_obj.prompts.scope_guard_classifier_prompt,
                refusal_message_en=settings_obj.guardrails.refusal_message_en,
                refusal_message_ar=settings_obj.guardrails.refusal_message_ar
            )
        return KeywordScopeGuardAdapter(
            similarity_threshold=settings_obj.guardrails.similarity_threshold,
            refusal_message_en=settings_obj.guardrails.refusal_message_en,
            refusal_message_ar=settings_obj.guardrails.refusal_message_ar
        )

    base_scope_guard = providers.Singleton(
        _create_scope_guard,
        settings_obj=settings,
        llm_provider=base_llm_provider
    )

    # Observable Decorated Ports
    observable_pii = providers.Singleton(ObservablePIIAdapter, inner=base_pii_adapter)
    observable_scope = providers.Singleton(ObservableScopeAdapter, inner=base_scope_guard)
    observable_session = providers.Singleton(ObservableSessionAdapter, inner=base_session_store)
    observable_vector = providers.Singleton(ObservableVectorAdapter, inner=base_vector_store)
    observable_llm = providers.Singleton(ObservableLLMAdapter, inner=base_llm_provider)

    # Application Use Case (system_prompt from config/prompts.yaml)
    chat_use_case = providers.Factory(
        ChatUseCase,
        pii_redactor=observable_pii,
        scope_guard=observable_scope,
        session_store=observable_session,
        vector_store=observable_vector,
        llm_provider=observable_llm,
        system_prompt=settings.provided.prompts.chat_system_prompt,
        eval_judge=base_eval_judge,
        eval_config=settings.provided.eval
    )

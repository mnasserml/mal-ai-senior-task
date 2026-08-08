import random
import time
import concurrent.futures
from typing import Optional

from src.domain.ports import (
    IPIIRedactor,
    IScopeGuard,
    ISessionStore,
    IVectorStore,
    ILLMProvider
)
from src.domain.models import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    LLMGenerationStepTrace,
    LLMJudgeEvalStepTrace
)
from src.infrastructure.observability.tracing import get_current_trace
from src.infrastructure.evaluation.llm_judge import LLMGroundingJudge
from src.config import EvalConfig

_eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="eval_judge")

class ChatUseCase:
    def __init__(
        self,
        pii_redactor: IPIIRedactor,
        scope_guard: IScopeGuard,
        session_store: ISessionStore,
        vector_store: IVectorStore,
        llm_provider: ILLMProvider,
        system_prompt: str = "",
        eval_judge: Optional[LLMGroundingJudge] = None,
        eval_config: Optional[EvalConfig] = None
    ):
        self.pii_redactor = pii_redactor
        self.scope_guard = scope_guard
        self.session_store = session_store
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt
        self.eval_judge = eval_judge
        self.eval_config = eval_config or EvalConfig()

    def execute(self, request: ChatRequest) -> ChatResponse:
        start_time = time.perf_counter()

        # Step 1: Detect and Redact PII
        lang = request.language or ("ar" if any('\u0600' <= c <= '\u06FF' for c in request.message) else "en")
        pii_result = self.pii_redactor.redact(request.message, language=lang)
        redacted_user_message = pii_result.redacted_text

        history = self.session_store.get_history(request.session_id, limit=5)

        # Step 2: Retrieve Context from Vector Store
        retrieval_result = self.vector_store.search(redacted_user_message, top_k=3)

        # Step 3: Check Scope Guardrail (Early refusal if out-of-scope)
        is_in_scope, refusal_message = self.scope_guard.is_in_scope(redacted_user_message, retrieval_result)
        if not is_in_scope and refusal_message:
            assistant_msg = ChatMessage(role="assistant", content=refusal_message)
            self.session_store.add_message(request.session_id, ChatMessage(role="user", content=redacted_user_message))
            self.session_store.add_message(request.session_id, assistant_msg)
            
            final_trace = get_current_trace()
            if final_trace:
                final_trace.total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            return ChatResponse(
                session_id=request.session_id,
                response=refusal_message,
                trace=final_trace
            )

        # Step 4: Call LLM with system prompt from config
        messages_for_llm = history + [ChatMessage(role="user", content=redacted_user_message)]
        llm_response = self.llm_provider.generate_response(
            messages=messages_for_llm,
            context_chunks=retrieval_result.chunks,
            system_prompt=self.system_prompt
        )

        trace = get_current_trace()
        if trace:
            trace.prompt_tokens = llm_response.prompt_tokens
            trace.completion_tokens = llm_response.completion_tokens
            trace.total_tokens = llm_response.prompt_tokens + llm_response.completion_tokens
            trace.steps.step_4_llm_generation = LLMGenerationStepTrace(
                latency_ms=round(trace.llm_latency_ms, 2),
                model_name=llm_response.model_name,
                response_text=llm_response.content,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.prompt_tokens + llm_response.completion_tokens
            )

        # Step 5: Trigger Sampled/Async LLM Judge Grounding Evaluation
        if self.eval_judge and self.eval_config and self.eval_config.enabled:
            if random.random() <= self.eval_config.sample_rate:
                context_str = "\n\n".join([f"[{c.doc_name}]: {c.content}" for c in retrieval_result.chunks])

                def _evaluate_background(trace_obj, q, ctx, ans):
                    t_start = time.perf_counter()
                    eval_res = self.eval_judge.evaluate_grounding(q, ctx, ans)
                    t_elapsed = (time.perf_counter() - t_start) * 1000.0
                    if trace_obj:
                        trace_obj.eval_latency_ms = t_elapsed
                        trace_obj.is_grounded = eval_res.is_grounded
                        trace_obj.is_relevant = eval_res.is_relevant
                        trace_obj.groundedness_score = eval_res.groundedness_score
                        trace_obj.relevance_score = eval_res.relevance_score
                        trace_obj.overall_score = eval_res.overall_score
                        trace_obj.reasoning = eval_res.reasoning
                        
                        judge_total = eval_res.prompt_tokens + eval_res.completion_tokens
                        trace_obj.steps.step_5_llm_judge_eval = LLMJudgeEvalStepTrace(
                            latency_ms=round(t_elapsed, 2),
                            is_grounded=eval_res.is_grounded,
                            is_relevant=eval_res.is_relevant,
                            groundedness_score=round(eval_res.groundedness_score, 2),
                            relevance_score=round(eval_res.relevance_score, 2),
                            overall_score=round(eval_res.overall_score, 2),
                            reasoning=eval_res.reasoning,
                            prompt_tokens=eval_res.prompt_tokens,
                            completion_tokens=eval_res.completion_tokens,
                            total_tokens=judge_total
                        )

                        # Aggregate judge tokens into top-level trace
                        trace_obj.prompt_tokens += eval_res.prompt_tokens
                        trace_obj.completion_tokens += eval_res.completion_tokens
                        trace_obj.total_tokens = trace_obj.prompt_tokens + trace_obj.completion_tokens

                if self.eval_config.async_eval:
                    _eval_executor.submit(_evaluate_background, trace, redacted_user_message, context_str, llm_response.content)
                else:
                    _evaluate_background(trace, redacted_user_message, context_str, llm_response.content)

        # Step 6: Store Messages in Session Memory & Return Response
        self.session_store.add_message(request.session_id, ChatMessage(role="user", content=redacted_user_message))
        self.session_store.add_message(request.session_id, ChatMessage(role="assistant", content=llm_response.content))

        final_trace = get_current_trace()
        if final_trace:
            final_trace.total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return ChatResponse(
            session_id=request.session_id,
            response=llm_response.content,
            trace=final_trace
        )

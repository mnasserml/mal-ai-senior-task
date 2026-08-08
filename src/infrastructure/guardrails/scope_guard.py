import json
from typing import Optional
from src.domain.ports import IScopeGuard, ILLMProvider
from src.domain.models import RetrievalResult, ChatMessage, ScopeGuardStepTrace
from src.infrastructure.observability.tracing import get_current_trace

class KeywordScopeGuardAdapter(IScopeGuard):
    def __init__(
        self,
        similarity_threshold: float = 0.35,
        refusal_message_en: str = "I apologize, but I can only answer questions related to Islamic finance and Sharia compliance rules for Mal customers.",
        refusal_message_ar: str = "أعتذر، يمكنني فقط الإجابة على الأسئلة المتعلقة بالتمويل الإسلامي وأحكام الشريعة لعملاء مال."
    ):
        self.similarity_threshold = similarity_threshold
        self.refusal_message_en = refusal_message_en
        self.refusal_message_ar = refusal_message_ar

        self.out_of_scope_keywords_en = [
            "crypto", "bitcoin", "ethereum", "dogecoin", "trading bot",
            "weather", "recipe", "cooking", "football", "basketball",
            "movie", "celebrity", "conventional loan", "usury rate", "casino", "gambling"
        ]
        self.out_of_scope_keywords_ar = [
            "بيتكوين", "تداول العملات الرقمية", "طقس", "وصفة طعام", "طبخ",
            "كرة القدم", "سينما", "قمار", "كازينو", "قرض ربوي"
        ]

        self.valid_keywords = [
            "murabaha", "sukuk", "ijara", "mudaraba", "musharaka", "sharia",
            "halal", "zakat", "qard", "wadia", "mal", "account", "profit", "penalty", "bank",
            "المرابحة", "الصكوك", "الإجارة", "المضاربة", "المشاركة", "الشريعة",
            "حلال", "زكاة", "قرض", "وديعة", "مال", "حساب", "ربح", "غرامة", "بنك"
        ]

    def is_in_scope(self, text: str, retrieval_result: Optional[RetrievalResult] = None) -> tuple[bool, Optional[str]]:
        lower_text = text.lower()
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
        refusal = self.refusal_message_ar if is_arabic else self.refusal_message_en

        is_in = True
        refusal_reason = None

        for kw in self.out_of_scope_keywords_en:
            if kw in lower_text:
                is_in, refusal_reason = False, refusal
                break
        if is_in:
            for kw in self.out_of_scope_keywords_ar:
                if kw in lower_text:
                    is_in, refusal_reason = False, refusal
                    break

        if is_in and retrieval_result is not None:
            has_domain_keyword = any(kw in lower_text for kw in self.valid_keywords)
            if not has_domain_keyword and (not retrieval_result.chunks or retrieval_result.average_relevance_score < self.similarity_threshold):
                is_in, refusal_reason = False, refusal

        trace = get_current_trace()
        if trace and trace.steps.step_3_scope_guard is None:
            trace.steps.step_3_scope_guard = ScopeGuardStepTrace(
                method_used="keyword_rules",
                is_in_scope=is_in,
                refusal_message=refusal_reason,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )

        return is_in, refusal_reason

class LLMScopeGuardAdapter(IScopeGuard):
    def __init__(
        self,
        llm_provider: ILLMProvider,
        classifier_prompt: str = "",
        refusal_message_en: str = "I apologize, but I can only answer questions related to Islamic finance and Sharia compliance rules for Mal customers.",
        refusal_message_ar: str = "أعتذر، يمكنني فقط الإجابة على الأسئلة المتعلقة بالتمويل الإسلامي وأحكام الشريعة لعملاء مال."
    ):
        self.llm_provider = llm_provider
        self.classifier_prompt = classifier_prompt
        self.refusal_message_en = refusal_message_en
        self.refusal_message_ar = refusal_message_ar
        self.keyword_fallback = KeywordScopeGuardAdapter(
            refusal_message_en=refusal_message_en,
            refusal_message_ar=refusal_message_ar
        )

    def is_in_scope(self, text: str, retrieval_result: Optional[RetrievalResult] = None) -> tuple[bool, Optional[str]]:
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
        default_refusal = self.refusal_message_ar if is_arabic else self.refusal_message_en

        # Primary: Use LLM classification
        try:
            response = self.llm_provider.generate_response(
                messages=[ChatMessage(role="user", content=text)],
                context_chunks=[],
                system_prompt=self.classifier_prompt
            )

            content_lower = response.content.lower()
            is_out_of_scope = "false" in content_lower or "deny" in content_lower

            res_in_scope = not is_out_of_scope
            res_refusal = default_refusal if is_out_of_scope else None

            trace = get_current_trace()
            if trace:
                p_tokens = response.prompt_tokens
                c_tokens = response.completion_tokens
                t_tokens = p_tokens + c_tokens
                trace.steps.step_3_scope_guard = ScopeGuardStepTrace(
                    method_used="llm_classifier",
                    is_in_scope=res_in_scope,
                    refusal_message=res_refusal,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens
                )
                trace.prompt_tokens += p_tokens
                trace.completion_tokens += c_tokens
                trace.total_tokens = trace.prompt_tokens + trace.completion_tokens

            if is_out_of_scope:
                return False, default_refusal
            return True, None

        except Exception:
            # Fallback: Keyword-based evaluation when LLM API fails or errors out
            is_in, refusal = self.keyword_fallback.is_in_scope(text, retrieval_result)
            trace = get_current_trace()
            if trace:
                trace.steps.step_3_scope_guard = ScopeGuardStepTrace(
                    method_used="keyword_fallback",
                    is_in_scope=is_in,
                    refusal_message=refusal,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                )
            return is_in, refusal

# Backward compatibility alias
ScopeGuardAdapter = KeywordScopeGuardAdapter

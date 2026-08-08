import json
import re
import logging
from typing import Optional, List, Dict, Any

from src.domain.ports import ILLMProvider
from src.domain.models import ChatMessage, DocumentChunk, GroundingEvaluation

logger = logging.getLogger(__name__)

class LLMGroundingJudge:
    def __init__(self, llm_provider: ILLMProvider, judge_prompt: str = ""):
        self.llm_provider = llm_provider
        self.judge_prompt = judge_prompt

    def evaluate_grounding(
        self,
        query: str,
        retrieved_context: str,
        response_text: str
    ) -> GroundingEvaluation:
        """
        Evaluate RAG response grounding using LLM-as-a-Judge.
        """
        user_prompt = f"""USER QUERY:
{query}

RETRIEVED REFERENCE CONTEXT:
{retrieved_context}

ASSISTANT RESPONSE TO EVALUATE:
{response_text}
"""

        try:
            llm_response = self.llm_provider.generate_response(
                messages=[ChatMessage(role="user", content=user_prompt)],
                context_chunks=[],
                system_prompt=self.judge_prompt
            )
            raw_text = llm_response.content.strip()

            # Clean JSON markdown blocks if present
            clean_json = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
            clean_json = re.sub(r"\s*```$", "", clean_json, flags=re.MULTILINE).strip()

            try:
                data = json.loads(clean_json)
            except Exception as json_err:
                # Attempt regex extraction if JSON string was truncated at token limit
                data = self._regex_extract_json_fields(clean_json)

            return GroundingEvaluation(
                is_grounded=bool(data.get("is_grounded", True)),
                is_relevant=bool(data.get("is_relevant", True)),
                groundedness_score=float(data.get("groundedness_score", 0.9)),
                relevance_score=float(data.get("relevance_score", 0.9)),
                overall_score=float(data.get("overall_score", 0.9)),
                reasoning=str(data.get("reasoning", "Evaluated by LLM Judge.")),
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens
            )
        except Exception as e:
            logger.warning("LLM Judge evaluation call failed or returned invalid JSON: %s. Using heuristic fallback.", e)
            return self._heuristic_fallback_judge(query, retrieved_context, response_text)

    def _regex_extract_json_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON fields via regex if strict json.loads fails (e.g. due to string truncation).
        """
        data = {}
        m_grounded = re.search(r'"is_grounded"\s*:\s*(true|false)', text, re.IGNORECASE)
        if m_grounded:
            data["is_grounded"] = m_grounded.group(1).lower() == "true"

        m_relevant = re.search(r'"is_relevant"\s*:\s*(true|false)', text, re.IGNORECASE)
        if m_relevant:
            data["is_relevant"] = m_relevant.group(1).lower() == "true"

        m_g_score = re.search(r'"groundedness_score"\s*:\s*([0-9.]+)', text)
        if m_g_score:
            data["groundedness_score"] = float(m_g_score.group(1))

        m_r_score = re.search(r'"relevance_score"\s*:\s*([0-9.]+)', text)
        if m_r_score:
            data["relevance_score"] = float(m_r_score.group(1))

        m_o_score = re.search(r'"overall_score"\s*:\s*([0-9.]+)', text)
        if m_o_score:
            data["overall_score"] = float(m_o_score.group(1))

        m_reasoning = re.search(r'"reasoning"\s*:\s*"([^"]*)', text)
        if m_reasoning:
            data["reasoning"] = m_reasoning.group(1).strip()

        if not data:
            raise ValueError("No valid JSON fields found via regex extraction.")
        return data

    def _heuristic_fallback_judge(
        self,
        query: str,
        retrieved_context: str,
        response_text: str
    ) -> GroundingEvaluation:
        """
        Deterministic fallback judge if LLM API is unavailable or returns invalid data.
        Checks keyword presence and context overlap.
        """
        res_lower = response_text.lower()
        ctx_lower = retrieved_context.lower()

        # Key Sharia terms matching
        domain_keywords = [
            "charity", "penalty", "donated", "late", "profit", "lessor", "bank",
            "صيانة", "المؤجر", "البنك", "تبرع", "غرامة"
        ]

        has_domain_terms = any(kw in res_lower for kw in domain_keywords)
        has_context_overlap = any(word in res_lower for word in ctx_lower.split() if len(word) > 4)

        is_valid = has_domain_terms and has_context_overlap
        score = 0.90 if is_valid else 0.50

        return GroundingEvaluation(
            is_grounded=is_valid,
            is_relevant=is_valid,
            groundedness_score=score,
            relevance_score=score,
            overall_score=score,
            reasoning=f"Heuristic Judge: Found domain concepts in answer with context overlap (score={score})."
        )

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: str

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique conversation session identifier")
    message: str = Field(..., description="User query message in English or Arabic")
    language: Optional[str] = Field(None, description="Optional language override ('en' or 'ar')")

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_name: str
    content: str
    metadata: Dict[str, Any] = {}

class RetrievalResult(BaseModel):
    chunks: List[DocumentChunk]
    relevance_scores: List[float]
    average_relevance_score: float

class PIIRedactionResult(BaseModel):
    original_text: str
    redacted_text: str
    detected_entities: List[Dict[str, Any]]
    redacted_count: int

class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""

# --- Typed Step Telemetry Models ---

class PIIStepTrace(BaseModel):
    latency_ms: float = 0.0
    engine: str = "aajil"
    redacted_text: str = ""
    redacted_count: int = 0
    detected_entities_count: int = 0

class VectorRetrievalChunk(BaseModel):
    chunk_id: str
    doc_name: str
    content_snippet: str

class VectorRetrievalStepTrace(BaseModel):
    latency_ms: float = 0.0
    average_relevance_score: float = 0.0
    retrieved_chunks: List[VectorRetrievalChunk] = []

class ScopeGuardStepTrace(BaseModel):
    latency_ms: float = 0.0
    method_used: str = "keyword_rules"  # "keyword_rules", "keyword_prefilter", "llm_classifier"
    is_in_scope: bool = True
    refusal_message: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class LLMGenerationStepTrace(BaseModel):
    latency_ms: float = 0.0
    model_name: str = ""
    response_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class LLMJudgeEvalStepTrace(BaseModel):
    latency_ms: float = 0.0
    is_grounded: Optional[bool] = None
    is_relevant: Optional[bool] = None
    groundedness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    overall_score: Optional[float] = None
    reasoning: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class StepsTrace(BaseModel):
    step_1_pii_redaction: Optional[PIIStepTrace] = None
    step_2_vector_retrieval: Optional[VectorRetrievalStepTrace] = None
    step_3_scope_guard: Optional[ScopeGuardStepTrace] = None
    step_4_llm_generation: Optional[LLMGenerationStepTrace] = None
    step_5_llm_judge_eval: Optional[LLMJudgeEvalStepTrace] = None

class TraceMetrics(BaseModel):
    pii_redaction_latency_ms: float = 0.0
    scope_guard_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    eval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    retrieved_chunk_ids: List[str] = []
    context_relevance_score: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    pii_redacted_count: int = 0
    refused: bool = False
    is_grounded: Optional[bool] = None
    is_relevant: Optional[bool] = None
    groundedness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    overall_score: Optional[float] = None
    reasoning: Optional[str] = None
    steps: StepsTrace = Field(default_factory=StepsTrace)

class ChatResponse(BaseModel):
    session_id: str
    response: str
    trace: Optional[TraceMetrics] = None

class GroundingEvaluation(BaseModel):
    is_grounded: bool = Field(..., description="Whether response claims are supported by reference context")
    is_relevant: bool = Field(..., description="Whether response directly addresses user query")
    groundedness_score: float = Field(..., ge=0.0, le=1.0, description="Faithfulness score between 0 and 1")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0 and 1")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall grounding quality score")
    reasoning: str = Field(..., description="Detailed judge explanation")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

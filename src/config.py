import os
import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Config Models — All non-secret settings from config/config.yaml
# ──────────────────────────────────────────────────────────────

class CustomRecognizerConfig(BaseModel):
    name: str
    regex: str
    score: float = 0.95

class AppConfig(BaseModel):
    name: str = "Mal Islamic Finance RAG Assistant"
    version: str = "1.0.0"
    supported_languages: List[str] = ["en", "ar"]

class ServerConfig(BaseModel):
    env: str = "development"
    port: int = 8000
    log_level: str = "INFO"

class LLMConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 500
    enable_reasoning: bool = False

class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    collection_name: str = "sharia_knowledge_base"
    embedding_provider: str = "sentence_transformer"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 3
    similarity_threshold: float = 0.35
    enable_chroma_download: bool = False

class PIIConfig(BaseModel):
    engine: str = "presidio"
    default_language: str = "en"
    supported_languages: List[str] = ["en", "ar"]
    ner_model: str = "xx_ent_wiki_sm"
    custom_recognizers: List[CustomRecognizerConfig] = []

class GuardrailsConfig(BaseModel):
    enabled: bool = True
    classifier_type: str = "llm_hybrid"
    similarity_threshold: float = 0.35
    refusal_message_en: str = "I apologize, but I can only answer questions related to Islamic finance and Sharia compliance rules for Mal customers."
    refusal_message_ar: str = "أعتذر، يمكنني فقط الإجابة على الأسئلة المتعلقة بالتمويل الإسلامي وأحكام الشريعة لعملاء مال."

class SessionConfig(BaseModel):
    provider: str = "in_memory"
    max_history_turns: int = 5

class EvalConfig(BaseModel):
    enabled: bool = True
    sample_rate: float = 1.0
    async_eval: bool = True

class PromptsConfig(BaseModel):
    chat_system_prompt: str = (
        "You are Mal's AI Sharia Assistant. You answer customer questions strictly based on the provided "
        "Sharia finance documents and rules. Provide clear, accurate, and professional answers in the same "
        "language as the customer query (English or Arabic). If the answer cannot be found in the context, "
        "state that clearly and do not hallucinate."
    )
    scope_guard_classifier_prompt: str = (
        "You are an intent classifier for Mal Bank. Classify if the user query is about Islamic finance, "
        "Sharia rules, bank accounts, or banking services (ALLOW) vs completely unrelated topics like crypto trading, "
        "weather, recipes, or gambling (DENY). Respond strictly with valid JSON: {\"is_in_scope\": true/false}"
    )
    llm_judge_system_prompt: str = ""

# ──────────────────────────────────────────────────────────────
# Unified Settings — Single source of truth
# ──────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    # Config sections (from config.yaml)
    app: AppConfig = AppConfig()
    server: ServerConfig = ServerConfig()
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    pii: PIIConfig = PIIConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    session: SessionConfig = SessionConfig()
    eval: EvalConfig = EvalConfig()
    prompts: PromptsConfig = PromptsConfig()

    # Secrets (from .env only)
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

def load_settings(
    config_path: str = "config/config.yaml",
    prompts_path: str = "config/prompts.yaml"
) -> Settings:
    # Load config.yaml
    yaml_data: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    # Load prompts.yaml
    prompts_data: Dict[str, Any] = {}
    if os.path.exists(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts_data = yaml.safe_load(f) or {}

    # Normalize base_url: treat empty strings as None
    llm_dict = yaml_data.get("llm", {})
    base_url = llm_dict.get("base_url")
    if base_url is not None and not str(base_url).strip():
        llm_dict["base_url"] = None

    settings = Settings(
        app=AppConfig(**yaml_data.get("app", {})),
        server=ServerConfig(**yaml_data.get("server", {})),
        llm=LLMConfig(**llm_dict),
        vector_store=VectorStoreConfig(**yaml_data.get("vector_store", {})),
        pii=PIIConfig(**yaml_data.get("pii", {})),
        guardrails=GuardrailsConfig(**yaml_data.get("guardrails", {})),
        session=SessionConfig(**yaml_data.get("session", {})),
        eval=EvalConfig(**yaml_data.get("eval", {})),
        prompts=PromptsConfig(**prompts_data),
        # Secrets — only values read from .env
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return settings

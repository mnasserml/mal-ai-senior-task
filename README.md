# Mal Islamic Finance RAG Assistant API

Production-ready RAG Assistant REST API for Mal customers querying Sharia finance rules and account context. Supports bilingual queries (English & Arabic), Microsoft Presidio PII redaction (Emirates ID, UAE IBAN, Account numbers, Names), out-of-scope refusal guardrails, pluggable vector embedding providers (Google Gemini & SentenceTransformers), LLM-as-a-Judge quality evaluation, structured JSON trace logging, and full Dependency Injection.

---

## Architecture Highlights
- **Simple Layered Architecture (Ports & Adapters)**: Clean decoupling of core domain logic from framework dependencies.
- **Pluggable Embedding Providers**: Abstracted behind `IEmbeddingProvider` port supporting **Google Gemini API** (`gemini-embedding-001`) and local `SentenceTransformer` models (`LaBSE`).
- **Dependency Injection**: Dynamic component wiring using `dependency-injector` configured via `config/config.yaml`.
- **Observable Decorator Pattern**: Transparent trace telemetry capture without modifying domain code.
- **Bilingual Sharia RAG**: Built-in Sharia knowledge base documents (`Murabaha`, `Sukuk`, `Ijara`, `Mudaraba`, `Account Rules`) in English and Arabic.
- **UAE PDPL Security**: Automated PII masking for Emirates ID (`784-YYYY-XXXXXXX-X`), IBAN (`AE...`), Account numbers, and bilingual English/Arabic Names powered by **Aajil-Labs int8 ONNX engine (`apii`)** and Microsoft Presidio (`~470MB RAM`, `<10ms` inference).
- **LLM-as-a-Judge Observability**: Async sampled grounding & relevance evaluation on production traffic.

---

## Configuration

The project uses a **three-layer configuration** architecture:

| File | Purpose |
|------|---------|
| `.env` | **Secrets only** — API keys `GEMINI_API_KEY`, `OPENAI_API_KEY` |
| `config/config.yaml` | **Single source of truth** — all non-secret settings (server, LLM provider, vector store & embedding provider, PII, guardrails, eval) |
| `config/prompts.yaml` | **All LLM prompts** — chat system prompt, scope guard classifier, LLM judge evaluation prompt |

### Vector Store & Embedding Options in `config/config.yaml`:
```yaml
vector_store:
  provider: "chroma"
  collection_name: "sharia_knowledge_base"
  embedding_provider: "google_gemini"  # Options: google_gemini, sentence_transformer
  embedding_model: "gemini-embedding-001"  # Options: gemini-embedding-001, sentence-transformers/LaBSE
  top_k: 3
  similarity_threshold: 0.35
  enable_chroma_download: true
```

---

## Quickstart

### 1. Environment & Dependencies (uv)
```bash
# Install dependencies using uv
uv sync

# Copy environment template and add your API key
cp .env.example .env
```

### 2. Configure
Edit `config/config.yaml` for LLM provider, embedding model selection, thresholds, and feature toggles.
Edit `config/prompts.yaml` to customize LLM prompts without code changes.

### 3. Run API Locally
```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

### 4. Run Automated Evaluation Tests
```bash
uv run pytest -v tests/
```

---

## API Endpoints

### 1. `POST /chat`
**Stateful Conversation Endpoint**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo_session_1",
    "message": "My name is Ahmed, Emirates ID 784-1990-1234567-1. What happens if I delay payment in Murabaha?"
  }'
```

**Response Payload**:
```json
{
  "session_id": "demo_session_1",
  "response": "If you delay payment in a Murabaha contract, any imposed late penalty fee cannot be taken as profit by Mal Bank. According to the **Charity Clause (شرط التبرع)**, the entire penalty amount must be donated to approved charitable causes.",
  "trace": {
    "pii_redaction_latency_ms": 274.62,
    "scope_guard_latency_ms": 5554.17,
    "retrieval_latency_ms": 0.88,
    "llm_latency_ms": 8475.18,
    "eval_latency_ms": 10102.86,
    "total_latency_ms": 24408.6,
    "retrieved_chunk_ids": ["murabaha.md_chunk_2", "murabaha.md_chunk_0", "murabaha.md_chunk_1"],
    "context_relevance_score": 0.5706,
    "prompt_tokens": 1023,
    "completion_tokens": 145,
    "total_tokens": 1168,
    "pii_redacted_count": 2,
    "refused": false,
    "is_grounded": true,
    "is_relevant": true,
    "groundedness_score": 1.0,
    "relevance_score": 1.0,
    "overall_score": 1.0,
    "reasoning": "The response is perfectly grounded in the provided context and directly answers the user's question regarding late payments in Murabaha.",
    "steps": {
      "step_1_pii_redaction": {
        "latency_ms": 274.62,
        "redacted_text": "My name is <PERSON>, Emirates ID <EMIRATES_ID>. What happens if I delay payment in Murabaha?",
        "redacted_count": 2,
        "detected_entities_count": 2
      },
      "step_2_vector_retrieval": {
        "latency_ms": 0.88,
        "average_relevance_score": 0.5706,
        "retrieved_chunks": [
          {
            "chunk_id": "murabaha.md_chunk_2",
            "doc_name": "murabaha.md",
            "content_snippet": "### Key Principles of Murabaha:\n1. **Asset Ownership**: Mal Bank..."
          }
        ]
      },
      "step_3_scope_guard": {
        "latency_ms": 5554.17,
        "method_used": "llm_classifier",
        "is_in_scope": true,
        "refusal_message": null,
        "prompt_tokens": 111,
        "completion_tokens": 9,
        "total_tokens": 120
      },
      "step_4_llm_generation": {
        "latency_ms": 8475.18,
        "model_name": "gemma-4-26b-a4b-it",
        "response_text": "If you delay payment in a Murabaha contract...",
        "prompt_tokens": 349,
        "completion_tokens": 52,
        "total_tokens": 401
      },
      "step_5_llm_judge_eval": {
        "latency_ms": 10102.86,
        "is_grounded": true,
        "is_relevant": true,
        "groundedness_score": 1.0,
        "relevance_score": 1.0,
        "overall_score": 1.0,
        "reasoning": "The response is perfectly grounded in the provided context...",
        "prompt_tokens": 674,
        "completion_tokens": 93,
        "total_tokens": 767
      }
    }
  }
}
```

### 2. `GET /health`
```bash
curl "http://localhost:8000/health"
```

---

## Project Structure

```
├── .env                    # Secrets only (API keys)
├── config/
│   ├── config.yaml         # All non-secret settings (single source of truth)
│   └── prompts.yaml        # All LLM prompt templates
├── data/sharia_docs/       # Sharia knowledge base (Markdown)
├── src/
│   ├── config.py           # Settings loader (YAML + secrets)
│   ├── api/                # FastAPI routes, middleware
│   ├── application/        # ChatUseCase pipeline
│   ├── domain/             # Ports (interfaces) & models
│   └── infrastructure/     # Adapters (LLM, PII, Vector, Guardrails, Eval)
├── tests/                  # Pytest evaluation suite
└── pyproject.toml
```

---

## Deployment (Railway / Render / Fly.io)

### Docker Deployment
```bash
docker build -t mal-assistant-api .
docker run -p 8000:8000 --env-file .env mal-assistant-api
```

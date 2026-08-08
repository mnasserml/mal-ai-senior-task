from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.models import (
    ChatMessage,
    DocumentChunk,
    RetrievalResult,
    PIIRedactionResult,
    LLMResponse
)

class IPIIRedactor(ABC):
    @abstractmethod
    def redact(self, text: str, language: str = "en") -> PIIRedactionResult:
        """Detect and redact PII from text."""
        pass

class IScopeGuard(ABC):
    @abstractmethod
    def is_in_scope(self, text: str, retrieval_result: Optional[RetrievalResult] = None) -> tuple[bool, Optional[str]]:
        """Check if request is in scope for Islamic finance. Returns (is_in_scope, optional_refusal_message)."""
        pass

class ISessionStore(ABC):
    @abstractmethod
    def get_history(self, session_id: str, limit: int = 5) -> List[ChatMessage]:
        """Fetch recent message history for session."""
        pass

    @abstractmethod
    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Save a new chat message to session history."""
        pass

class IVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[DocumentChunk]) -> None:
        """Index document chunks into vector database."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Search vector database and return matching chunks with relevance scores."""
        pass

class ILLMProvider(ABC):
    @abstractmethod
    def generate_response(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk],
        system_prompt: str
    ) -> LLMResponse:
        """Generate LLM response given context and chat history."""
        pass

class IEmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for document chunks (task_type=RETRIEVAL_DOCUMENT)."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate dense vector embedding for a query string (task_type=RETRIEVAL_QUERY)."""
        pass


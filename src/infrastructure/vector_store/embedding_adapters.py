import os
import logging
from typing import List, Optional
from src.domain.ports import IEmbeddingProvider

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingAdapter(IEmbeddingProvider):
    def __init__(self, model_name: str = "sentence-transformers/LaBSE"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error("Failed to load SentenceTransformer model '%s': %s", self.model_name, e)
                raise e
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()


class GoogleGeminiEmbeddingAdapter(IEmbeddingProvider):
    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: Optional[str] = None
    ):
        if model_name and not model_name.startswith("models/"):
            self.model_name = f"models/{model_name}"
        else:
            self.model_name = model_name or "models/gemini-embedding-001"
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                logger.warning("GEMINI_API_KEY is not configured for GoogleGeminiEmbeddingAdapter.")
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Google GenAI client for embeddings: %s", e)
                raise e
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        from google.genai import types
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            if hasattr(response, "embeddings") and response.embeddings:
                return [e.values for e in response.embeddings]
            return []
        except Exception as e:
            logger.error("Google Gemini embed_documents failed for model '%s': %s", self.model_name, e)
            raise RuntimeError(f"Gemini Embedding Error: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        from google.genai import types
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            if hasattr(response, "embeddings") and response.embeddings:
                return response.embeddings[0].values
            return []
        except Exception as e:
            logger.error("Google Gemini embed_query failed for model '%s': %s", self.model_name, e)
            raise RuntimeError(f"Gemini Embedding Error: {str(e)}") from e

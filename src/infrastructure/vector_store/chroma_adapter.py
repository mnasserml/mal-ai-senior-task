import math
import logging
from typing import List, Dict, Any, Optional
from src.domain.ports import IVectorStore, IEmbeddingProvider
from src.domain.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)

class ChromaVectorAdapter(IVectorStore):
    def __init__(
        self,
        collection_name: str = "sharia_knowledge_base",
        enable_chroma_download: bool = False,
        embedding_provider: Optional[IEmbeddingProvider] = None
    ):
        self.collection_name = collection_name
        self.documents: List[DocumentChunk] = []
        self._use_chroma = False
        self.embedding_provider = embedding_provider

        try:
            import chromadb
            if enable_chroma_download and self.embedding_provider is not None:
                self.client = chromadb.Client()
                self.collection = self.client.get_or_create_collection(name=collection_name)
                self._use_chroma = True
        except Exception as e:
            logger.warning("ChromaDB initialization fallback enabled: %s", e)
            self._use_chroma = False

    def add_documents(self, documents: List[DocumentChunk]) -> None:
        self.documents.extend(documents)
        if self._use_chroma and documents:
            texts = [doc.content for doc in documents]
            ids = [doc.chunk_id for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            if self.embedding_provider is not None:
                embeddings = self.embedding_provider.embed_documents(texts)
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        if not self.documents:
            return RetrievalResult(chunks=[], relevance_scores=[], average_relevance_score=0.0)

        if self._use_chroma:
            try:
                if self.embedding_provider is not None:
                    query_embedding = self.embedding_provider.embed_query(query)
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(top_k, len(self.documents))
                    )
                else:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=min(top_k, len(self.documents))
                    )
                
                retrieved_chunks = []
                scores = []
                if results and results.get("ids") and results["ids"][0]:
                    ids = results["ids"][0]
                    distances = results["distances"][0] if "distances" in results else [0.5] * len(ids)
                    
                    for doc_id, dist in zip(ids, distances):
                        sim_score = max(0.0, 1.0 - float(dist))
                        matching_doc = next((d for d in self.documents if d.chunk_id == doc_id), None)
                        if matching_doc:
                            retrieved_chunks.append(matching_doc)
                            scores.append(sim_score)
                
                avg_score = sum(scores) / len(scores) if scores else 0.0
                return RetrievalResult(
                    chunks=retrieved_chunks,
                    relevance_scores=scores,
                    average_relevance_score=avg_score
                )
            except Exception as e:
                logger.warning("Chroma search fallback triggered: %s", e)

        # Fallback: Word-overlap / Cosine term similarity
        return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int) -> RetrievalResult:
        import re
        clean_query = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', query.lower())
        query_words = set(w for w in clean_query.split() if len(w) > 2)
        scored_docs = []

        for doc in self.documents:
            content_lower = doc.content.lower()
            doc_words = set(re.sub(r'[^\w\s\u0600-\u06FF]', ' ', content_lower).split())
            
            intersection = query_words.intersection(doc_words)
            score = len(intersection) / max(1, math.sqrt(len(query_words) * max(1, len(doc_words))))
            
            domain_terms = [
                "murabaha", "sukuk", "ijara", "mudaraba", "wadia", "qard",
                "merabaha", "sharia", "charity", "penalty", "delay", "lease",
                "المرابحة", "الصكوك", "الإجارة", "المضاربة", "القرض", "التمليك", "الصيانة", "الهيكلية"
            ]
            for term in domain_terms:
                if term in clean_query and term in content_lower:
                    score += 0.35

            scored_docs.append((doc, min(1.0, score)))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_docs[:top_k]

        chunks = [item[0] for item in top_results if item[1] > 0.05]
        scores = [item[1] for item in top_results if item[1] > 0.05]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return RetrievalResult(
            chunks=chunks,
            relevance_scores=scores,
            average_relevance_score=avg_score
        )

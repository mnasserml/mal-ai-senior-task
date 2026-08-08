import os
import glob
from fastapi import FastAPI
from src.infrastructure.container import Container
from src.api.middleware import TracingMiddleware
from src.api.routes import router
from src.domain.models import DocumentChunk

def index_knowledge_base(container: Container):
    vector_store = container.observable_vector()
    docs_dir = os.path.join(os.path.dirname(__file__), "../../data/sharia_docs")
    md_files = glob.glob(os.path.join(docs_dir, "*.md"))

    document_chunks = []
    for filepath in md_files:
        doc_name = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        import re
        sections = re.split(r'\n(?=#+ )', content)
        for idx, section in enumerate(sections):
            if not section.strip():
                continue
            chunk_id = f"{doc_name}_chunk_{idx}"
            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_name=doc_name,
                    content=section.strip(),
                    metadata={"source": doc_name, "section": idx}
                )
            )

    vector_store.add_documents(document_chunks)

def create_app() -> FastAPI:
    container = Container()
    index_knowledge_base(container)
    
    app = FastAPI(
        title="Mal Islamic Finance RAG Assistant API",
        version="1.0.0",
        description="Production RAG Assistant REST API for Mal customers querying Sharia finance rules and account context."
    )
    
    app.container = container
    app.add_middleware(TracingMiddleware)
    app.include_router(router)

    return app

app = create_app()

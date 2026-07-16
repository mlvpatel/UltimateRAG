"""rag-multimodal-2026 API: multimodal RAG chat plus document and image management.

The chat endpoint retrieves across text and images at once and returns the
answer along with the images that contributed, so callers and the UI can see
which visual sources were used.
"""

import os
import shutil
import uuid
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.db_utils import (
    delete_document_record,
    get_all_documents,
    get_chat_history,
    init_db,
    insert_application_logs,
)
from src.api.pydantic_models import (
    DeleteFileRequest,
    DocumentInfo,
    QueryInput,
    QueryResponse,
)
from src.api.security import limiter, sanitize_question
from src.core.config import settings
from src.core.logging_config import configure_logging, logger
from src.multimodal.engine import run_multimodal
from src.multimodal.store import delete as delete_content
from src.worker.celery_app import celery_app
from src.worker.tasks import process_document

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".html",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        init_db()
    except Exception as exc:
        logger.warning("init_db skipped at startup: %s", exc)
    yield


app = FastAPI(title="rag-multimodal-2026 API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

v1 = APIRouter(prefix="/v1")


@app.get("/health")
def health():
    return {"status": "ok"}


@v1.post("/chat", response_model=QueryResponse)
@limiter.limit("60/minute")
def chat(request: Request, query: QueryInput):
    """Answer with multimodal retrieval and return the images that contributed."""
    session_id = query.session_id or str(uuid.uuid4())
    question = sanitize_question(query.question)
    history = get_chat_history(session_id)
    result = run_multimodal(query.model, question, history)
    answer = result["answer"]
    try:
        insert_application_logs(session_id, question, answer, query.model)
    except Exception as exc:
        logger.error("Failed to persist chat turn: %s", exc)
    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=query.model,
        steps=result["steps"],
        images=result.get("images", []),
        sources=result.get("sources", []),
    )


@v1.post("/upload-doc")
@limiter.limit("10/minute")
async def upload_doc(request: Request, file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    upload_dir = os.getenv("UPLOAD_DIR", "data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, os.path.basename(file.filename))
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    task = process_document.delay(path, file.filename)
    return {"task_id": task.id, "status": "processing", "filename": file.filename}


@v1.get("/list-docs", response_model=list[DocumentInfo])
def list_docs():
    return get_all_documents()


@v1.post("/delete-doc")
def delete_document(req: DeleteFileRequest):
    content_ok = delete_content(req.file_id)
    record_ok = delete_document_record(req.file_id)
    if not (content_ok and record_ok):
        raise HTTPException(status_code=500, detail="Delete failed")
    return {"status": "deleted", "file_id": req.file_id}


@v1.get("/task/{task_id}")
def task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": result.status}


app.include_router(v1)

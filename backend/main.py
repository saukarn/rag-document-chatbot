from fastapi import FastAPI, UploadFile, File, HTTPException

from backend.config import UPLOAD_DIR
from backend.ingestion import ingest_pdf
from backend.vector_store import add_documents_to_vector_store
from backend.rag_graph import answer_question
from backend.schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="Production RAG Engine",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "production-rag-engine"
    }


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported."
            )

        chunks = ingest_pdf(file, UPLOAD_DIR)
        chunk_count = add_documents_to_vector_store(chunks)

        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": file.filename,
            "chunks": chunk_count
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = answer_question(request.question)
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
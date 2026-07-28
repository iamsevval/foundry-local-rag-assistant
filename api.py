from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
from typing import List

from src.rag_core import answer_query
from src.document_processor import process_and_chunk_file
from src.vector_db import init_db

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    doc_id: str
    content: str
    rerank_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

app = FastAPI(
    title="Advanced Local RAG Engine API",
    description="Headless REST API for the Microsoft Foundry Local RAG system.",
    version="1.0.0"
)

# Initialize DB on startup
@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/health")
def health_check():
    """Check if the AI Engine is running."""
    return {"status": "ok", "message": "Advanced Local RAG Engine is running."}

@app.post("/query", response_model=QueryResponse)
def query_ai(request: QueryRequest):
    """
    Send a query to the Local AI and receive an answer with sources.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        # Call the core RAG pipeline (this handles retrieval, reranking, and generation)
        # Note: Streamlit's answer_query yields generator for streaming. 
        # For REST API, we consume the generator to return a complete JSON response.
        generator = answer_query(request.query)
        
        full_answer = ""
        context_used = []
        
        for chunk in generator:
            if isinstance(chunk, str):
                full_answer += chunk
            elif isinstance(chunk, list):
                # The generator yields the context list at the end
                context_used = chunk
                
        # Format the sources
        sources = [
            Source(
                doc_id=doc["doc_id"],
                content=doc["content"][:200] + "...",  # Preview
                rerank_score=doc.get("rerank_score", 0.0)
            ) for doc in context_used
        ]
        
        return QueryResponse(answer=full_answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF/TXT) to be processed and indexed by the Vector DB.
    """
    allowed_extensions = [".pdf", ".txt", ".docx"]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed: {allowed_extensions}")
        
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        chunks = process_and_chunk_file(temp_file_path)
        return {"status": "success", "message": f"File indexed successfully. Generated {len(chunks)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

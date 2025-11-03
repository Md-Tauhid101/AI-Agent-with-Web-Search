import os
import time
from typing import List, Dict, Any
import tempfile

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader

from agent import rag_agent
from vectorstore import add_document_to_vectorstore

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph RAG Agent API",
    description="API for the LangGraph-powered RAG agent with pinecone and Groq."
    # version="1.0.0"
)

# IN-MEMORY SESSION MANAGER
memory = MemorySaver()

# ------- PyDantic Models for API ---------
class TraceEvent(BaseModel):
    step: int
    node_name: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    event_type = str

class QueryRequest(BaseModel):
    session_id: str
    query: str
    enabled_web_search: bool = True

class AgentResponse(BaseModel):
    response: str
    trace_events: List[TraceEvent] = Field(default_factory=list)

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int

# Document upload endpoint
@app.post("/upload-document/", response_model=DocumentUploadResponse, status_code=status.HTTP_200_OK)
async def ipload_document(file: UploadFile=File(...)):
    """
        Uploads a PDF document, extract text, and adds it to the RAG knowledge base.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            details = "Only pdf files are supported"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        file_content = await file.read()
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    print(f"Recieved PDF for upload : {file.filename}. Saved temporarly to {temp_file_path}")

    try:
        loader = PyPDFLoader(temp_file_path)
        

    except Exception as e:
        print(f"Error processing PDF document : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details = f"Failed to process PDF : {e}"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temprorily file :{temp_file_path}")

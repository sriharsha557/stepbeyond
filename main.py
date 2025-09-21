from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os
import time
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import tempfile
import shutil

# Import updated modules
from crew_agents import setup_agent, StepBeyondAgent
from rag_pipeline import QdrantRAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StepBeyond API v2.0",
    description="AI-powered career counseling assistant API with Qdrant vector storage",
    version="2.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for serving HTML frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to store system instances
agent_instance: Optional[StepBeyondAgent] = None
rag_instance: Optional[QdrantRAGPipeline] = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    max_chunks: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    processing_time: float
    success: bool
    error: Optional[str] = None

class SystemStatus(BaseModel):
    rag_status: str
    rag_documents: int
    agent_status: str
    groq_api_configured: bool
    qdrant_configured: bool

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    total_chunks: int

# Check environment variables
def check_required_env_vars():
    """Check if all required environment variables are present"""
    required_vars = ['GROQ_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return False, missing_vars
    
    return True, []

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the system when the API starts"""
    global agent_instance, rag_instance
    
    try:
        logger.info("Starting StepBeyond API v2.0...")
        
        # Check environment variables
        env_ok, missing_vars = check_required_env_vars()
        if not env_ok:
            logger.warning(f"Missing environment variables: {missing_vars}")
            logger.warning("System will not be fully initialized until all variables are set")
            return
        
        # Initialize Qdrant RAG pipeline
        logger.info("Initializing Qdrant RAG pipeline...")
        rag_instance = QdrantRAGPipeline()
        
        # Initialize StepBeyond Agent
        logger.info("Initializing StepBeyond Agent...")
        agent_instance = setup_agent()
        
        logger.info("System initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}")
        # Don't crash the app, allow manual initialization

# Root endpoint - serve HTML frontend
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML frontend"""
    try:
        # Try to serve index.html from the current directory
        if os.path.exists("index.html"):
            with open("index.html", "r") as f:
                return HTMLResponse(content=f.read())
        else:
            # Fallback simple HTML
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>StepBeyond API v2.0</title>
            </head>
            <body>
                <h1>🎓 StepBeyond API v2.0</h1>
                <p>Backend is running! Please upload your index.html file or access the API endpoints directly.</p>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/status">System Status</a></li>
                </ul>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(f"<h1>StepBeyond API</h1><p>Status: Running</p><p>Error loading frontend: {str(e)}</p>")

# API Health check endpoint
@app.get("/api/health")
async def api_health():
    """API health check endpoint"""
    return {"status": "ok", "message": "StepBeyond API v2.0 is running", "version": "2.0.0"}

# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status"""
    global agent_instance, rag_instance
    
    # Check environment variables
    groq_api_configured = bool(os.getenv('GROQ_API_KEY'))
    qdrant_configured = bool(os.getenv('QDRANT_URL')) and bool(os.getenv('QDRANT_API_KEY'))
    
    # Check agent status
    agent_status = "ready" if agent_instance else "not_initialized"
    
    # Check RAG status and document count
    rag_status = "not_initialized"
    rag_documents = 0
    
    if rag_instance:
        try:
            collection_info = rag_instance.get_collection_info()
            if collection_info:
                rag_status = "ready"
                rag_documents = collection_info.get('points_count', 0)
            else:
                rag_status = "connected_empty"
        except Exception as e:
            rag_status = "error"
            logger.error(f"Error getting RAG status: {str(e)}")
    
    return SystemStatus(
        rag_status=rag_status,
        rag_documents=rag_documents,
        agent_status=agent_status,
        groq_api_configured=groq_api_configured,
        qdrant_configured=qdrant_configured
    )

# Initialize system endpoint (can be called manually)
@app.post("/initialize")
async def initialize_system():
    """Manually initialize or reinitialize the system"""
    global agent_instance, rag_instance
    
    try:
        # Check environment variables
        env_ok, missing_vars = check_required_env_vars()
        if not env_ok:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing environment variables: {missing_vars}"
            )
        
        # Initialize Qdrant RAG if not already done
        if rag_instance is None:
            logger.info("Initializing Qdrant RAG pipeline...")
            rag_instance = QdrantRAGPipeline()
        
        # Initialize Agent if not already done
        if agent_instance is None:
            logger.info("Initializing StepBeyond Agent...")
            agent_instance = setup_agent()
        
        # Get current status
        collection_info = rag_instance.get_collection_info()
        
        return {
            "success": True,
            "message": "System initialized successfully",
            "rag_status": "ready" if collection_info else "ready_empty",
            "agent_status": "ready",
            "documents_count": collection_info.get('points_count', 0) if collection_info else 0
        }
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

# File upload endpoint
@app.post("/upload-documents", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents into the knowledge base"""
    global rag_instance
    
    if not rag_instance:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please call /initialize endpoint first."
        )
    
    try:
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = []
            
            # Save uploaded files to temporary directory
            for file in files:
                if not file.filename:
                    continue
                    
                # Check file type
                allowed_extensions = ['.pdf', '.txt', '.docx']
                file_extension = os.path.splitext(file.filename)[1].lower()
                
                if file_extension not in allowed_extensions:
                    continue
                
                file_path = os.path.join(temp_dir, file.filename)
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                saved_files.append(file.filename)
                logger.info(f"Saved uploaded file: {file.filename}")
            
            if not saved_files:
                return DocumentUploadResponse(
                    success=False,
                    message="No valid files uploaded. Please upload PDF, TXT, or DOCX files.",
                    documents_processed=0,
                    total_chunks=0
                )
            
            # Process documents into Qdrant
            chunks_created = rag_instance.ingest_documents_from_directory(temp_dir)
            
            return DocumentUploadResponse(
                success=True,
                message=f"Successfully processed {len(saved_files)} files: {', '.join(saved_files)}",
                documents_processed=len(saved_files),
                total_chunks=chunks_created
            )
    
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

# Clear knowledge base endpoint
@app.post("/clear-knowledge-base")
async def clear_knowledge_base():
    """Clear all documents from the knowledge base"""
    global rag_instance
    
    if not rag_instance:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please call /initialize endpoint first."
        )
    
    try:
        rag_instance.clear_collection()
        
        return {
            "success": True,
            "message": "Knowledge base cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {str(e)}")

# Main query processing endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return AI-generated response"""
    global agent_instance
    
    if not agent_instance:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please call /initialize endpoint first."
        )
    
    try:
        logger.info(f"Processing query: {request.query}")
        start_time = time.time()
        
        try:
            # Try comprehensive processing first
            response = agent_instance.process_query(request.query)
        except Exception as e:
            logger.warning(f"Comprehensive processing failed, trying quick answer: {str(e)}")
            # Fallback to quick answer
            response = agent_instance.quick_answer(request.query)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=response,
            processing_time=processing_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            response="I apologize, but I encountered an error while processing your query. Please try again later.",
            processing_time=0.0,
            success=False,
            error=str(e)
        )

# Quick answer endpoint (faster alternative)
@app.post("/quick-answer", response_model=QueryResponse)
async def get_quick_answer(request: QueryRequest):
    """Get a quick answer using RAG context and LLM directly"""
    global agent_instance
    
    if not agent_instance:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please call /initialize endpoint first."
        )
    
    try:
        logger.info(f"Generating quick answer for: {request.query}")
        start_time = time.time()
        
        response = agent_instance.quick_answer(request.query)
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=response,
            processing_time=processing_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error generating quick answer: {str(e)}")
        return QueryResponse(
            response="I apologize, but I encountered an error while processing your query.",
            processing_time=0.0,
            success=False,
            error=str(e)
        )

# Get relevant context endpoint
@app.post("/context")
async def get_context(request: QueryRequest):
    """Get relevant context from the knowledge base for a query"""
    global rag_instance
    
    if not rag_instance:
        raise HTTPException(
            status_code=503, 
            detail="Knowledge base not loaded. Please initialize system first."
        )
    
    try:
        context = rag_instance.get_context_for_query(request.query, request.max_chunks)
        results = rag_instance.similarity_search(request.query, limit=request.max_chunks)
        
        return {
            "query": request.query,
            "context": context,
            "sources": [
                {
                    "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "filename": result["filename"],
                    "file_type": result["file_type"],
                    "score": result["score"],
                    "chunk_index": result["chunk_index"]
                }
                for result in results
            ],
            "total_chunks": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")

# Example queries endpoint
@app.get("/example-queries")
async def get_example_queries():
    """Get example queries that users can try"""
    return {
        "examples": [
            "What documents do I need for a UK student visa?",
            "How to get scholarships for studying in the US?",
            "Best universities in Canada for engineering?",
            "IELTS requirements for Australian universities",
            "How to write a strong SOP?",
            "Work permit rules for international students",
            "What is the cost of studying in Germany?",
            "How to apply for Australian student visa?",
            "PhD application process in the US",
            "Part-time job opportunities for international students"
        ]
    }

# Health check with detailed info
@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    global agent_instance, rag_instance
    
    # Get collection info safely
    rag_info = {"status": "not_ready", "documents": 0}
    if rag_instance:
        try:
            collection_info = rag_instance.get_collection_info()
            if collection_info:
                rag_info = {
                    "status": "ready", 
                    "documents": collection_info.get('points_count', 0),
                    "collection_status": collection_info.get('status', 'unknown')
                }
            else:
                rag_info = {"status": "connected_empty", "documents": 0}
        except Exception as e:
            rag_info = {"status": "error", "documents": 0, "error": str(e)}
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "components": {
            "qdrant_rag": rag_info,
            "stepbeyond_agent": {
                "status": "ready" if agent_instance else "not_ready"
            },
            "environment": {
                "groq_api": bool(os.getenv('GROQ_API_KEY')),
                "qdrant_url": bool(os.getenv('QDRANT_URL')),
                "qdrant_api_key": bool(os.getenv('QDRANT_API_KEY'))
            }
        }
    }

# Collection information endpoint
@app.get("/collection-info")
async def get_collection_info():
    """Get detailed information about the Qdrant collection"""
    global rag_instance
    
    if not rag_instance:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please call /initialize endpoint first."
        )
    
    try:
        collection_info = rag_instance.get_collection_info()
        return {
            "success": True,
            "collection_info": collection_info
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

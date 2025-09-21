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
from rag_pipeline import LightweightRAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StepBeyond API v2.0",
    description="AI-powered career counseling assistant API with lightweight Qdrant integration",
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
rag_instance: Optional[LightweightRAGPipeline] = None

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

# LAZY INITIALIZATION - Only initialize when first API call is made
# This prevents memory overflow during startup
@app.on_event("startup")
async def startup_event():
    """Startup event - keep minimal to avoid memory issues"""
    logger.info("Starting StepBeyond API v2.0 (Lightweight mode)...")
    logger.info("Components will be initialized on first use to save memory")
    
    # Just check environment variables, don't initialize heavy components yet
    env_ok, missing_vars = check_required_env_vars()
    if not env_ok:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("System will not be fully functional until all variables are set")

def lazy_init_rag():
    """Initialize RAG pipeline only when needed"""
    global rag_instance
    
    if rag_instance is None:
        logger.info("Lazy initializing Qdrant RAG pipeline...")
        rag_instance = LightweightRAGPipeline()
        logger.info("RAG pipeline initialized")
    
    return rag_instance

def lazy_init_agent():
    """Initialize agent only when needed"""
    global agent_instance
    
    if agent_instance is None:
        logger.info("Lazy initializing StepBeyond Agent...")
        agent_instance = setup_agent()
        logger.info("Agent initialized")
    
    return agent_instance

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
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
                    .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 40px; border-radius: 15px; }
                    h1 { font-size: 2.5rem; text-align: center; margin-bottom: 20px; }
                    .status { background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; margin: 20px 0; }
                    .btn { background: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
                    .btn:hover { background: #218838; }
                    a { color: #87CEEB; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎓 StepBeyond API v2.0</h1>
                    <div class="status">
                        <h3>✅ Backend is running!</h3>
                        <p>Lightweight mode - Components load on demand to save memory</p>
                        <p><strong>Next Steps:</strong></p>
                        <ol>
                            <li>Upload your index.html file to this directory, or</li>
                            <li>Use the API endpoints directly</li>
                        </ol>
                    </div>
                    
                    <div class="status">
                        <h3>🔗 Available Endpoints:</h3>
                        <p><a href="/docs">📖 API Documentation (Swagger)</a></p>
                        <p><a href="/health">🏥 Health Check</a></p>
                        <p><a href="/status">📊 System Status</a></p>
                        <p><button class="btn" onclick="location.href='/initialize'">🚀 Initialize System</button></p>
                    </div>
                    
                    <div class="status">
                        <h3>💾 Memory Optimization:</h3>
                        <p>This version uses TF-IDF instead of sentence-transformers to fit in Render's 512MB limit.</p>
                        <p>Performance is optimized for the free tier while maintaining full functionality.</p>
                    </div>
                </div>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(f"<h1>StepBeyond API</h1><p>Status: Running</p><p>Error loading frontend: {str(e)}</p>")

# API Health check endpoint
@app.get("/api/health")
async def api_health():
    """API health check endpoint"""
    return {"status": "ok", "message": "StepBeyond API v2.0 is running", "version": "2.0.0", "mode": "lightweight"}

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
    try:
        # Check environment variables
        env_ok, missing_vars = check_required_env_vars()
        if not env_ok:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing environment variables: {missing_vars}"
            )
        
        # Initialize components
        logger.info("Initializing system components...")
        
        rag = lazy_init_rag()
        agent = lazy_init_agent()
        
        # Get current status
        collection_info = rag.get_collection_info()
        
        return {
            "success": True,
            "message": "System initialized successfully (lightweight mode)",
            "rag_status": "ready" if collection_info else "ready_empty",
            "agent_status": "ready",
            "documents_count": collection_info.get('points_count', 0) if collection_info else 0,
            "mode": "lightweight"
        }
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

# File upload endpoint
@app.post("/upload-documents", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents into the knowledge base"""
    try:
        # Initialize RAG if needed
        rag = lazy_init_rag()
        
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
            chunks_created = rag.ingest_documents_from_directory(temp_dir)
            
            return DocumentUploadResponse(
                success=True,
                message=f"Successfully processed {len(saved_files)} files: {', '.join(saved_files)}",
                documents_processed=len(saved_files),
                total_chunks=chunks_created
            )
    
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

# Main query processing endpoint
@app.post("/query", response_model=QueryResponse)
async def

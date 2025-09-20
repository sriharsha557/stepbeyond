from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Import your existing modules
from crew_agents import setup_crew, StepBeyondCrew
from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StepBeyond API",
    description="AI-powered career counseling assistant API for studying abroad",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store system instances
crew_instance: Optional[StepBeyondCrew] = None
rag_instance: Optional[RAGPipeline] = None

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
    crew_status: str
    groq_api_configured: bool

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: float

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the system when the API starts"""
    global crew_instance, rag_instance
    
    try:
        logger.info("Initializing system...")
        
        # Check for GROQ API key
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
            return
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_instance = RAGPipeline()
        rag_instance.load_index()
        
        # Initialize CrewAI
        logger.info("Initializing CrewAI...")
        crew_instance = setup_crew(groq_api_key)
        
        logger.info("System initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "StepBeyond API is running"}

# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status"""
    global crew_instance, rag_instance
    
    # Check RAG status
    rag_status = "ready" if rag_instance and rag_instance.index else "not_loaded"
    rag_documents = rag_instance.index.ntotal if rag_instance and rag_instance.index else 0
    
    # Check Crew status
    crew_status = "ready" if crew_instance else "not_initialized"
    
    # Check GROQ API key
    groq_api_configured = bool(os.getenv('GROQ_API_KEY'))
    
    return SystemStatus(
        rag_status=rag_status,
        rag_documents=rag_documents,
        crew_status=crew_status,
        groq_api_configured=groq_api_configured
    )

# Initialize system endpoint (can be called manually)
@app.post("/initialize")
async def initialize_system():
    """Manually initialize or reinitialize the system"""
    global crew_instance, rag_instance
    
    try:
        # Check for GROQ API key
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise HTTPException(
                status_code=400, 
                detail="GROQ_API_KEY not found in environment variables"
            )
        
        # Initialize RAG if not already done
        if rag_instance is None:
            rag_instance = RAGPipeline()
        
        if not rag_instance.load_index():
            return {
                "success": False,
                "message": "No existing knowledge base found. Please rebuild knowledge base first.",
                "rag_status": "no_index",
                "crew_status": "not_initialized"
            }
        
        # Initialize CrewAI
        crew_instance = setup_crew(groq_api_key)
        
        return {
            "success": True,
            "message": "System initialized successfully",
            "rag_status": "ready",
            "crew_status": "ready"
        }
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

# Rebuild knowledge base endpoint
@app.post("/rebuild-knowledge-base")
async def rebuild_knowledge_base():
    """Rebuild the knowledge base from data folder"""
    global rag_instance, crew_instance
    
    try:
        logger.info("Starting knowledge base rebuild...")
        
        if rag_instance is None:
            rag_instance = RAGPipeline()
        
        # Rebuild the knowledge base
        rag_instance.ingest_data()
        
        # Reinitialize crew if it exists
        if crew_instance:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                crew_instance = setup_crew(groq_api_key)
        
        return {
            "success": True,
            "message": "Knowledge base rebuilt successfully",
            "documents_count": rag_instance.index.ntotal if rag_instance.index else 0
        }
        
    except Exception as e:
        logger.error(f"Error rebuilding knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Knowledge base rebuild failed: {str(e)}")

# Main query processing endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return AI-generated response"""
    global crew_instance
    
    if not crew_instance:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please call /initialize endpoint first."
        )
    
    try:
        logger.info(f"Processing query: {request.query}")
        start_time = time.time()
        
        try:
            # Try CrewAI first
            response = crew_instance.process_query(request.query)
        except Exception as e:
            logger.warning(f"CrewAI failed, trying quick answer: {str(e)}")
            # Fallback to quick answer
            response = crew_instance.quick_answer(request.query)
        
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
    global crew_instance
    
    if not crew_instance:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please call /initialize endpoint first."
        )
    
    try:
        logger.info(f"Generating quick answer for: {request.query}")
        start_time = time.time()
        
        response = crew_instance.quick_answer(request.query)
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
    
    if not rag_instance or not rag_instance.index:
        raise HTTPException(
            status_code=503, 
            detail="Knowledge base not loaded. Please rebuild knowledge base first."
        )
    
    try:
        context = rag_instance.get_context_for_query(request.query, request.max_chunks)
        results = rag_instance.similarity_search(request.query, k=request.max_chunks)
        
        return {
            "query": request.query,
            "context": context,
            "sources": [
                {
                    "content": result["content"][:200] + "...",  # Truncated preview
                    "source": result["metadata"].get("source", "Unknown"),
                    "distance": result["distance"],
                    "rank": result["rank"]
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
            "Work permit rules for international students"
        ]
    }

# Health check with detailed info
@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    global crew_instance, rag_instance
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "rag_pipeline": {
                "status": "ready" if rag_instance and rag_instance.index else "not_ready",
                "documents": rag_instance.index.ntotal if rag_instance and rag_instance.index else 0
            },
            "crew_ai": {
                "status": "ready" if crew_instance else "not_ready"
            },
            "groq_api": {
                "configured": bool(os.getenv('GROQ_API_KEY'))
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

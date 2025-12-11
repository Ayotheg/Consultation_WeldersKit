from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Import the RAG system from bot.py
from bot import AlwaysHybridRAG

# Load environment variables
if os.path.exists('.env'):
    load_dotenv()
    print("🔍 Loaded .env file")

# Initialize FastAPI
app = FastAPI(
    title="WeldersKit AI API",
    description="Lightweight RAG system for welding consultations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://consultation-welderskit.streamlit.app",
        "http://localhost:8501",
        "*"  # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system once at startup
rag = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when FastAPI starts"""
    global rag
    try:
        print("\n" + "="*70)
        print("🚀 Starting WeldersKit AI Backend")
        print("="*70)
        
        # Initialize RAG with keyword search (no embeddings)
        rag = AlwaysHybridRAG(csv_file='welders_data-main.csv')
        rag.initialize()
        
        print("\n✅ Backend ready to serve requests!")
        print("="*70 + "\n")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("⚠️ API will run in fallback mode (AI only, no database)")
        rag = None

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the best welding machine for beginners in Nigeria?"
            }
        }

class AnswerResponse(BaseModel):
    answer: str
    sources: str
    model_used: str
    database_matches: int

# API endpoints
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "WeldersKit AI API",
        "status": "online",
        "version": "1.0.0",
        "model": "google/gemini-2.0-flash-exp:free (via OpenRouter)",
        "search_method": "Lightweight keyword search (no embeddings)",
        "features": [
            "Keyword-based database search",
            "AI-powered answers with Gemini 2.0 Flash",
            "Nigerian market context",
            "Hybrid RAG (combines database + AI knowledge)",
            "Low memory footprint (~100MB)"
        ],
        "endpoints": {
            "/": "GET - API information",
            "/api/ask": "POST - Ask a welding question",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if not rag:
        return {
            "status": "degraded",
            "rag_initialized": False,
            "message": "RAG system not initialized"
        }
    
    return {
        "status": "healthy",
        "rag_initialized": True,
        "model": rag.model_name if hasattr(rag, 'model_name') else "unknown",
        "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "database_loaded": rag.df is not None,
        "database_entries": len(rag.df) if rag.df is not None else 0,
        "search_method": "keyword-based"
    }

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a welding-related question
    
    The system will:
    1. Search the database using keyword matching
    2. Combine database results with AI knowledge
    3. Return a comprehensive answer with Nigerian market context
    """
    try:
        if not rag:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Service temporarily unavailable."
            )
        
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Use the RAG system to get answer
        result = rag.query(request.question)
        
        return AnswerResponse(
            answer=result['answer'],
            sources=result['sources'],
            model_used=result.get('model_used', 'google/gemini-2.0-flash-exp:free'),
            database_matches=result.get('database_matches', 0)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in ask_question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )

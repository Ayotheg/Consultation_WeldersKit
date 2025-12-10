from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Import the RAG system from bot.py
from bot import AlwaysHybridRAG

if os.path.exists('.env'):
    load_dotenv()
    print("🔍 Loaded .env file")

# Initialize FastAPI
app = FastAPI(title="WeldersKit AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://consultation-welderskit.streamlit.app",
        "http://localhost:8501",
        "*"
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
        
        rag = AlwaysHybridRAG(csv_file='welders_data-main.csv')
        rag.initialize()
        
        print("\n✅ Backend ready to serve requests!")
        print("="*70 + "\n")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("⚠️ API will run in fallback mode (AI only, no database)")
        rag = None

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: str = "AI"
    model_used: str = ""
    database_matches: int = 0

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a welding-related question"""
    try:
        if not rag:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Service temporarily unavailable."
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "WeldersKit AI API",
        "status": "online",
        "model": "google/gemini-2.0-flash-exp:free (via OpenRouter)",
        "features": [
            "Database search with embeddings",
            "AI-powered answers with Gemini 2.0 Flash",
            "Nigerian market context",
            "Hybrid RAG (always combines database + AI)"
        ],
        "endpoints": {
            "/api/ask": "POST - Ask a welding question",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if rag else "degraded",
        "rag_initialized": bool(rag),
        "model": "google/gemini-2.0-flash-exp:free",
        "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "database_loaded": rag.df is not None if rag else False,
        "embeddings_ready": rag.index is not None if rag else False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

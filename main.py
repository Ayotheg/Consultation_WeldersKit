from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

if os.path.exists('.env'):
    load_dotenv()
    print("🔍 Loaded .env file")

# Debug: Check if .env loaded
print("🔍 Debug Info:")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir()}")
print(f"GOOGLE_API_KEY value: {os.getenv('GOOGLE_API_KEY')}")

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

# Configure OpenRouter API
api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("❌ ERROR: API key not set!")
else:
    print(f"✅ API Key found: {api_key[:15]}...")
    
    # Validate it's an OpenRouter key
    if not api_key.startswith('sk-or-'):
        print(f"⚠️ WARNING: API key doesn't look like an OpenRouter key (should start with sk-or-)")

# OpenRouter configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "google/gemini-2.0-flash-exp:free"

# Backup models if primary fails
BACKUP_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemini-flash-1.5:free",
    "qwen/qwen-2-7b-instruct:free",
]

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: str = "AI"
    model_used: str = ""

def call_openrouter(question: str, model: str) -> dict:
    """Call OpenRouter API with a specific model"""
    
    system_prompt = """You are an expert welding and construction consultant for WeldersKit in Nigeria.

Provide comprehensive, practical answers about:
- Welding techniques and safety
- Materials and pricing (in Nigerian Naira ₦)
- Nigerian market context (Idumota, Ladipo, Trade Fair markets)
- Climate considerations (humid, tropical)
- Common projects (gates, railings, window protectors, carports)

Be helpful, accurate, and consider local Nigerian context."""

    try:
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://consultation-welderskit.onrender.com",
                "X-Title": "WeldersKit AI"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            timeout=30
        )
        
        if response.ok:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return {
                    "success": True,
                    "answer": result['choices'][0]['message']['content'],
                    "model": model
                }
        
        return {
            "success": False,
            "error": response.text
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        print(f"\n{'='*70}")
        print(f"❓ Question: {request.question}")
        print(f"{'='*70}")
        
        # Try primary model first
        print(f"🔄 Trying primary model: {PRIMARY_MODEL}")
        result = call_openrouter(request.question, PRIMARY_MODEL)
        
        if result["success"]:
            print(f"✅ Success with {result['model']}")
            return AnswerResponse(
                answer=result["answer"],
                sources="AI Knowledge",
                model_used=result["model"]
            )
        
        print(f"⚠️ Primary model failed: {result['error'][:100]}...")
        
        # Try backup models
        for backup_model in BACKUP_MODELS:
            print(f"🔄 Trying backup model: {backup_model}")
            result = call_openrouter(request.question, backup_model)
            
            if result["success"]:
                print(f"✅ Success with {result['model']}")
                return AnswerResponse(
                    answer=result["answer"],
                    sources="AI Knowledge",
                    model_used=result["model"]
                )
            
            print(f"⚠️ {backup_model} failed: {result['error'][:100]}...")
        
        # All models failed
        raise HTTPException(
            status_code=500,
            detail=f"All AI models failed. Last error: {result.get('error', 'Unknown error')}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "WeldersKit AI API",
        "status": "online",
        "endpoints": {
            "/api/ask": "POST - Ask a welding question"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_configured": bool(api_key),
        "primary_model": PRIMARY_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

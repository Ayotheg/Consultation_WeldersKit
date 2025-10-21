from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


# Debug: Check if .env loaded
print("🔍 Debug Info:")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir()}")
print(f"GOOGLE_API_KEY value: {os.getenv('GOOGLE_API_KEY')}")
print(f"All env vars: {dict(os.environ)}")

# Initialize FastAPI
app = FastAPI(title="WeldersKit AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
               "https://consultation-welderskit.streamlit.app",
          "http://localhost:8501",
        "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
api_key =os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("❌ ERROR: GOOGLE_API_KEY not set!")
else:
    print(f"✅ API Key found: {api_key[:10]}...")
    genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.0-flash-lite')

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: str = "AI"

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
            
        response = model.generate_content(f"You are a welding expert for WeldersKit in Nigeria.\n\nQuestion: {request.question}\n\nProvide a helpful answer.")
        answer = response.text
        
        return AnswerResponse(answer=answer, sources="AI Knowledge")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

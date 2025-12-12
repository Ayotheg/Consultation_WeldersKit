import pandas as pd
import numpy as np
import requests
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# DEBUG: Check what environment variables are available
print("=" * 70, file=sys.stderr)
print("DEBUG: Checking environment variables...", file=sys.stderr)
google_key = os.getenv('GOOGLE_API_KEY')
openrouter_key = os.getenv('OPENROUTER_API_KEY')
print(f"GOOGLE_API_KEY present: {bool(google_key)}", file=sys.stderr)
print(f"OPENROUTER_API_KEY present: {bool(openrouter_key)}", file=sys.stderr)
if google_key:
    print(f"GOOGLE_API_KEY value starts with: {google_key[:15]}...", file=sys.stderr)
if openrouter_key:
    print(f"OPENROUTER_API_KEY value starts with: {openrouter_key[:15]}...", file=sys.stderr)
print("=" * 70, file=sys.stderr)

class AlwaysHybridRAG:
    """
    LIGHTWEIGHT RAG: Simple keyword search + AI (no embeddings!)
    Uses only ~100MB memory - perfect for 512MB free tier
    """
    
    def __init__(self, csv_file='welders_data-main.csv'):
        self.csv_file = csv_file
        self.df = None
        
        # Initialize OpenRouter - Check BOTH possible env var names
        print("\n🔑 Checking for API key...", file=sys.stderr)
        
        # Try OPENROUTER_API_KEY first, then fall back to GOOGLE_API_KEY
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("⚠️ OPENROUTER_API_KEY not found, trying GOOGLE_API_KEY...", file=sys.stderr)
            api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            raise ValueError(
                "❌ API key not found! Please set one of these environment variables:\n"
                "   - OPENROUTER_API_KEY=sk-or-v1-...\n"
                "   - GOOGLE_API_KEY=sk-or-v1-...\n"
                "Get your key from: https://openrouter.ai/keys"
            )
        
        print(f"✅ Found API key: {api_key[:15]}...", file=sys.stderr)
        
        # Validate it's an OpenRouter key
        if not api_key.startswith('sk-or-'):
            raise ValueError(
                f"❌ Invalid OpenRouter key format!\n"
                f"   Expected: sk-or-v1-...\n"
                f"   Got: {api_key[:15]}...\n"
                f"   Get your OpenRouter key from: https://openrouter.ai/keys"
            )
        
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Updated model list - Most Reliable Free Models (Dec 2024)
        # Note: Free models require privacy settings enabled on OpenRouter
        self.models = [
            "meta-llama/llama-3.2-3b-instruct:free",               # Most reliable free model
            "meta-llama/llama-3.1-8b-instruct:free",               # Stable Llama alternative
            "qwen/qwen-2-7b-instruct:free",                        # Good quality Chinese model
            "mistralai/mistral-7b-instruct:free",                  # Stable Mistral
            "nousresearch/hermes-3-llama-3.1-405b:free",          # Powerful backup
            "google/gemini-2.0-flash-exp:free",                    # Gemini (if available)
        ]
        self.model_name = self.models[0]  # Track which model is currently being used
        
        print(f"✅ OpenRouter API configured successfully!", file=sys.stderr)
        print(f"   Primary Model: {self.model_name}", file=sys.stderr)
        print(f"   Backup Models: {len(self.models)-1}", file=sys.stderr)
        print(f"   Endpoint: {self.api_url}", file=sys.stderr)
    
    def load_data(self):
        """Load dataset"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.df)} entries from dataset")
            
            # Verify columns
            required = ['question', 'answer']
            if not all(col in self.df.columns for col in required):
                print(f"⚠️ Missing required columns. Found: {self.df.columns.tolist()}")
                return False
            
            return True
        except Exception as e:
            print(f"⚠️ Could not load dataset: {e}")
            return False
    
    def search_dataset_simple(self, query, k=5):
        """Simple keyword-based search (no embeddings needed!)"""
        if self.df is None or len(self.df) == 0:
            return []
        
        try:
            # Convert query to lowercase and split into keywords
            keywords = query.lower().split()
            
            # Score each row based on keyword matches
            scores = []
            for idx, row in self.df.iterrows():
                text = f"{row['question']} {row['answer']}".lower()
                score = sum(1 for keyword in keywords if keyword in text)
                scores.append((idx, score))
            
            # Sort by score and take top k
            scores.sort(key=lambda x: x[1], reverse=True)
            top_matches = scores[:k]
            
            # Return results with score > 0
            results = []
            for idx, score in top_matches:
                if score > 0:
                    results.append({
                        'question': self.df.iloc[idx]['question'],
                        'answer': self.df.iloc[idx]['answer'],
                        'score': float(score) / len(keywords),  # Normalize score
                        'price': self.df.iloc[idx].get('price_range', 'N/A'),
                        'location': self.df.iloc[idx].get('location', 'Nigeria')
                    })
            
            return results
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []
    
    def generate_combined_answer(self, query, dataset_results):
        """
        ALWAYS combine dataset + AI knowledge using OpenRouter
        """
        
        # Build dataset context
        dataset_context = ""
        if dataset_results:
            dataset_context = "RELEVANT DATA FROM YOUR DATABASE:\n\n"
            for i, result in enumerate(dataset_results[:3], 1):  # Use top 3
                dataset_context += f"{i}. Q: {result['question']}\n"
                dataset_context += f"   A: {result['answer']}\n"
                if result['price'] != 'N/A':
                    dataset_context += f"   Price: {result['price']}\n"
                dataset_context += f"   Relevance: {result['score']:.1%}\n\n"
        
        # Enhanced system prompt
        system_prompt = """You are an expert welding and construction consultant for WeldersKit in Nigeria.

Your role: Provide comprehensive answers by COMBINING:
1. Specific data from the WeldersKit database (when available)
2. Your general welding and construction knowledge
3. Nigerian market context and practices

IMPORTANT INSTRUCTIONS:
- If database has relevant info, START with that data
- Then ADD your general knowledge to expand the answer
- Fill in any gaps the database doesn't cover
- Provide practical, actionable advice
- Mention prices in Nigerian Naira (₦)
- Consider Nigerian climate (humid, tropical)
- Reference local markets when relevant (Idumota, Ladipo, Trade Fair)
- If database info seems outdated or incomplete, mention it and provide updated guidance
- Never say "I don't have information" if you can provide general guidance

Nigerian Context:
- Main markets: Idumota (Lagos), Ladipo (Lagos), Trade Fair Complex, Dei Dei (Abuja)
- Climate: Hot, humid (rust prevention important)
- Common projects: Gates, railings, window protectors, carports, roofing
- Popular welding: Arc/SMAW (affordable, works outdoors)
"""
        
        # Build user message
        user_message = f"""{dataset_context if dataset_context else "No specific database matches found for this query. Provide comprehensive answer using your general knowledge."}

USER QUESTION: {query}

COMPREHENSIVE ANSWER (combine database info + your knowledge):"""
        
        # Try models in order of priority
        last_error = None
        
        for attempt, model in enumerate(self.models, 1):
            try:
                print(f"🔄 Attempt {attempt}: Trying {model}...", file=sys.stderr)
                
                response = requests.post(
                    url=self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://consultation-welderskit.onrender.com",
                        "X-Title": "WeldersKit RAG"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1024
                    },
                    timeout=30
                )
                
                print(f"📥 API Response Status: {response.status_code}", file=sys.stderr)
                
                if response.ok:
                    result = response.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        answer = result['choices'][0]['message']['content']
                        print(f"✅ Successfully generated answer with {model} ({len(answer)} chars)", file=sys.stderr)
                        
                        # IMPORTANT: Update the active model name so it shows correctly
                        self.model_name = model
                        
                        return answer
                
                # If this model failed, log and try next one
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                last_error = str(error_data)
                print(f"⚠️ Model {model} failed: {last_error[:200]}...", file=sys.stderr)
                
            except requests.exceptions.Timeout:
                print(f"⚠️ Model {model} timed out", file=sys.stderr)
                last_error = "Request timed out"
                continue
            except Exception as e:
                print(f"⚠️ Model {model} error: {e}", file=sys.stderr)
                last_error = str(e)
                continue
        
        # All models failed
        return f"❌ All AI models failed. Last error: {last_error}"
    
    def query(self, question):
        """
        Main query - ALWAYS uses both dataset AND AI
        """
        print(f"\n{'='*70}")
        print(f"❓ Question: {question}")
        print('='*70 + "\n")
        
        # Step 1: Search dataset using simple keyword matching
        dataset_results = self.search_dataset_simple(question, k=5)
        
        if dataset_results:
            print(f"📚 Found {len(dataset_results)} relevant entries in database")
            print(f"   Best match: {dataset_results[0]['question'][:60]}...")
            print(f"   Relevance: {dataset_results[0]['score']:.1%}")
            print("\n🤖 Combining database data with AI knowledge...\n")
        else:
            print("ℹ️ No direct matches in database")
            print("🤖 Generating answer using AI general knowledge...\n")
        
        # Step 2: ALWAYS generate with AI (using dataset as context)
        answer = self.generate_combined_answer(question, dataset_results)
        
        print(f"💬 ANSWER:\n")
        print(answer)
        print(f"\n{'='*70}\n")
        
        return {
            'question': question,
            'answer': answer,
            'database_matches': len(dataset_results),
            'model_used': self.model_name
        }
    
    def initialize(self):
        """Initialize system"""
        print("\n🚀 Initializing LIGHTWEIGHT RAG (Keyword Search + AI)")
        print("="*70 + "\n")
        
        has_data = self.load_data()
        
        if has_data:
            print("\n✅ Mode: Dataset + AI Knowledge (ALWAYS COMBINED)")
            print("   Using lightweight keyword search (no embeddings)")
            print("   Memory usage: ~100MB (fits in 512MB free tier)")
        else:
            print("\n⚠️ Mode: AI Knowledge Only (no database)")
        
        print("="*70)
        return True

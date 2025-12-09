import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests  # Changed from google.generativeai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AlwaysHybridRAG:
    """
    RAG system that ALWAYS combines:
    1. Dataset knowledge (your CSV)
    2. AI general knowledge (Gemini via OpenRouter)
    
    Never falls back - always uses both together!
    """
    
    def __init__(self, csv_file='welders_data-main.csv'):
        self.csv_file = csv_file
        self.df = None
        self.embeddings = None
        self.index = None
        self.embedding_model = None
        
        # Initialize OpenRouter
        # Check both OPENROUTER_API_KEY and GOOGLE_API_KEY (for backwards compatibility)
        api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("API key not set! Set either OPENROUTER_API_KEY or GOOGLE_API_KEY environment variable")
        
        # Validate it's an OpenRouter key
        if not api_key.startswith('sk-or-'):
            raise ValueError("Invalid OpenRouter key! Key should start with 'sk-or-'. Get your key from https://openrouter.ai/keys")
        
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model_name = "google/gemini-2.0-flash-exp:free"  # Free Gemini model via OpenRouter
        print("✅ OpenRouter API configured")
    
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
    
    def setup_embeddings(self):
        """Setup embeddings for dataset search"""
        if self.df is None or len(self.df) == 0:
            print("⚠️ No dataset available for embeddings")
            return False
        
        try:
            print("🔮 Creating embeddings...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            docs = [f"{row['question']} {row['answer']}" for _, row in self.df.iterrows()]
            self.embeddings = self.embedding_model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
            
            # FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            print(f"✅ Embeddings ready: {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"⚠️ Embedding failed: {e}")
            return False
    
    def search_dataset(self, query, k=5, threshold=0.2):
        """Search dataset (lower threshold to get more context)"""
        if self.index is None or self.embedding_model is None:
            return []
        
        try:
            query_vec = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_vec)
            
            scores, indices = self.index.search(query_vec, k)
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if score >= threshold and idx < len(self.df):
                    results.append({
                        'question': self.df.iloc[idx]['question'],
                        'answer': self.df.iloc[idx]['answer'],
                        'score': float(score),
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
        
        try:
            # OpenRouter API call
            response = requests.post(
                url=self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "WeldersKit RAG"
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            return f"❌ Error generating answer: {e}"
    
    def query(self, question):
        """
        Main query - ALWAYS uses both dataset AND AI
        """
        print(f"\n{'='*70}")
        print(f"❓ Question: {question}")
        print('='*70 + "\n")
        
        # Step 1: Search dataset
        dataset_results = self.search_dataset(question, k=5)
        
        if dataset_results:
            print(f"📚 Found {len(dataset_results)} relevant entries in database")
            print(f"   Best match: {dataset_results[0]['question'][:60]}...")
            print(f"   Similarity: {dataset_results[0]['score']:.1%}")
            print("\n🤖 Combining database data with AI knowledge...\n")
        else:
            print("ℹ️ No direct matches in database")
            print("🤖 Generating answer using AI general knowledge...\n")
        
        # Step 2: ALWAYS generate with AI (using dataset as context)
        answer = self.generate_combined_answer(question, dataset_results)
        
        print(f"💬 ANSWER:\n")
        print(answer)
        
        # Show what was used
        print(f"\n{'─'*70}")
        print(f"📊 Sources Used:")
        if dataset_results:
            print(f"   ✅ Database: {len(dataset_results)} relevant entries")
            print(f"   ✅ AI Knowledge: Enhanced with general expertise")
        else:
            print(f"   ⚠️ Database: No relevant matches")
            print(f"   ✅ AI Knowledge: Full answer from general knowledge")
        print(f"{'='*70}\n")
        
        return {
            'question': question,
            'answer': answer,
            'database_matches': len(dataset_results),
            'sources': 'Database + AI' if dataset_results else 'AI Only'
        }
    
    def initialize(self):
        """Initialize system"""
        print("\n🚀 Initializing HYBRID RAG (Always Combined)")
        print("="*70 + "\n")
        
        has_data = self.load_data()
        
        if has_data:
            has_embeddings = self.setup_embeddings()
            if has_embeddings:
                print("\n✅ Mode: Dataset + AI Knowledge (ALWAYS COMBINED)")
            else:
                print("\n⚠️ Mode: AI Knowledge Only (database failed)")
        else:
            print("\n⚠️ Mode: AI Knowledge Only (no database)")
        
        print("="*70)
        return True


# ============= USAGE =============

if __name__ == "__main__":
    print("\n" + "="*70)
    print("     WELDERSKIT HYBRID RAG - ALWAYS COMBINES DATA + AI")
    print("="*70)
    
    # Set API key (do this before running)
    # os.environ['OPENROUTER_API_KEY'] = 'sk-or-v1-...'
    
    try:
        # Initialize
        rag = AlwaysHybridRAG(csv_file='welders_data-main.csv')
        rag.initialize()
        
        # Test questions (mix of database and general topics)
        test_questions = [
            # Questions likely in database
            "What is the price of galvanized pipes?",
            "Where can I buy square pipes in Nigeria?",
            "How much does stainless steel pipe cost?",
            
            # Questions likely NOT in database (AI will fill in)
            "How do I prevent rust on my gate?",
            "What's the best welding rod for aluminum?",
            "How to calculate material cost for a 10ft gate?",
            "What safety equipment is essential for welding?",
            "How do I weld in rainy weather?",
        ]
        
        print("\n" + "="*70)
        print("TESTING HYBRID SYSTEM")
        print("="*70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[Test {i}/{len(test_questions)}]")
            rag.query(question)
            
            if i < len(test_questions):
                input("\nPress Enter for next question...")
        
        # Interactive mode
        print("\n" + "="*70)
        print("🤖 INTERACTIVE MODE")
        print("="*70)
        print("Ask anything about welding, materials, prices, techniques...")
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                user_q = input("\nYour question: ").strip()
                
                if user_q.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thank you for using WeldersKit AI!")
                    break
                
                if not user_q:
                    continue
                
                rag.query(user_q)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("\nMake sure:")
        print("1. OPENROUTER_API_KEY is set")
        print("2. welders_data-main.csv exists")
        print("3. Required packages installed")

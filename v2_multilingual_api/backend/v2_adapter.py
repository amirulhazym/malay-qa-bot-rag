# -- v2_multilingual_api/backend/v2_adapter.py --

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
import google.generativeai as genai

# --- 1. Initialization Block (from main.py) ---
# This block loads API keys and initializes all the necessary models and services.
# It runs once when the script is first imported.

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("API keys not found. Check your .env file.")

print("Initializing V2 models and services for evaluation...")
retriever_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("auracart-multilingual-kb")

genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel('gemini-2.5-flash-lite') 
print("V2 Adapter Initialized.")

# --- 2. The Adapter Function ---
# This function contains the core logic from our FastAPI endpoint.

def get_v2_rag_response(question: str, history: list = []):
    """
    Takes a question and returns the answer and retrieved contexts in a dictionary
    formatted for RAGAs evaluation.
    """
    
    # --- This entire block is copied from the logic inside our /api/ask endpoint ---
    
    # 1. Retrieve
    query_embedding = retriever_model.encode(question).tolist()
    retrieved_docs = index.query(vector=query_embedding, top_k=20, include_metadata=True)
    original_docs = retrieved_docs['matches']

    # 2. Re-rank
    reranker_input_pairs = [(question, doc['metadata'].get('text', '')) for doc in original_docs]
    rerank_scores = reranker_model.predict(reranker_input_pairs)
    ranked_docs = sorted(zip(rerank_scores, original_docs), reverse=True)
    
    # 3. Prepare Final Context and Sources
    top_docs = [doc for score, doc in ranked_docs[:4]]
    context_str = "\n\n".join([doc['metadata'].get('text', '') for doc in top_docs])
    
    history_str = "\n".join([f"User: {h.get('user', '')}\nBot: {h.get('bot', '')}" for h in history])

    prompt = f"""
    You are "Aura," AuraCart's professional and bilingual customer service AI, fluent in both English and Malay.
    Your primary goal is to provide helpful and accurate answers based ONLY on the context provided from the knowledge base.
    IMPORTANT: You MUST answer in the same language as the user's question. If the user asks in Malay, you must answer in Malay.
    If the context does not contain the answer, clearly state that you don't have enough information in the user's language. Do not invent answers.

    CONVERSATION HISTORY:
    {history_str}

    KNOWLEDGE BASE CONTEXT:
    ---
    {context_str}
    ---

    USER'S QUESTION:
    {question}

    YOUR HELPFUL, BILINGUAL ANSWER:
    """
    
    response = generation_model.generate_content(prompt)
    
    # --- The key change: Return a dictionary for RAGAs, not a FastAPI response ---
    return {
        "answer": response.text,
        "contexts": [doc['metadata'].get('text', '') for doc in top_docs]
    }

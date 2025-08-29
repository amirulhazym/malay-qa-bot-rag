# --- Full, Final, and Verified Code for v2_multilingual_api/backend/main.py ---

import os
import json # <-- FIX 1: Added the missing json import
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
import google.generativeai as genai

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("API keys not found. Check your .env file.")

print("Initializing multilingual models...")
retriever_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
print("Models initialized.")

print("Initializing services...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("auracart-multilingual-kb")
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel('gemini-2.5-flash-lite')
print("Services initialized.")

app = FastAPI()

class Source(BaseModel):
    content: str
    source_file: str = Field(alias="source") # Using an alias is fine, it maps the 'source' key to this field

class QueryRequest(BaseModel):
    question: str
    history: list = []

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

class SuggestionRequest(BaseModel):
    history: list

class SuggestionResponse(BaseModel):
    suggestions: list[str]

@app.post("/api/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    # --- FIX 2: Corrected the retrieval and source handling logic ---
    
    # 1. Retrieve
    query_embedding = retriever_model.encode(request.question).tolist()
    retrieved_docs = index.query(vector=query_embedding, top_k=20, include_metadata=True)
    
    # Keep the full document objects, not just the text
    original_docs = retrieved_docs['matches']

    # 2. Re-rank
    # Prepare pairs of (question, passage_text) for the reranker model
    reranker_input_pairs = [(request.question, doc['metadata'].get('text', '')) for doc in original_docs]
    rerank_scores = reranker_model.predict(reranker_input_pairs)
    
    # Combine the original documents with their new scores and sort
    ranked_docs = sorted(zip(rerank_scores, original_docs), reverse=True)
    
    # 3. Prepare Final Context and Sources
    # Select the top 4 most relevant documents after re-ranking
    top_docs = [doc for score, doc in ranked_docs[:4]]
    
    # Prepare the context string for the LLM prompt
    context_str = "\n\n".join([doc['metadata'].get('text', '') for doc in top_docs])
    
    # Prepare the source list for the final API response
    sources_for_response = [
        {
            "content": doc['metadata'].get('text', ''),
            "source": doc['metadata'].get('source', 'unknown')
        }
        for doc in top_docs
    ]
    # --- END OF FIX 2 ---
    
    history_str = "\n".join([f"User: {h.get('user', '')}\nBot: {h.get('bot', '')}" for h in request.history])

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
    {request.question}

    YOUR HELPFUL, BILINGUAL ANSWER:
    """
    
    response = generation_model.generate_content(prompt)
    
    # Return the generated answer AND the list of sources used
    return QueryResponse(answer=response.text, sources=sources_for_response)

@app.get("/")
def read_root():
    return {"message": "AuraCart Multilingual AI Backend is ready."}

@app.post("/api/suggest_questions", response_model=SuggestionResponse)
def suggest_questions(request: SuggestionRequest):
    history_str = "\n".join([f"User: {h.get('user', '')}\nBot: {h.get('bot', '')}" for h in request.history])

    prompt = f"""
    Based on the following conversation history between a user and an e-commerce chatbot, generate exactly three, concise, and highly relevant follow-up questions that the user might ask next.
    The questions should be natural and continue the flow of the conversation.
    Return ONLY a JSON list of strings. Do not add any other text.

    Example Format:
    ["What are the shipping costs?", "Can I track my order?", "What is the return policy for electronics?"]

    Conversation History:
    ---
    {history_str}
    ---

    JSON list of three suggested questions:
    """
    
    response = generation_model.generate_content(prompt)
    
    try:
        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        suggestions = json.loads(clean_response)
    except (json.JSONDecodeError, AttributeError):
        suggestions = ["How do I return an item?", "What are the payment options?", "How can I track my order?"]
        
    return SuggestionResponse(suggestions=suggestions)
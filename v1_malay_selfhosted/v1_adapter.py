# -- v1_malay_selfhosted/v1_adapter.py --

import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- 1. Constants and Configuration (Adapted from self-testing-app.py) ---
# CRITICAL: Point to the NEUTRALIZED index for a fair evaluation.
INDEX_SAVE_PATH = "faiss_v1_neutral_kb_index" 
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_CHECKPOINT = "google/flan-t5-small"
SEARCH_K = 2 # The optimal setting we found for V1

# --- 2. The Pipeline Loading Function (Adapted from Streamlit's cached function) ---
def load_v1_rag_pipeline():
    """
    Loads the entire V1 RAG pipeline (models, index, chain) into memory.
    This is a direct adaptation of the cached function from the Streamlit app.
    """
    print("--- Initializing V1 RAG Pipeline for Evaluation ---")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        
        # Load FAISS Index
        if not os.path.exists(INDEX_SAVE_PATH):
            print(f"FATAL: V1 neutral index not found at '{INDEX_SAVE_PATH}'. Please run the reindex script.")
            return None
        vector_store = FAISS.load_local(
            INDEX_SAVE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load LLM
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT, legacy=False)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT)
        pipeline_device = 0 if device == 'cuda' else -1
        pipe = pipeline(
            "text2text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            device=pipeline_device
        )
        llm_pipe = HuggingFacePipeline(pipeline=pipe)
        
        # Create Retriever and Chain
        retriever = vector_store.as_retriever(search_kwargs={'k': SEARCH_K})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_pipe,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        print("--- V1 RAG Pipeline Ready ---")
        return qa_chain
        
    except Exception as e:
        print(f"FATAL ERROR loading V1 pipeline: {e}")
        return None

# --- 3. Load the pipeline once when this module is imported ---
QA_CHAIN_V1 = load_v1_rag_pipeline()

# --- 4. The Adapter Function ---

def get_v1_rag_response(question: str):
    """
    Takes a question and returns the answer and retrieved contexts in a dictionary
    formatted for RAGAs evaluation.
    """
    if not QA_CHAIN_V1:
        return {"answer": "Error: V1 Chain not loaded.", "contexts": []}
        
    # LangChain v1 used .invoke(), which is equivalent to calling the chain.
    result = QA_CHAIN_V1.invoke({"query": question})
    
    answer = result.get('result', "")
    contexts = [doc.page_content for doc in result.get('source_documents', [])]
    
    return {"answer": answer, "contexts": contexts}

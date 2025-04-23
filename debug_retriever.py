# --- debug_retriever.py ---
import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import numpy as np # Keep imports needed
# from sklearn.metrics.pairwise import cosine_similarity # Uncomment if using direct similarity check

# --- Configuration ---
INDEX_PATH = "faiss_malay_ecommerce_kb_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Back to MiniLM
SEARCH_TYPE = "similarity" # Testing Similarity Search
SEARCH_K = 5

test_queries = [
    "Status Penghantaran",
    "Berapa lama tempoh pemulangan LazMall?",
    "Adakah produk ini original?",
    "Lazmall",
    "Hi"
]

print("--- Starting Retriever Debug Script ---")

# --- Load Embedding Model ---
print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    print(f">> Embedding model loaded on {device}.")
except Exception as e:
    print(f"FATAL: Error loading embedding model: {e}")
    exit()

# --- Load FAISS Index ---
print(f"\nLoading FAISS index from: {INDEX_PATH}...")
vector_store = None # Initialize vector_store
if not os.path.exists(INDEX_PATH):
    print(f"FATAL: FAISS index not found at {INDEX_PATH}. Run reindex.py first!")
    exit()
try:
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f">> FAISS index loaded. Contains {vector_store.index.ntotal} vectors.")
except Exception as e:
    print(f"FATAL: Error loading FAISS index: {e}")
    exit()

# --- Create Retriever ---
print(f"\nCreating retriever (Type: {SEARCH_TYPE}, k: {SEARCH_K})...")
retriever = None # <<< Initialize retriever to None >>>
if vector_store: # Ensure vector_store loaded successfully
    try:
        retriever = vector_store.as_retriever( # <<< Define retriever here >>>
            search_type=SEARCH_TYPE,
            search_kwargs={'k': SEARCH_K}
        )
        print(f">> Retriever created (Type: {SEARCH_TYPE}, k: {SEARCH_K}).") # Correct indentation
    except Exception as e:
        print(f"FATAL: Error creating retriever: {e}")
        # No exit() here yet, let the check below handle it

# --- Check if Retriever Creation Succeeded BEFORE Testing ---
if not retriever: # <<< Add this check >>>
    print("\nFATAL: Retriever object was not created successfully. Exiting.")
    exit()

# --- Test Queries ---
print("\n--- Testing Queries ---")
for query in test_queries:
    print(f"\n>>> Testing Query: '{query}'")
    try:
        start_time = time.time()
        # Now 'retriever' is guaranteed to exist if we reached here
        retrieved_docs = retriever.invoke(query)
        end_time = time.time()
        print(f"    Time taken: {end_time - start_time:.2f} seconds")
        print(f"    Retrieved {len(retrieved_docs)} documents.")

        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"\n    --- Doc {i+1} ---")
                print(f"    Source: {doc.metadata.get('source', 'N/A')}")
                print(f"    Content Snippet: {doc.page_content[:250]}...")
        else:
            print("    !!! No documents retrieved !!!")

    except Exception as e:
        # This except block should now only catch errors from .invoke()
        print(f"    ERROR running retriever invoke() for query '{query}': {e}")

print("DEBUG SCRIPTED FINISHED")
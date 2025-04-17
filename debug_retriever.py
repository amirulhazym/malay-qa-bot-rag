# --- debug_retriever.py ---
import os
import torch
# Use the specific, potentially newer imports if you updated based on warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time # To measure time if needed

# --- Configuration (Match your app_v3.py and reindex.py) ---
INDEX_PATH = "faiss_malay_ecommerce_kb_index"
# IMPORTANT: Use the SAME embedding model currently configured in your app/reindex scripts!
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Retrieval settings to test (match app_v3.py)
SEARCH_TYPE = "similarity" # Or "similarity"
SEARCH_K = 5
#SEARCH_FETCH_K = 10

# --- Queries to Test ---
test_queries = [
    "Status Penghantaran",
    "Berapa lama tempoh pemulangan LazMall?",
    "Adakah produk ini original?",
    "Lazmall", # A query known to work sometimes
    "Hi" # A query known to be irrelevant
]

print("--- Starting Retriever Debug Script ---")

# --- Load Embedding Model ---
print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}...")
try:
    # Note: No Streamlit caching here, loads every time script runs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    print(f">> Embedding model loaded on {device}.")
except Exception as e:
    print(f"FATAL: Error loading embedding model: {e}")
    exit() # Exit script if embeddings fail

# --- Load FAISS Index ---
print(f"\nLoading FAISS index from: {INDEX_PATH}...")
if not os.path.exists(INDEX_PATH):
    print(f"FATAL: FAISS index not found at {INDEX_PATH}. Run reindex.py first!")
    exit() # Exit script if index is missing
try:
    # Note: No Streamlit caching here
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f">> FAISS index loaded. Contains {vector_store.index.ntotal} vectors.")
except Exception as e:
    print(f"FATAL: Error loading FAISS index: {e}")
    exit() # Exit script if index fails

# --- Create Retriever ---
retriever = vector_store.as_retriever(
    search_type=SEARCH_TYPE,
    search_kwargs={'k': SEARCH_K}
)
print(f">> Retriever created (Type: {SEARCH_TYPE}, k: {SEARCH_K}).") # Update print statement
except Exception as e:
    print(f"FATAL: Error creating retriever: {e}")
    exit()

# --- Test Queries ---
print("\n--- Testing Queries ---")
for query in test_queries:
    print(f"\n>>> Testing Query: '{query}'")
    try:
        start_time = time.time()
        # Use .invoke() which is the newer standard for retrievers too
        retrieved_docs = retriever.invoke(query)
        end_time = time.time()
        print(f"    Time taken: {end_time - start_time:.2f} seconds")
        print(f"    Retrieved {len(retrieved_docs)} documents.")

        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"\n    --- Doc {i+1} ---")
                print(f"    Source: {doc.metadata.get('source', 'N/A')}")
                print(f"    Content Snippet: {doc.page_content[:250]}...") # Show a snippet
                # OPTIONAL: Calculate direct similarity if needed for deeper debug
                # query_embedding = np.array(embeddings.embed_query(query)).reshape(1, -1)
                # doc_embedding = np.array(embeddings.embed_documents([doc.page_content])[0]).reshape(1, -1)
                # similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
                # print(f"    Direct Cosine Similarity to Query: {similarity:.4f}")
        else:
            print("    !!! No documents retrieved !!!")

    except Exception as e:
        print(f"    ERROR running retriever for query '{query}': {e}")

print("\n--- Debug Script Finished ---")
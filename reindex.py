# --- reindex.py ---
# Purpose: Re-create the FAISS index locally based on the knowledge_base folder.

import os
import torch
import time
# Use updated imports for newer LangChain versions
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("--- Starting Local Re-indexing Script ---")

# --- Configuration (Using relative paths for local execution) ---
KB_DIR = "knowledge_base" # Assumes 'knowledge_base' is in the same dir as reindex.py
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index" # Assumes save dir is in the same dir
EMBEDDING_MODEL_NAME = "mesolitica/mistral-embedding-191m-8k-contrastive"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

# --- Step 1: Load Documents ---
print(f"\n[1/4] Loading documents from: '{KB_DIR}'")
if not os.path.isdir(KB_DIR):
    print(f"!!! ERROR: Knowledge base directory '{KB_DIR}' not found in {os.getcwd()}.")
    print("!!! Please ensure the folder exists and contains your updated .txt files.")
    exit() # Stop the script if KB directory is missing

docs = [] # Initialize docs list
try:
    loader = DirectoryLoader(
        KB_DIR,
        glob="**/*.txt", # Load all .txt files recursively
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}, # Specify encoding
        show_progress=True,
        use_multithreading=False # Can be safer for local runs
    )
    documents = loader.load()
    print(f"--- Successfully loaded {len(documents)} document(s).")

except Exception as e:
    print(f"!!! ERROR loading documents: {e}")
    exit()

# --- Step 2: Split Documents ---
if documents:
    print(f"\n[2/4] Splitting {len(documents)} document(s) into chunks...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents)
        print(f"--- Successfully split into {len(docs)} chunks.")
    except Exception as e:
        print(f"!!! ERROR splitting documents: {e}")
        exit()
else:
    print("--- No documents loaded, skipping chunking and indexing.")
    docs = []

# --- Step 3: Load Embedding Model ---
# Only proceed if we have chunks to index
if docs:
    print(f"\n[3/4] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = None # Initialize
    try:
        # Determine device (CPU is most likely locally)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- Using device: {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False} # Usually False is fine
        )
        print(f"--- Embedding model loaded successfully.")
    except Exception as e:
        print(f"!!! ERROR loading embedding model: {e}")
        exit()

    # --- Step 4: Create and Save FAISS Index ---
    if embeddings:
        print(f"\n[4/4] Creating FAISS index from {len(docs)} chunks (this may take time on CPU)...")
        try:
            start_time = time.time()
            # Create index from documents and embeddings
            vectorstore = FAISS.from_documents(docs, embeddings)
            end_time = time.time()
            print(f"--- FAISS index created in memory. Time taken: {end_time - start_time:.2f} seconds.")

            # Save the index locally
            print(f"--- Saving FAISS index to: '{INDEX_SAVE_PATH}'")
            vectorstore.save_local(INDEX_SAVE_PATH)
            print("--- FAISS index saved successfully.")

        except Exception as e:
            print(f"!!! ERROR creating/saving FAISS index: {e}")
            exit()
    else:
        print("!!! ERROR: Embedding model failed to load, cannot create index.")

else:
    print("\n--- No document chunks found. Index not created or updated. ---")


print("\n--- Local Re-indexing Script Finished ---")
# --- reindex.py ---
# Purpose: Load documents from a specified directory, chunk them,
#          load a specified embedding model, create a FAISS index,
#          and save the index locally.

import os
import torch
import time
import argparse # For command-line arguments

# --- Attempt LangChain Imports (Community First) ---
try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("Using langchain_community imports.")
except ImportError:
    print("langchain_community not found, falling back to older langchain imports...")
    try:
        from langchain.document_loaders import DirectoryLoader, TextLoader
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        print("!!! ERROR: Could not import necessary LangChain components.")
        print("!!! Please ensure 'langchain', 'langchain-community', 'langchain-huggingface',")
        print("!!! 'faiss-cpu', 'sentence-transformers', 'torch', 'pandas' are installed.")
        exit(1) # Exit with error code

# Must import this separately for chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("--- Starting Local Re-indexing Script ---")

# --- Configuration via Command-Line Arguments ---
parser = argparse.ArgumentParser(description="Re-index knowledge base for RAG using FAISS.")
parser.add_argument("--kb-dir", type=str, default="knowledge_base", help="Directory containing knowledge base .txt files.")
parser.add_argument("--index-path", type=str, default="faiss_malay_ecommerce_kb_index", help="Path to save the created FAISS index.")
parser.add_argument("--embedding-model", type=str, default="mesolitica/mistral-embedding-191m-8k-contrastive", help="Hugging Face embedding model name (Sentence Transformer compatible).")
parser.add_argument("--chunk-size", type=int, default=300, help="Maximum characters per text chunk.")
parser.add_argument("--chunk-overlap", type=int, default=50, help="Character overlap between chunks.")
parser.add_argument("--device", type=str, default="auto", choices=['auto', 'cuda', 'cpu'], help="Device for embedding model ('cuda', 'cpu', 'auto').")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation.")
parser.add_argument("--normalize-embeddings", action='store_true', help="Normalize embeddings before indexing (use for cosine similarity search).")

# Parse arguments from command line
args = parser.parse_args()

# --- Determine Device ---
if args.device == "auto":
    selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    selected_device = args.device
print(f"--- Using device: {selected_device}")

# --- Step 1: Load Documents ---
print(f"\n[1/4] Loading documents from directory: '{args.kb_dir}'")
if not os.path.isdir(args.kb_dir):
    print(f"!!! ERROR: Knowledge base directory '{args.kb_dir}' not found in '{os.getcwd()}'.")
    print("!!! Please create the directory and add your .txt files.")
    exit(1)

all_documents = []
try:
    # Use DirectoryLoader to handle loading multiple files
    loader = DirectoryLoader(
        args.kb_dir,
        glob="**/*.txt", # Pattern to match text files
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}, # Ensure correct encoding
        show_progress=True,
        use_multithreading=True # Speed up loading if many files
    )
    all_documents = loader.load() # Load documents into LangChain Document objects

    if not all_documents:
        print("--- WARNING: No .txt documents found in the specified directory.")
        # Allow script to continue, will result in empty index if no docs
    else:
        print(f"--- Successfully loaded {len(all_documents)} document(s).")

except Exception as e:
    print(f"!!! ERROR loading documents: {e}")
    exit(1)

# --- Step 2: Split Documents into Chunks ---
docs_chunked = [] # Initialize list for chunked documents
if all_documents: # Only split if documents were loaded
    print(f"\n[2/4] Splitting {len(all_documents)} document(s) into chunks...")
    print(f"--- Chunk Size: {args.chunk_size}, Chunk Overlap: {args.chunk_overlap}")
    try:
        # Use RecursiveCharacterTextSplitter for robust chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            # Default separators are usually good: ["\n\n", "\n", " ", ""]
        )
        docs_chunked = text_splitter.split_documents(all_documents)

        if not docs_chunked:
             print("--- WARNING: Splitting resulted in zero chunks. Check document content or splitter settings.")
        else:
             print(f"--- Successfully split into {len(docs_chunked)} chunks.")
             # Optional: Print a sample chunk for verification
             # print("\n--- Sample Chunk 0 ---")
             # print(docs_chunked[0].page_content[:300] + "...")
             # print(f"Metadata: {docs_chunked[0].metadata}")
             # print("---------------------")

    except Exception as e:
        print(f"!!! ERROR splitting documents: {e}")
        exit(1)
else:
    print("--- Skipping document splitting as no documents were loaded.")

# --- Step 3: Load Embedding Model ---
print(f"\n[3/4] Loading embedding model: {args.embedding_model}...")

# Define cache folder path (uses .cache_st in current dir)
cache_dir_st = os.path.join(os.getcwd(), ".cache_st")
os.makedirs(cache_dir_st, exist_ok=True)
print(f"--- Using cache directory: {cache_dir_st}")

embeddings = None # Initialize variable
try:
    # Instantiate the LangChain wrapper
    # Pass cache_folder as a TOP-LEVEL argument as determined by testing
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={ # Arguments for the underlying SentenceTransformer model
            'device': selected_device,
            # DO NOT put cache_folder here based on previous error
        },
        encode_kwargs={ # Arguments for the .encode() method
            'normalize_embeddings': args.normalize_embeddings, # Control normalization
            'batch_size': args.batch_size
        },
        cache_folder=cache_dir_st # Specify cache_folder HERE at the top level
    )
    print(f"--- Embedding model '{args.embedding_model}' loaded successfully.")

except Exception as e:
    print(f"!!! ERROR loading embedding model via LangChain: {e}")
    # Provide guidance based on potential errors
    if "ConnectionError" in str(e) or "Max retries exceeded" in str(e):
        print("!!! Suggestion: Check internet connection and proxy settings (if required).")
    elif "multiple values for keyword argument 'cache_folder'" in str(e):
         print("!!! Suggestion: Internal error - cache_folder specified incorrectly. Check code.")
    elif "got an unexpected keyword argument" in str(e):
         print("!!! Suggestion: Argument mismatch - Check HuggingFaceEmbeddings parameters or model_kwargs.")
    else:
         print("!!! Suggestion: Check model name and installation of sentence-transformers, torch.")
    exit(1) # Exit if model fails to load


# --- Step 4: Create and Save FAISS Index ---
# Only proceed if we have chunks AND the embedding model loaded
if docs_chunked and embeddings:
    print(f"\n[4/4] Creating FAISS index from {len(docs_chunked)} chunks...")
    print(f"--- Using device: {selected_device} for embedding calculation within FAISS.")
    try:
        start_time = time.time()
        # Create index using FAISS.from_documents
        # This will internally call embeddings.embed_documents(chunk_texts)
        vectorstore = FAISS.from_documents(
            documents=docs_chunked, # Pass the list of LangChain Document objects
            embedding=embeddings    # Pass the instantiated HuggingFaceEmbeddings object
        )
        end_time = time.time()
        print(f"--- FAISS index created in memory. Time taken: {end_time - start_time:.2f} seconds.")

        # Save the index locally
        index_dir = os.path.dirname(args.index_path)
        if index_dir and not os.path.exists(index_dir):
            print(f"--- Creating directory for index: {index_dir}")
            os.makedirs(index_dir)

        print(f"--- Saving FAISS index to: '{args.index_path}'")
        vectorstore.save_local(folder_path=args.index_path) # Save to specified path
        print("--- FAISS index saved successfully.")
        print(f"--- Index contains {vectorstore.index.ntotal} vectors.")

    except TypeError as e:
        # Catch the specific 'input_ids' error if it occurs with a different model
        if "got an unexpected keyword argument 'input_ids'" in str(e):
             print(f"!!! ERROR during FAISS creation: {e}")
             print(f"!!! This likely means the embedding model '{args.embedding_model}' (potentially Mistral type)")
             print("!!! is incompatible with the default HuggingFaceEmbeddings -> FAISS workflow.")
             print("!!! RECOMMENDATION: Use a standard Sentence Transformer model instead, like:")
             print("!!! 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'")
             print("!!! Specify it using: --embedding-model 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'")
        else:
             print(f"!!! ERROR creating/saving FAISS index (TypeError): {e}")
        exit(1)
    except Exception as e:
        print(f"!!! ERROR creating/saving FAISS index: {e}")
        # Consider adding more specific error handling if needed
        exit(1)

elif not docs_chunked:
    print("\n--- No document chunks found. Index not created. ---")
else: # embeddings object is None
    print("\n--- Embedding model failed to load earlier. Index not created. ---")


print("\n--- Local Re-indexing Script Finished ---")
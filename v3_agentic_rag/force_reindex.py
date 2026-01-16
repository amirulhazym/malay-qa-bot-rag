import os
import shutil
from app.utils import setup_retriever

# 1. Clean up old databases to prevent corruption
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("🗑️  Deleted old ChromaDB.")

if os.path.exists("bm25_retriever.pkl"):
    os.remove("bm25_retriever.pkl")
    print("🗑️  Deleted old BM25 Index.")

# 2. Re-run the setup
print("🔄 Re-building Indexes (This handles the Bonus Point)...")
try:
    # This calls your existing logic but forces it to run from scratch
    # because we deleted the persistence files
    retriever = setup_retriever() 
    print("✅ SUCCESS: Hybrid Search (BM25 + Vector) is ready.")
except Exception as e:
    print(f"❌ ERROR: {e}")

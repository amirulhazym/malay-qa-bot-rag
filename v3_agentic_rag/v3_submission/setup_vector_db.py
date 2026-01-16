import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Load environment variables
load_dotenv()

def setup_database():
    print("--- 1. LOADING DOCUMENTS ---")
    data_path = "data/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created '{data_path}' folder. Please place PDF files there and run again.")
        return

    # Load PDFs
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    if not docs:
        print("No documents found in 'data/'.")
        return

    print(f"Loaded {len(docs)} documents.")

    print("--- 2. SPLITTING TEXT ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("--- 3. CREATING HYBRID RETRIEVER ---")
    
    # 3a. Vector Store (Chroma)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    persist_dir = "chroma_db"
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3b. Keyword Search (BM25)
    # We create BM25 from the splits
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5
    
    # Save BM25 retriever to disk (since it's in-memory)
    with open("bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print("Vector store saved to './chroma_db'")
    print("BM25 retriever saved to './bm25_retriever.pkl'")
    
    # 3c. Verify Ensemble (Just to show it works)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    
    print("--- SETUP COMPLETE ---")

if __name__ == "__main__":
    setup_database()

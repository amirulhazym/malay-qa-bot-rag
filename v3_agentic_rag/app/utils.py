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

def setup_retriever(data_path: str = "data", persist_dir: str = "chroma_db"):
    """
    Initializes and returns a Hybrid Retriever (Ensemble: BM25 + Chroma).
    If the vector DB doesn't exist, it processes documents from `data_path`.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    bm25_path = os.path.join(os.path.dirname(persist_dir), "bm25_retriever.pkl")

    # Check if vectorstore exists to avoid re-ingestion every run
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("✅ Loading existing Vector Store...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
        # Try loading BM25
        if os.path.exists(bm25_path):
            print("✅ Loading existing BM25 Retriever...")
            with open(bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
        else:
            print("⚠️ BM25 not found. Hybrid search might fail. Please re-ingest.")
            return vectorstore.as_retriever(search_kwargs={"k": 5})

    else:
        print("⚡ Initialization: Processing Documents...")
        
        # 1. Load Documents
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"⚠️ Data directory '{data_path}' created. Please add PDFs.")
            return None

        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()

        if not docs:
            print("⚠️ No documents found. Retriever will be empty.")
            return None

        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"📄 Processed {len(splits)} chunks.")

        # 3. Create Vector Store (Chroma)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        # 4. Create BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5
        
        # Save BM25
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

    # 5. Create Ensemble Retriever
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

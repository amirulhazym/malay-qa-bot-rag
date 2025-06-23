# Full Code for: v2_multilingual_api/backend/index_knowledge_base.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Please set it in your .env file.")

print("Loading documents from multilingual knowledge base...")
# This path is relative to the root, so we run it from the root folder
loader = DirectoryLoader('knowledge_base/v2_multilingual/', glob="**/*.md", show_progress=True)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

print("Initializing MULTILINGUAL embedding model...")
# Use the powerful multilingual model for embeddings
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "auracart-multilingual-kb"

if index_name not in pc.list_indexes().names():
    print(f"Creating new serverless index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=model.get_sentence_embedding_dimension(), # 768
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(index_name)
print("Pinecone index is ready.")

print("Embedding chunks and uploading to Pinecone...")
batch_size = 100
for i in range(0, len(docs), batch_size):
    i_end = min(i + batch_size, len(docs))
    batch = docs[i:i_end]
    texts = [doc.page_content for doc in batch]
    metadata = [{"source": doc.metadata.get('source', 'unknown'), "text": doc.page_content} for doc in batch]
    embeddings = model.encode(texts).tolist()
    ids = [f"doc_{i+j}" for j in range(len(batch))]
    index.upsert(vectors=zip(ids, embeddings, metadata))
    print(f"Uploaded batch {i // batch_size + 1}")
print("\n--- Multilingual Knowledge Base Indexing Complete ---")

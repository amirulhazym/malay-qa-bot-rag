# --- app.py (Combined App & Re-indexing) ---
# Purpose: Runs Streamlit Chat UI & includes function to rebuild FAISS index.

import streamlit as st
import time
import torch
import os
import re
import traceback
# LangChain/Community/HF Imports
# Using newer paths where possible, assuming recent langchain installation
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline # Deprecated but using for consistency
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# Base Transformers
from transformers import AutoModel, AutoTokenizer, pipeline
# Other
import numpy as np
from typing import List

# --- Page Config & Constants ---
st.set_page_config(page_title="Bot Soal Jawab BM", page_icon="🇲🇾", layout="centered")

# --- !! CONFIGURATION !! ---
KB_DIR = "knowledge_base" # Relative path to KB folder
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index" # Relative path for FAISS index
# --- Choose Your Embedding Model ---
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"
EMBEDDING_MODEL_NAME = "mesolitica/mistral-embedding-191m-8k-contrastive" # Using Mesolitica
# --- Choose Your Generative LLM ---
LLM_CHECKPOINT = "google/mt5-small" # Keeping mt5-small for now
# --- UI Constants ---
ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "👤"
HEADER_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/194/194938.png"
# --- Chunking Constants (for re-indexing) ---
CHUNK_SIZE = 1000 # Adjust as needed (e.g., 500)
CHUNK_OVERLAP = 150 # Adjust as needed (e.g., 50)
# --- !! END CONFIGURATION !! ---


# --- Custom Embedder Class (Using Direct .encode()) ---
class MistralDirectEmbeddings(Embeddings):
    """Custom LangChain Embeddings class using Mesolitica's direct .encode()."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # Add checks to prevent redundant console prints during Streamlit reruns
        if "custom_embedder_loaded" not in st.session_state:
            print(f">> Initializing Custom Embedder: {model_name}")
            st.session_state.custom_embedder_loaded = True # Mark as loaded for this session

        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if "custom_embedder_device" not in st.session_state:
            print(f">> Using device: {self.device}")
            st.session_state.custom_embedder_device = self.device

        try:
            # Load only once and store references if needed, or rely on from_pretrained cache
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            if "custom_embedder_model_loaded" not in st.session_state:
                 print(">> Custom embedder model and tokenizer loaded.")
                 st.session_state.custom_embedder_model_loaded = True

        except Exception as e:
            # Use Streamlit error reporting if possible during init
            st.error(f"!!! ERROR initializing custom embedder: {e}")
            traceback.print_exc() # Print full traceback to console
            # Stop the app if the embedder fails catastrophically
            st.stop()

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.array([])
        try:
            inputs = self.tokenizer(
                texts, return_tensors='pt', padding=True, truncation=True,
                max_length=8192 # Use model's max length
            ).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            return embeddings.detach().cpu().numpy()
        except Exception as e:
            print(f"!!! ERROR during custom embedding: {e}")
            traceback.print_exc() # Print error to console
            st.error(f"Ralat semasa mengira embedding: {e}") # Show error in UI
            return np.array([]) # Return empty, handle downstream

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f">> Custom embed_documents called for {len(texts)} texts.")
        embeddings_np = self._embed(texts)
        if embeddings_np.size == 0 and len(texts) > 0:
            print("!!! WARNING: embed_documents received empty embeddings.")
            # Determine expected dimension dynamically if possible
            embed_dim = getattr(getattr(self.model, 'config', None), 'hidden_size', 768)
            return [[0.0] * embed_dim] * len(texts)
        return embeddings_np.tolist()

    def embed_query(self, text: str) -> List[float]:
        print(f">> Custom embed_query called for query: '{text[:50]}...'")
        embeddings_np = self._embed([text])
        if embeddings_np.size == 0:
            print("!!! WARNING: embed_query received empty embeddings.")
            embed_dim = getattr(getattr(self.model, 'config', None), 'hidden_size', 768)
            return [0.0] * embed_dim
        # Ensure it returns a flat list, not a list containing a list
        return embeddings_np.flatten().tolist()

# --- Re-indexing Function ---
def rebuild_index(embedding_instance: Embeddings):
    """Loads KB, chunks, embeds using provided instance, saves new FAISS index."""
    st.sidebar.info(f"Memulakan proses re-indexing...\nKB: {KB_DIR}\nChunk: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
    overall_start_time = time.time()

    # --- 1. Load Documents ---
    status_placeholder = st.sidebar.empty()
    status_placeholder.write("[1/4] Memuatkan dokumen...")
    print(f"\n[Rebuild] Loading documents from: '{KB_DIR}'")
    if not os.path.isdir(KB_DIR):
        st.sidebar.error(f"Direktori KB '{KB_DIR}' tidak dijumpai.")
        return False
    docs = []
    try:
        loader = DirectoryLoader(
            KB_DIR, glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}, show_progress=False, # Progress bar in UI instead
            use_multithreading=False
        )
        documents = loader.load()
        print(f"[Rebuild] Loaded {len(documents)} document(s).")
        if not documents:
             st.sidebar.warning("Tiada dokumen ditemui dalam KB.")
             return False # Nothing to index
    except Exception as e:
        st.sidebar.error(f"Ralat memuatkan dokumen: {e}")
        traceback.print_exc()
        return False

    # --- 2. Split Documents ---
    status_placeholder.write("[2/4] Memecahkan dokumen...")
    print(f"[Rebuild] Splitting {len(documents)} document(s)...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents)
        print(f"[Rebuild] Split into {len(docs)} chunks.")
        if not docs:
            st.sidebar.warning("Tiada chunks terhasil selepas pemecahan.")
            return False # Nothing to index
    except Exception as e:
        st.sidebar.error(f"Ralat memecahkan dokumen: {e}")
        traceback.print_exc()
        return False

    # --- 3. Check Embedding Instance ---
    # Embedding model already loaded and passed as argument 'embedding_instance'
    if not embedding_instance:
         st.sidebar.error("Instance model embedding tidak sah.")
         return False
    print("[Rebuild] Menggunakan instance embedding model sedia ada.")
    status_placeholder.write("[3/4] Menggunakan model embedding sedia ada...")

    # --- 4. Create and Save FAISS Index ---
    status_placeholder.write(f"[4/4] Mencipta index FAISS ({len(docs)} chunks)... (Mungkin lambat)")
    print(f"[Rebuild] Creating FAISS index from {len(docs)} chunks...")
    index_creation_time = time.time()
    try:
        # Delete old index folder first for a clean save
        if os.path.exists(INDEX_SAVE_PATH):
            print(f"[Rebuild] Removing old index folder: {INDEX_SAVE_PATH}")
            import shutil
            shutil.rmtree(INDEX_SAVE_PATH)

        # Create index - This calls embedding_instance.embed_documents()
        vectorstore = FAISS.from_documents(docs, embedding_instance)
        print(f"[Rebuild] Index created in memory. Time: {time.time() - index_creation_time:.2f}s")

        # Save the index locally
        print(f"[Rebuild] Saving FAISS index to: '{INDEX_SAVE_PATH}'")
        vectorstore.save_local(INDEX_SAVE_PATH)
        print("[Rebuild] FAISS index saved successfully.")
        status_placeholder.empty() # Clear status message
        overall_time = time.time() - overall_start_time
        st.sidebar.success(f"Re-indexing selesai!\n({len(docs)} chunks, {overall_time:.1f}s)")
        st.sidebar.warning("SILA RESTART Streamlit (Ctrl+C & `streamlit run app.py`) untuk memuatkan index baru.") # IMPORTANT instruction
        # Clear specific cache? Difficult for resources. Restart is reliable.
        # st.cache_resource.clear() # Clears ALL resource caches, might reload LLM too
        return True

    except Exception as e:
        status_placeholder.empty()
        st.sidebar.error(f"Ralat mencipta/menyimpan index FAISS: {e}")
        traceback.print_exc()
        return False


# --- Utility Function to Clean LLM Output ---
def clean_llm_output(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<pad>', '', text)
    cleaned_text = text.strip()
    # Check if only whitespace or common punctuation remains
    if not cleaned_text or all(c in ' .,;:!?\n\t-' for c in cleaned_text):
        # Keep the specific fallback message consistent
        return "Maaf, saya tidak pasti jawapannya berdasarkan maklumat ini."
        # Or maybe return None/empty string and let the calling code handle it
        # return ""
    return cleaned_text


# --- Cached Loading Functions Using Custom Embedder ---
@st.cache_resource # Cache the custom embedder instance
def load_embeddings_model():
    """Loads the custom MistralDirectEmbeddings model."""
    # Initialization logic moved inside the class __init__
    # The decorator caches the *instance* of the class
    try:
        embed_model = MistralDirectEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return embed_model
    except Exception as e:
        # Error handling done inside __init__, but add a stop here too
        st.error(f"Gagal kritikal semasa memuatkan model embedding custom.")
        st.stop()


@st.cache_resource # Cache the loaded FAISS index (depends on embed_model instance)
def load_faiss_index(_embeddings: Embeddings): # Type hint
    """Loads the FAISS index from local path using the provided embedder instance."""
    # This will only run again if _embeddings object changes (new session) OR cache cleared
    if "faiss_loaded_msg" not in st.session_state:
         print(f">> (Cache Trigger) Loading FAISS index from: {INDEX_SAVE_PATH}...")
         st.session_state.faiss_loaded_msg = True
    if not _embeddings:
         st.error("Tidak dapat memuatkan index FAISS tanpa model embedding.")
         return None
    if not os.path.exists(INDEX_SAVE_PATH):
        st.error(f"Index FAISS tidak dijumpai di {INDEX_SAVE_PATH}. Sila bina semula menggunakan butang di sidebar.")
        return None
    try:
        vector_store = FAISS.load_local(
            INDEX_SAVE_PATH,
            _embeddings, # Pass the embedder instance
            allow_dangerous_deserialization=True
        )
        if "faiss_vector_count" not in st.session_state:
             print(f">> FAISS index loaded. Contains {vector_store.index.ntotal} vectors.")
             st.session_state.faiss_vector_count = vector_store.index.ntotal
        return vector_store
    except Exception as e:
        st.error(f"Ralat memuatkan index FAISS: {e}")
        traceback.print_exc()
        return None


@st.cache_resource # Cache the LLM pipeline
def load_llm_qa_pipeline():
    """Loads the LLM pipeline for generation."""
    if "llm_loaded_msg" not in st.session_state:
        print(f">> (Cache Trigger) Loading LLM pipeline: {LLM_CHECKPOINT}...")
        st.session_state.llm_loaded_msg = True
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT, trust_remote_code=True) # Add trust_remote_code just in case
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT, trust_remote_code=True)
        # Determine device for LLM pipeline
        llm_device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text2text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=100, # Keep reasonable limit
            device=llm_device
        )
        llm_pipe = HuggingFacePipeline(pipeline=pipe)
        if "llm_device_msg" not in st.session_state:
            print(f">> LLM pipeline loaded on device {llm_device}.")
            st.session_state.llm_device_msg = llm_device
        return llm_pipe
    except Exception as e:
        st.error(f"Ralat memuatkan LLM pipeline: {e}")
        traceback.print_exc()
        st.stop()


# --- Main App Execution Flow ---

# --- Sidebar for Re-indexing ---
st.sidebar.title("Panel Kawalan")
st.sidebar.markdown("Gunakan butang di bawah untuk membina semula index vektor FAISS jika anda mengemaskini fail dalam folder `knowledge_base`.")
st.sidebar.warning("Proses ini mungkin mengambil masa beberapa minit pada CPU.")
if st.sidebar.button("Bina Semula Index FAISS"):
    # Load embedder model (will be cached if already loaded)
    current_embedder = load_embeddings_model()
    if current_embedder:
        # Run the re-indexing function
        rebuild_success = rebuild_index(current_embedder)
        # No explicit cache clearing here, rely on user restarting Streamlit

# --- Load Resources & Create Chain ---
# These will use cached versions after the first run per session
embeddings_model = load_embeddings_model()
vector_store = load_faiss_index(embeddings_model) # Pass the potentially cached embedder
llm_pipeline = load_llm_qa_pipeline()

qa_chain = None
if vector_store and llm_pipeline:
    # Prevent recreating chain on every minor rerun if components are same
    if "qa_chain_created" not in st.session_state or not st.session_state.qa_chain_created:
        print(">> Creating/Recreating QA Chain...")
        try:
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 3, 'fetch_k': 10}
            )
            # Define Custom Prompt Template (ensure it's defined)
            prompt_template_text = """Gunakan konteks berikut untuk menjawab soalan di akhir. Jawab hanya berdasarkan konteks yang diberikan. Jika jawapan tiada dalam konteks, nyatakan "Maaf, maklumat tiada dalam pangkalan data.".

            Konteks:
            {context}

            Soalan: {question}
            Jawapan Membantu:"""
            PROMPT = PromptTemplate(
                template=prompt_template_text, input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": PROMPT}
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_pipeline,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs
            )
            print(">> QA Chain ready.")
            st.session_state.qa_chain_created = True # Mark as created
            st.session_state.qa_chain_instance = qa_chain # Store instance if needed
        except Exception as e:
            st.error(f"Ralat mencipta QA chain: {e}")
            traceback.print_exc()
            st.session_state.qa_chain_created = False
    else:
         # Reuse stored chain if possible (though chain itself is usually cheap to recreate)
         qa_chain = st.session_state.get("qa_chain_instance")

# --- Inject Custom CSS ---
# ... (CSS remains the same) ...
st.markdown("""<style>/* ... CSS here ... */</style>""", unsafe_allow_html=True)

# --- Custom Header ---
# ... (Header markdown remains the same) ...
st.markdown(f"""<div class="chat-header">...</div>""", unsafe_allow_html=True)

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": "Salam! 👋 Taip soalan anda di bawah.", "id": 0} # Simplified initial message
    ]
if not all("id" in msg for msg in st.session_state.messages):
     for i, msg in enumerate(st.session_state.messages): msg["id"] = i

# --- Display Chat History ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Taip mesej anda..."):
    st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": prompt, "id": len(st.session_state.messages)})
    # Force immediate rerun to display user message before processing
    st.rerun()

# --- Generate Response if Last Message is from User ---
# Check based on ID to prevent infinite loops with rerun
last_message_id = st.session_state.messages[-1].get("id", -1) if st.session_state.messages else -1
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and \
   (last_message_id > st.session_state.get("last_processed_id", -1)):

    last_user_message = st.session_state.messages[-1]["content"]
    st.session_state.last_processed_id = last_message_id # Mark as being processed

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        response_placeholder = st.empty() # Placeholder for streaming or final answer
        sources_expander_placeholder = st.expander("Lihat Sumber Rujukan", expanded=False)
        caption_placeholder = st.empty()

        # Check if RAG chain is ready
        if not qa_chain:
            response_placeholder.error("Maaf, sistem RAG tidak bersedia. Sila pastikan index FAISS dimuatkan.")
            assistant_final_content = "Maaf, sistem RAG tidak bersedia."
        else:
            with response_placeholder.status("Mencari jawapan...", expanded=False): # Use status UI
                try:
                    start_time = time.time()
                    print(f">> Running QA chain for query: '{last_user_message[:50]}...'")
                    result = qa_chain({"query": last_user_message})
                    end_time = time.time()
                    processing_time = end_time - start_time

                    generated_answer_raw = result.get('result', "Maaf, ralat semasa menjana jawapan.")
                    source_docs = result.get('source_documents', [])

                    # Fallback Logic
                    if "<extra_id_" in generated_answer_raw and source_docs:
                        fallback_source_content = source_docs[0].page_content
                        fallback_source_content = re.sub(r'\s+', ' ', fallback_source_content).strip()
                        assistant_final_content = f"Saya tidak pasti jawapan tepat, tetapi berikut adalah maklumat berkaitan yang ditemui:\n\n---\n_{fallback_source_content}_"
                        print(">> LLM failed (<extra_id>), falling back to first source.")
                    elif "<extra_id_" in generated_answer_raw:
                        assistant_final_content = "Maaf, saya tidak pasti jawapannya berdasarkan maklumat yang ada."
                        print(">> LLM failed (<extra_id>), no sources.")
                    else:
                        assistant_final_content = clean_llm_output(generated_answer_raw)
                        print(">> LLM generated response, applying cleaning.")

                except Exception as e:
                    st.error(f"Ralat semasa memproses RAG: {e}")
                    traceback.print_exc()
                    assistant_final_content = "Maaf, berlaku ralat semasa mencari jawapan."
                    source_docs = [] # Ensure source_docs is empty on error
                    processing_time = 0

            # Update placeholders AFTER status block finishes
            response_placeholder.markdown(assistant_final_content)
            with sources_expander_placeholder:
                if source_docs:
                    for k, doc in enumerate(source_docs):
                        source_name = doc.metadata.get('source', f'Sumber {k+1}')
                        st.caption(f"**{source_name}:**")
                        # Use st.text or st.code for better formatting of potentially long source text
                        st.text(doc.page_content)
                elif qa_chain: # Only show no sources if chain was supposed to run
                     st.caption("Tiada sumber rujukan khusus ditemui.")

            if processing_time > 0:
                 caption_placeholder.caption(f"Masa mencari: {processing_time:.2f} saat")

    # Add the final response to session state *once*
    st.session_state.messages.append({
        "role": "assistant",
        "avatar": ASSISTANT_AVATAR,
        "content": assistant_final_content,
        "id": len(st.session_state.messages) # Ensure unique ID
    })
    # We already did st.rerun() after user input, so it should display now.
    # Avoid another rerun here unless absolutely necessary.
# --- app.py (Chat UI Version) ---
import streamlit as st
import time
import torch
from langchain_huggingface import HuggingFaceEmbeddings # Correct import path
from langchain_community.vectorstores import FAISS      # Correct import path
from langchain_community.llms import HuggingFacePipeline # Correct import path
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os

# --- Page Config ---
st.set_page_config(page_title="Bot Soal Jawab BM", page_icon="🇲🇾", layout="centered")

# --- Constants ---
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_CHECKPOINT = "google/mt5-small"
ASSISTANT_AVATAR = "🤖" # Or use a URL: "https://..."
USER_AVATAR = "👤"

# --- Cached Loading Functions (Keep these as they are essential) ---

@st.cache_resource
def load_embeddings_model():
    """Loads the Sentence Transformer embedding model."""
    print(">> (Cache) Loading embedding model...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        print(f">> Embedding model loaded on {device}.")
        return embed_model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop() # Stop execution if embeddings fail

@st.cache_resource
def load_faiss_index(_embeddings):
    """Loads the FAISS index from local path."""
    print(f">> (Cache) Loading FAISS index from: {INDEX_SAVE_PATH}...")
    if not _embeddings:
         st.error("Cannot load FAISS index without embedding model.")
         return None # Allow app to continue but show error
    if not os.path.exists(INDEX_SAVE_PATH):
        st.error(f"FAISS index not found at {INDEX_SAVE_PATH}. Pastikan ia wujud hasil dari Notebook Level 2.")
        return None # Allow app to continue but show error
    try:
        vector_store = FAISS.load_local(
            INDEX_SAVE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        print(f">> FAISS index loaded. Contains {vector_store.index.ntotal} vectors.")
        return vector_store
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None # Allow app to continue but show error

@st.cache_resource
def load_llm_qa_pipeline():
    """Loads the LLM pipeline for generation."""
    print(f">> (Cache) Loading LLM pipeline: {LLM_CHECKPOINT}...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text2text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=150, # Limit generated tokens
            # temperature=0.7, # Optionally adjust creativity
            device=device
        )
        # Note: Using HuggingFacePipeline is deprecated, but kept for consistency with original code
        # Consider replacing with direct pipeline usage or newer LangChain integrations if updating further.
        llm_pipe = HuggingFacePipeline(pipeline=pipe)
        print(f">> LLM pipeline loaded on device {device}.")
        return llm_pipe
    except Exception as e:
        st.error(f"Error loading LLM pipeline: {e}")
        st.stop() # Stop execution if LLM fails

# --- Load Resources ---
# These functions run only once thanks to @st.cache_resource
embeddings_model = load_embeddings_model()
vector_store = load_faiss_index(embeddings_model)
llm_pipeline = load_llm_qa_pipeline()

# --- Create QA Chain (only if vector_store loaded successfully) ---
qa_chain = None
if vector_store and llm_pipeline:
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_pipeline,
            chain_type="stuff", # Stuffs context into prompt - might hit token limits
            retriever=retriever,
            return_source_documents=True # Get sources back
        )
        print(">> QA Chain ready.")
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        # qa_chain remains None

# --- Initialize Chat History and State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": "Salam! 👋 Saya Bot Soal Jawab BM. Anda boleh tanya saya soalan berkaitan polisi e-dagang (contoh: Lazada/Shopee) dari pangkalan data saya."}
    ]
# Add other states if needed, e.g., st.session_state.mode = "qa"

# --- Display Chat History ---
st.title("🇲🇾 Bot Soal Jawab Bahasa Melayu (E-Dagang)")
st.caption("Dibangunkan dengan G-v5.6-Go | Streamlit | LangChain | Hugging Face")
st.divider() # Add a visual separator

# Loop through messages stored in session state and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"]) # Use markdown to render text

# --- Handle User Input ---
if prompt := st.chat_input("Masukkan soalan anda di sini..."):
    # 1. Add user message to history and display it
    st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # 2. Generate and display assistant response
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        # Check if RAG chain is ready
        if not qa_chain:
            st.error("Maaf, sistem RAG tidak bersedia. Sila pastikan FAISS index dimuatkan dengan betul.")
        else:
            # Use a spinner while processing
            with st.spinner("Mencari jawapan..."):
                try:
                    start_time = time.time()
                    # Run the RAG chain
                    result = qa_chain({"query": prompt})
                    end_time = time.time()

                    generated_answer = result.get('result', "Maaf, saya tidak dapat menjana jawapan.")
                    # Basic check for sentinel tokens
                    if "<extra_id_" in generated_answer:
                         generated_answer = "Maaf, saya tidak pasti jawapannya berdasarkan maklumat yang ada."

                    st.markdown(generated_answer) # Display the main answer

                    # Optionally display sources in the same message or a new one
                    source_docs = result.get('source_documents', [])
                    if source_docs:
                        with st.expander("Lihat Sumber Rujukan", expanded=False):
                            for i, doc in enumerate(source_docs):
                                source_name = doc.metadata.get('source', f'Sumber {i+1}')
                                st.info(f"**{source_name}:**\n\n```\n{doc.page_content}\n```")
                            st.caption(f"Masa mencari: {end_time - start_time:.2f} saat")
                    else:
                         st.warning("Tiada sumber rujukan ditemui.")

                except Exception as e:
                    st.error(f"Ralat semasa memproses RAG: {e}")

    # Add the complete assistant response (including sources) to session state *after* displaying
    assistant_response_content = generated_answer
    # You could format sources into the main message string if desired
    # assistant_response_content += "\n\n**Sumber:**\n..."
    st.session_state.messages.append({"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": assistant_response_content})

    # Optional: Scroll to bottom (experimental, might not work perfectly)
    # st.experimental_rerun() # Rerun to potentially show latest message at bottom
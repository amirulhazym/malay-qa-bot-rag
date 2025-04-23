# --- app_final.py (Full Code - Corrected Indentation) ---
import streamlit as st
import time
import torch
import random
import os
import re
import logging
from typing import Dict, Any, List, Optional

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LangChain Component Imports ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
         from langchain_huggingface import HuggingFacePipeline as HFPipelineCommunity
    except ImportError:
         from langchain_community.llms import HuggingFacePipeline as HFPipelineCommunity
    logger.info("Using langchain_huggingface for Embeddings (or community fallback).")
except ImportError:
    logger.warning("langchain-huggingface not found, trying older community paths...")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.llms import HuggingFacePipeline as HFPipelineCommunity
    except ImportError:
        logger.critical("!!! ERROR: Core LangChain embedding/LLM components not found.")
        st.error("Ralat kritikal: Pustaka LangChain yang diperlukan tidak dijumpai.")
        st.stop()

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
     logger.critical("!!! ERROR: Could not import FAISS from langchain_community.")
     st.error("Ralat kritikal: Komponen FAISS LangChain tidak dijumpai.")
     st.stop()

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- Page Config ---
st.set_page_config(page_title="Bantuan E-Dagang", page_icon="🛍️", layout="centered")

# --- Constants ---
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_CHECKPOINT = "google/mt5-small" # Sticking with mt5-small for fallback predictability
ASSISTANT_AVATAR_URL = "https://cdn-icons-png.flaticon.com/512/6134/6134346.png"
USER_AVATAR = "👤"
CACHE_DIR_ST = os.path.join(os.getcwd(), ".cache_st")
os.makedirs(CACHE_DIR_ST, exist_ok=True)
SEARCH_TYPE = "similarity" # Use the best one found in debugging (similarity/mmr)
SEARCH_K = 3 # Retrieve top 3
SEARCH_FETCH_K = 10 # Only if SEARCH_TYPE="mmr"

# Predefined Suggestions
SUGGESTIONS = {
    "pemulangan": ["Apakah Status Pemulangan?", "Boleh pulangkan sebab tukar fikiran?", "Berapa lama proses bayaran balik?", "Perlu hantar balik barang?"],
    "pembayaran": ["Cara bayar guna ShopeePay/Lazada Wallet?", "Ada pilihan ansuran?", "Kenapa pembayaran gagal?", "Bagaimana guna baucar?"],
    "penghantaran": ["Bagaimana jejak pesanan saya?", "Berapa lama tempoh penghantaran?", "Boleh tukar alamat lepas pesan?", "Apa jadi jika barang hilang masa hantar?"],
    "pembatalan": ["Boleh batal jika sudah bayar?", "Bagaimana dapat refund lepas batal?", "Kenapa tidak boleh batal pesanan?"],
    "umum": ["Cara hubungi Customer Service?", "Promosi terkini apa?", "Adakah produk LazMall original?", "Isu log masuk akaun"]
}
DEFAULT_SUGGESTIONS = SUGGESTIONS["umum"]

# --- Helper Functions ---
def clean_llm_output(text: Optional[str]) -> str:
    """Removes common unwanted tokens and excessive whitespace. Returns empty if invalid."""
    if not isinstance(text, str): return ""
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<pad>', '', text)
    text = re.sub(r'^[ .,;:!?]+$', '', text.strip())
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    if not cleaned_text or all(c in ' .,;:!?()[]{}<>/\\"\'`~#@$%^&*-_=+|\t\n' for c in cleaned_text):
         logger.warning("Cleaned LLM output was empty or trivial.")
         return ""
    return cleaned_text

def get_suggestions(last_assistant_message: Optional[str]) -> List[str]:
    """Generates contextual suggestions based on keywords."""
    # (Keep function code as before)
    if not isinstance(last_assistant_message, str): return random.sample(DEFAULT_SUGGESTIONS, min(len(DEFAULT_SUGGESTIONS), 3))
    last_assistant_message_lower = last_assistant_message.lower()
    matched_keys = []
    if any(k in last_assistant_message_lower for k in ["pulang", "refund", "pemulangan", "balik"]): matched_keys.extend(SUGGESTIONS["pemulangan"])
    if any(k in last_assistant_message_lower for k in ["bayar", "payment", "pembayaran", "ansuran", "baucar"]): matched_keys.extend(SUGGESTIONS["pembayaran"])
    if any(k in last_assistant_message_lower for k in ["hantar", "shipping", "penghantaran", "kurier", "jejak", "alamat"]): matched_keys.extend(SUGGESTIONS["penghantaran"])
    if any(k in last_assistant_message_lower for k in ["batal", "cancel", "pembatalan"]): matched_keys.extend(SUGGESTIONS["pembatalan"])
    if len(matched_keys) < 3: matched_keys.extend(DEFAULT_SUGGESTIONS)
    unique_suggestions = list(dict.fromkeys(matched_keys))
    return random.sample(unique_suggestions, min(len(unique_suggestions), 3))

def add_message(role: str, content: str, avatar: Optional[str] = None, suggestions: Optional[List[str]] = None):
    """Adds a message to the session state history and updates suggestion tracking."""
    message_id = len(st.session_state.get("messages", []))
    msg = {"role": role, "content": content, "id": message_id}
    if avatar: msg["avatar"] = avatar
    if suggestions:
        msg["suggestions"] = suggestions
        # Only assistant messages with suggestions should update the tracker
        if role == "assistant":
            st.session_state.last_assistant_message_id_with_suggestions = message_id
        else:
            # User message shouldn't have suggestions tied to it directly
             # Invalidate any previous assistant suggestions when user speaks
            st.session_state.last_assistant_message_id_with_suggestions = -1

    # Add the main message
    st.session_state.messages.append(msg)
    logger.debug(f"Added message ID {message_id}: Role={role}, Suggestions Provided={suggestions is not None}")


# --- Cached Loading of RAG Pipeline ---
@st.cache_resource(show_spinner="Memuatkan komponen AI... 🧠")
def load_rag_pipeline(embed_model_name: str, llm_checkpoint: str, index_path: str) -> Optional[RetrievalQA]:
    """Loads embeddings, FAISS index, LLM pipeline, and creates the RAG QA chain."""
    # (Keep function code exactly as before, ensuring all internal logging and error checks are present)
    logger.info("--- Attempting to load RAG Pipeline ---")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        # 1. Load Embeddings
        logger.info(f"Loading embedding model: {embed_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_name, model_kwargs={'device': device}, cache_folder=CACHE_DIR_ST)
        logger.info("Embedding model ready.")
        # 2. Load FAISS Index
        logger.info(f"Loading FAISS index from: {index_path}")
        if not os.path.exists(index_path):
            logger.error(f"FAISS index not found at specified path: {index_path}")
            st.error(f"Ralat Kritikal: Fail index FAISS ('{index_path}') tidak dijumpai. Sila jalankan `reindex.py`.")
            return None
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"FAISS index ready ({vector_store.index.ntotal} vectors).")
        # 3. Load LLM Pipeline
        logger.info(f"Loading LLM pipeline: {llm_checkpoint}")
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint, legacy=False)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_checkpoint)
        pipeline_device = 0 if device == 'cuda' else -1
        pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=100, temperature=0.6, device=pipeline_device)
        llm_pipe = HFPipelineCommunity(pipeline=pipe) # Still using community pending upgrade
        logger.info(f"LLM pipeline ready on {'CPU' if pipeline_device==-1 else 'GPU'}.")
        # 4. Define Prompt Template
        prompt_template_text = """Gunakan Konteks yang diberi SAHAJA untuk menjawab Soalan berikut. Jangan tambah maklumat luar. Jika jawapan tiada dalam Konteks, sila nyatakan "Maaf, maklumat tentang itu tiada dalam pangkalan data saya.". Jawab dalam Bahasa Melayu sepenuhnya.\n\nKonteks:\n{context}\n\nSoalan: {question}\nJawapan:"""
        PROMPT = PromptTemplate(template=prompt_template_text, input_variables=["context", "question"])
        logger.info("Prompt template defined.")
        # 5. Create QA Chain
        logger.info(f"Creating retriever (Type: {SEARCH_TYPE}, k: {SEARCH_K})...")
        retriever = vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={'k': SEARCH_K} if SEARCH_TYPE == "similarity" else {'k': SEARCH_K, 'fetch_k': SEARCH_FETCH_K})
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(llm=llm_pipe, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
        logger.info("--- RAG Pipeline Ready ---")
        return qa_chain
    except Exception as e:
        logger.critical(f"FATAL ERROR loading RAG pipeline: {e}", exc_info=True)
        st.error(f"Ralat kritikal semasa memuatkan komponen AI: {e}")
        return None


# --- Load RAG Chain ---
qa_chain = load_rag_pipeline(EMBEDDING_MODEL_NAME, LLM_CHECKPOINT, INDEX_SAVE_PATH)

# --- Inject Custom CSS ---
st.markdown("""
<style>
    /* --- Base & Layout --- */
    .stApp { background-color: #f0f2f5; }
    .main .block-container { max-width: 600px; margin: auto; padding: 1rem 1rem 6rem 1rem; box-sizing: border-box; background-color: #ffffff; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); min-height: calc(100vh - 40px); display: flex; flex-direction: column; }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] {
        flex-grow: 1; /* Allow message container to grow */
        overflow-y: auto; /* Enable scroll */
        max-height: calc(100vh - 150px); /* Approximate height calculation minus header/input */
        padding-right: 10px;
    }

    /* --- Header --- */
    .chat-header {
        background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%); /* Soft Blue to Darker Blue */
        color: white; padding: 12px 18px; border-radius: 8px 8px 0 0; display: flex; align-items: center; margin: -1rem -1rem 1rem -1rem; /* Adjusted margins */ position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-header img.avatar { width: 36px; height: 36px; border-radius: 50%; margin-right: 12px; }
    .chat-header .title { font-weight: 600; font-size: 1.05em; margin-bottom: 1px; }
    .chat-header .subtitle { font-size: 0.8em; opacity: 0.9; }

    /* --- Chat Messages --- */
    div[data-testid="stChatMessage"] { padding: 10px 14px; border-radius: 18px; margin-bottom: 8px; width: fit-content; max-width: 85%; line-height: 1.5; border: 1px solid #E5E7EB; box-shadow: 0 1px 1px rgba(0,0,0,0.04); }
    div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) { background-color: #F9FAFB; color: #374151; margin-right: auto; }
    div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) { background-color: #3B82F6; color: white; margin-left: auto; margin-right: 0; border: none; }
    div[data-testid="stChatMessage"] p { margin-bottom: 0.3rem; }

    /* --- Suggestion Buttons --- */
    .suggestion-container { padding-top: 5px; padding-left: 40px; display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .suggestion-container .stButton>button { background-color: #EFF6FF; color: #3B82F6; border: 1px solid #BFDBFE; border-radius: 16px; padding: 5px 12px; font-size: 0.85em; font-weight: 500; cursor: pointer; transition: all 0.2s ease; }
    .suggestion-container .stButton>button:hover { background-color: #DBEAFE; border-color: #93C5FD; }

    /* --- Chat Input --- */
    div[data-testid="stChatInput"] { background-color: #f0f2f5; border-top: 1px solid #E5E7EB; padding: 0.75rem 1rem; position: fixed; bottom: 0; left: 0; right: 0; margin: auto; max-width: 600px; width: 100%; box-sizing: border-box; z-index: 100; } /* Matched max-width */
    div[data-testid="stChatInput"] textarea { border-radius: 18px; border: 1px solid #D1D5DB; background-color: #fff; }
    div[data-testid="stChatInput"] button { background-color: #2563EB; svg {fill: white;} }
    div[data-testid="stChatInput"] button:hover { background-color: #1D4ED8; }

    /* --- Source Box Styling --- */
    .source-box { background-color: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; font-size: 0.9rem; }
    .source-box strong { display: block; margin-bottom: 5px; color: #374151; }
    .source-box pre { white-space: pre-wrap; word-wrap: break-word; font-size: 0.85em; background-color: #e9ecef; padding: 5px; border-radius: 4px;}

    /* --- Hide Streamlit UI Elements --- */
    header[data-testid="stHeader"], footer, #MainMenu, .stDeployButton { display: none !important; visibility: hidden !important; }
    .main .block-container { padding-top: 80px !important; } /* INCREASED padding for sticky header */

</style>
""", unsafe_allow_html=True)


# --- Custom Header ---
st.markdown(f"""
<div class="chat-header">
    <img class="avatar" src="{ASSISTANT_AVATAR_URL}" alt="Bot Avatar">
    <div>
        <div class="title">Bot Bantuan E-Dagang</div>
        <div class="subtitle">Sedia membantu anda ⚡</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Initialize Chat History & State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_assistant_message_id_with_suggestions" not in st.session_state:
    st.session_state.last_assistant_message_id_with_suggestions = -1
if "button_states" not in st.session_state:
    st.session_state.button_states = {} # Tracks {msg_id: True/False}
if "processing_user_input" not in st.session_state:
    st.session_state.processing_user_input = None

# --- Add initial assistant message if history is empty ---
if not st.session_state.messages:
    initial_suggestions = random.sample(DEFAULT_SUGGESTIONS, 3)
    initial_msg_id = 0 # ID for the first message
    st.session_state.messages.append({
        "role": "assistant", "avatar": ASSISTANT_AVATAR_URL,
        "content": "Salam! 👋 Ada apa yang boleh saya bantu? Sila tanya soalan atau pilih topik.",
        "id": initial_msg_id, "suggestions": initial_suggestions
    })
    st.session_state.last_assistant_message_id_with_suggestions = initial_msg_id
    st.session_state.button_states[initial_msg_id] = False # Ensure initial state is not used

# --- Display Chat History ---
# Outer container for messages might help layout
message_area = st.container()
with message_area:
    for message in st.session_state.messages:
        msg_id = message["id"]
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
            # Display suggestions below the relevant assistant message if needed
            if (message["role"] == "assistant" and
                "suggestions" in message and
                msg_id == st.session_state.last_assistant_message_id_with_suggestions and
                not st.session_state.button_states.get(msg_id, False)):

                st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
                suggestions_to_show = message["suggestions"][:3] # Show up to 3 suggestions
                cols = st.columns(len(suggestions_to_show))
                for j, label in enumerate(suggestions_to_show):
                    button_key = f"button_{msg_id}_{j}"
                    if cols[j].button(label, key=button_key):
                        logger.info(f"Button '{label}' (msg {msg_id}) clicked.")
                        # Mark buttons used for this message ID
                        st.session_state.button_states[msg_id] = True
                        # Append user action
                        st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": label, "id": len(st.session_state.messages)})
                        # Set flag to process
                        st.session_state.processing_user_input = label
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


# --- Handle User Text Input ---
prompt = st.chat_input("Taip soalan anda di sini...", key="chat_input")
if prompt:
    logger.info(f"Received text input: '{prompt}'")
    st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": prompt, "id": len(st.session_state.messages)})
    # Reset suggestion display trigger
    st.session_state.last_assistant_message_id_with_suggestions = -1
    st.session_state.button_states = {k: True for k in st.session_state.button_states} # Mark all old buttons used
    st.session_state.processing_user_input = prompt
    st.rerun()

# --- Generate and Display Assistant Response ---
if st.session_state.get("processing_user_input"):

    user_input_to_process = st.session_state.processing_user_input
    # --- Clear flag ---
    st.session_state.processing_user_input = None
    logger.info(f"Processing input: '{user_input_to_process}'")

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR_URL):
        response_placeholder = st.empty()
        response_placeholder.markdown("...") # Thinking indicator

        final_assistant_content = "Maaf, ralat berlaku."
        source_docs = []
        new_suggestions = random.sample(DEFAULT_SUGGESTIONS, 3) # Default suggestions
        processing_time = 0

        if not qa_chain:
            final_assistant_content = "Maaf, sistem QA tidak aktif."
            st.error(final_assistant_content)
        else:
            try:
                start_time = time.time()
                logger.info("Invoking RAG chain...")
                # Ensure using .invoke() here
                result = qa_chain.invoke({"query": user_input_to_process})
                end_time = time.time()
                processing_time = end_time - start_time

                generated_answer_raw = result.get('result', "")
                source_docs = result.get('source_documents', [])
                logger.info(f"Raw LLM output snippet: {generated_answer_raw[:100]}")
                logger.info(f"Retrieved {len(source_docs)} sources.")

                cleaned_answer = clean_llm_output(generated_answer_raw)
                generation_failed = not cleaned_answer or cleaned_answer.startswith("Maaf,")

                if generation_failed and source_docs:
                    fallback_texts = []
                    # Fallback displays max 2 sources now
                    for i, doc in enumerate(source_docs[:2]):
                         clean_source = re.sub(r'\s+', ' ', doc.page_content).strip()
                         if len(clean_source) > 500: clean_source = clean_source[:500] + "..."
                         fallback_texts.append(f"**Sumber {i+1} ({os.path.basename(doc.metadata.get('source', 'N/A'))})**: _{clean_source}_")
                    final_assistant_content = "Jawapan tepat tidak jelas, tetapi berikut maklumat berkaitan dari pangkalan data:\n\n---\n" + "\n\n---\n".join(fallback_texts)
                    logger.warning("LLM generation failed/weak; displaying fallback from source(s).")

                elif generation_failed:
                    final_assistant_content = "Maaf, tiada maklumat relevan dijumpai."
                    logger.warning("LLM generation failed/weak, and no relevant sources found.")
                else:
                    final_assistant_content = cleaned_answer
                    logger.info("LLM generated valid response.")

                new_suggestions = get_suggestions(final_assistant_content)

            except Exception as e:
                logger.error(f"Error during RAG chain execution: {str(e)}", exc_info=True)
                final_assistant_content = "Maaf, ralat teknikal semasa memproses."
                source_docs = [] # Reset sources on error

        # --- Display Final Response & Sources ---
        # Use the placeholder to overwrite the "..." with the final content
        with response_placeholder.container():
             st.markdown(final_assistant_content)
             if source_docs: # Show sources even if fallback was used
                  with st.expander("Lihat Sumber Rujukan", expanded=False):
                       for k, doc in enumerate(source_docs):
                           source_name = os.path.basename(doc.metadata.get('source', f'Dokumen {k+1}'))
                           st.markdown(f"""<div class="source-box"><strong>{source_name}</strong><pre>{doc.page_content}</pre></div>""", unsafe_allow_html=True)
                       if processing_time > 0:
                           st.caption(f"Masa diambil: {processing_time:.2f} saat")

    # --- Append final message AFTER displaying ---
    add_message("assistant", final_assistant_content, ASSISTANT_AVATAR_URL, new_suggestions)

    # --- Rerun to update the message list with the assistant's response + new suggestions ---
    st.rerun()

# --- Sidebar Content ---
with st.sidebar:
    st.title("ℹ️ Info Bot")
    st.markdown("**Bot QA E-dagang BM**")
    st.image(ASSISTANT_AVATAR_URL, width=80) # Using the constant defined
    st.markdown("Bot ini menjawab soalan polisi berdasarkan pangkalan data yang disediakan.")
    st.markdown("---")
    st.markdown("#### ⚙️ Teknologi")
    # Use f-strings to include constants dynamically
    st.markdown(f"""
    - **Arsitektur**: RAG (LangChain)
    - **Embeddings**: `{os.path.basename(EMBEDDING_MODEL_NAME)}`
    - **Vector Store**: FAISS (Lokal)
    - **LLM**: `{os.path.basename(LLM_CHECKPOINT)}`
    - **UI**: Streamlit
    """)
    st.markdown("---")
    # Ensure this caption is correctly indented within the sidebar block
    st.caption("Pastikan fail index FAISS wujud di root direktori.")

# --- Footer ---
st.markdown("""
<div class="footer">
    Project 3: Malay QA Bot with RAG | © Amirulhazym 2025
</div>
""", unsafe_allow_html=True)
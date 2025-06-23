# --- app_final_fixed_v2.py (Responsive Input, Theme Variables, Toggle Info) ---
import streamlit as st
import time
import torch
import random
import os
import re
import logging
from typing import List, Optional

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LangChain Component Imports ---
# (Keep imports as before)
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
# (Keep constants as before)
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_CHECKPOINT = "google/flan-t5-small"
ASSISTANT_AVATAR_URL = "https://cdn-icons-png.flaticon.com/512/6134/6134346.png"
USER_AVATAR = "👤"
CACHE_DIR_ST = os.path.join(os.getcwd(), ".cache_st")
os.makedirs(CACHE_DIR_ST, exist_ok=True)
SEARCH_TYPE = "mmr"
SEARCH_K = 3
SEARCH_FETCH_K = 10

SUGGESTIONS = {
    "pemulangan": ["Apakah Status Pemulangan?", "Boleh pulangkan sebab tukar fikiran?", "Berapa lama proses bayaran balik?", "Perlu hantar balik barang?"],
    "pembayaran": ["Cara bayar guna ShopeePay/Lazada Wallet?", "Ada pilihan ansuran?", "Kenapa pembayaran gagal?", "Bagaimana guna baucar?"],
    "penghantaran": ["Bagaimana jejak pesanan saya?", "Berapa lama tempoh penghantaran?", "Boleh tukar alamat lepas pesan?", "Apa jadi jika barang hilang masa hantar?"],
    "pembatalan": ["Boleh batal jika sudah bayar?", "Bagaimana dapat refund lepas batal?", "Kenapa tidak boleh batal pesanan?"],
    "umum": ["Cara hubungi Customer Service?", "Promosi terkini apa?", "Adakah produk LazMall original?", "Isu log masuk akaun"]
}
DEFAULT_SUGGESTIONS = SUGGESTIONS["umum"]

# --- Helper Functions ---
# (Keep helper functions clean_llm_output, generate_contextual_suggestions, add_message as before)
def clean_llm_output(text: Optional[str]) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<pad>', '', text)
    text = re.sub(r'^[ .,;:!?]+$', '', text.strip())
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    if not cleaned_text or all(c in ' .,;:!?()[]{}<>/\\"\'`~#@$%^&*-_=+|\t\n' for c in cleaned_text):
         logger.warning("Cleaned LLM output was empty or trivial.")
         return ""
    return cleaned_text

def generate_contextual_suggestions(last_assistant_message: Optional[str]) -> List[str]:
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
    """Adds a message to the session state history and updates suggestion tracking if needed."""
    if "messages" not in st.session_state: st.session_state.messages = []
    message_id = len(st.session_state.messages)
    msg = {"role": role, "content": content, "id": message_id}
    if avatar: msg["avatar"] = avatar
    if suggestions:
        msg["suggestions"] = suggestions
        if role == "assistant":
            st.session_state.last_assistant_message_id_with_suggestions = message_id
            if "button_states" not in st.session_state: st.session_state.button_states = {}
            st.session_state.button_states[message_id] = False
            logger.debug(f"Adding assistant message ID {message_id} with suggestions.")
        else:
             logger.debug(f"Adding user message ID {message_id}. Suggestions passed but not stored directly.")
             st.session_state.last_assistant_message_id_with_suggestions = -1
    st.session_state.messages.append(msg)
    logger.debug(f"Message list length now: {len(st.session_state.messages)}")
    return message_id

# --- Cached Loading of RAG Pipeline ---
# (Keep load_rag_pipeline function exactly as before)
@st.cache_resource(show_spinner="Memuatkan komponen AI... 🧠")
def load_rag_pipeline(embed_model_name: str, llm_checkpoint: str, index_path: str) -> Optional[RetrievalQA]:
    logger.info("--- Attempting to load RAG Pipeline ---")
    # ... [Rest of function unchanged] ...
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'; logger.info(f"Using device: {device}")
        logger.info(f"Loading embedding model: {embed_model_name}"); embeddings = HuggingFaceEmbeddings(model_name=embed_model_name, model_kwargs={'device': device}, cache_folder=CACHE_DIR_ST); logger.info("Embedding model ready.")
        logger.info(f"Loading FAISS index from: {index_path}");
        if not os.path.exists(index_path): logger.error(f"FAISS index not found at specified path: {index_path}"); st.error(f"Ralat Kritikal: Fail index FAISS ('{index_path}') tidak dijumpai. Sila jalankan `reindex.py`."); return None
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True); logger.info(f"FAISS index ready ({vector_store.index.ntotal} vectors).")
        logger.info(f"Loading LLM pipeline: {llm_checkpoint}"); llm_tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint, legacy=False); llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_checkpoint); pipeline_device = 0 if device == 'cuda' else -1
        pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=100, temperature=0.6, device=pipeline_device); llm_pipe = HFPipelineCommunity(pipeline=pipe); logger.info(f"LLM pipeline ready on {'CPU' if pipeline_device==-1 else 'GPU'}.")
        prompt_template_text = """Gunakan Konteks yang diberi SAHAJA untuk menjawab Soalan berikut. Jangan tambah maklumat luar. Jika jawapan tiada dalam Konteks, sila nyatakan "Maaf, maklumat tentang itu tiada dalam pangkalan data saya.". Jawab dalam Bahasa Melayu sepenuhnya.\n\nKonteks:\n{context}\n\nSoalan: {question}\nJawapan:"""
        PROMPT = PromptTemplate(template=prompt_template_text, input_variables=["context", "question"]); logger.info("Prompt template defined.")
        logger.info(f"Creating retriever (Type: {SEARCH_TYPE}, k: {SEARCH_K})...")
        retriever = vector_store.as_retriever(search_type=SEARCH_TYPE, search_kwargs={'k': SEARCH_K} if SEARCH_TYPE == "" else {'k': SEARCH_K, 'fetch_k': SEARCH_FETCH_K})
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(llm=llm_pipe, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
        logger.info("--- RAG Pipeline Ready ---")
        return qa_chain
    except Exception as e: logger.critical(f"FATAL ERROR loading RAG pipeline: {e}", exc_info=True); st.error(f"Ralat kritikal semasa memuatkan komponen AI: {e}"); return None

# --- Load RAG Chain ---
qa_chain = load_rag_pipeline(EMBEDDING_MODEL_NAME, LLM_CHECKPOINT, INDEX_SAVE_PATH)

# --- Inject Custom CSS (MODIFIED FOR THEME VARIABLES & LAYOUT) ---
# *** THIS IS THE MAINLY MODIFIED SECTION ***
st.markdown("""
<style>
    /* --- Base & Layout --- */
    /* Apply theme variable to overall app background */
    .stApp {
        background-color: var(--background-color);
    }
    /* Main chat container area */
    .main .block-container {
        max-width: 700px; /* Slightly wider chat area */
        margin: auto;
        /* Reduced top padding, increased bottom significantly */
        padding: 0.5rem 1rem 8rem 1rem; /* Less top, MORE bottom */
        box-sizing: border-box;
        /* Use theme variable for chat area background */
        background-color: var(--secondary-background-color);
        color: var(--text-color); /* Use theme text color */
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        min-height: calc(100vh - 20px);
        display: flex;
        flex-direction: column;
    }
    /* Chat message display area - Allow scrolling */
    .message-scroll-area {
    flex-grow: 1; overflow-y: auto;
    padding: 1rem;
    background-color: #f8fafc; /* FORCE very light grey */
    /* INCREASE this padding to make more space above input */
    padding-bottom: 100px; /* Example: Increased from 80px */
    box-sizing: border-box;
    }

    /* --- Header --- */
    .chat-header {
        background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%); /* Keep Gradient */
        color: white;
        padding: 12px 18px; border-radius: 8px; /* Rounded all corners */
        display: flex; align-items: center;
        margin-bottom: 1rem; /* Space below header */
        /* Removed negative margins and sticky positioning */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* (Other header styles remain the same) */
    .chat-header img.avatar { width: 36px; height: 36px; border-radius: 50%; margin-right: 12px; }
    .chat-header .title { font-weight: 600; font-size: 1.05em; margin-bottom: 1px; }
    .chat-header .subtitle { font-size: 0.8em; opacity: 0.9; }

    /* --- Chat Messages (Use Theme Variables) --- */
    div[data-testid="stChatMessage"] {
        padding: 10px 14px; border-radius: 18px; margin-bottom: 8px;
        width: fit-content; max-width: 85%; line-height: 1.5;
        border: 1px solid var(--gray-300); /* Theme border */
        box-shadow: 0 1px 1px rgba(0,0,0,0.04);
    }
    /* Assistant */
    div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
        background-color: var(--secondary-background-color); /* Theme bg */
        color: var(--text-color); /* Theme text */
        margin-right: auto;
    }
    /* User */
    div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
        background-color: var(--primary-color); /* Theme primary */
        color: white; /* Assume white works on primary */
        margin-left: auto; margin-right: 0; border: none;
    }
    div[data-testid="stChatMessage"] p { margin-bottom: 0.3rem; color: inherit; }

    /* --- Suggestion Buttons (Use Theme Variables) --- */
    .suggestion-container { padding-top: 5px; padding-left: 40px; display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .suggestion-container .stButton>button {
        background-color: var(--secondary-background-color);
        color: var(--primary-color); border: 1px solid var(--primary-color); opacity: 0.8;
        border-radius: 16px; padding: 5px 12px; font-size: 0.85em; font-weight: 500;
        cursor: pointer; transition: all 0.2s ease;
    }
    .suggestion-container .stButton>button:hover {
         opacity: 1.0; background-color: color-mix(in srgb, var(--primary-color) 15%, transparent);
         border-color: var(--primary-color);
    }

    /* --- Chat Input (Let it flow at bottom, Theme variables) --- */
    /* REMOVE fixed positioning styles */
    div[data-testid="stChatInput"] {
         background-color: var(--secondary-background-color); /* Theme bg */
         border-top: 1px solid var(--gray-300); /* Theme border */
         padding: 0.75rem 1rem;
         /* Remove fixed, bottom, left, right, margin auto, max-width, width, z-index */
    }
    div[data-testid="stChatInput"] textarea {
         border-radius: 18px; border: 1px solid var(--gray-400);
         background-color: var(--background-color); /* Theme main background */
         color: var(--text-color); /* Theme text color */
    }
    div[data-testid="stChatInput"] button {
         background-color: var(--primary-color); svg {fill: white;} /* Use primary for send */
    }
    div[data-testid="stChatInput"] button:hover { background-color: color-mix(in srgb, var(--primary-color) 85%, black); }


    /* --- Source Box Styling (Use Theme Variables) --- */
    .source-box {
        background-color: var(--secondary-background-color); border: 1px solid var(--gray-300);
        border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; font-size: 0.9rem;
    }
    .source-box strong { display: block; margin-bottom: 5px; color: var(--text-color); }
    .source-box pre {
         white-space: pre-wrap; word-wrap: break-word; font-size: 0.85em;
         background-color: color-mix(in srgb, var(--secondary-background-color) 90%, black);
         padding: 5px; border-radius: 4px; color: var(--text-color);
    }

    /* --- Hide Streamlit UI Elements --- */
    header[data-testid="stHeader"], footer, #MainMenu, .stDeployButton { display: none !important; visibility: hidden !important; }

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
# (Keep state management logic as before)
if "messages" not in st.session_state: st.session_state.messages = []
if "last_assistant_message_id_with_suggestions" not in st.session_state: st.session_state.last_assistant_message_id_with_suggestions = -1
if "button_states" not in st.session_state: st.session_state.button_states = {}
if "processing_user_input" not in st.session_state: st.session_state.processing_user_input = None

# --- Add initial assistant message ---
# (Keep initial message logic as before)
if not st.session_state.messages:
     initial_suggestions = random.sample(DEFAULT_SUGGESTIONS, 3)
     initial_msg_id = add_message("assistant", "Salam! 👋 Ada apa yang boleh saya bantu? Sila tanya soalan atau pilih topik.", ASSISTANT_AVATAR_URL, initial_suggestions)
     st.session_state.button_states[initial_msg_id] = False

# --- Display Chat History ---
# (Keep history display logic as before)
message_area = st.container()
with message_area:
    for message in st.session_state.messages:
        msg_id = message["id"]
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
            # Display suggestions
            if (message["role"] == "assistant" and
                "suggestions" in message and
                msg_id == st.session_state.last_assistant_message_id_with_suggestions and
                not st.session_state.button_states.get(msg_id, False)):

                st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
                suggestions_to_show = message["suggestions"][:3]
                cols = st.columns(len(suggestions_to_show))
                for j, label in enumerate(suggestions_to_show):
                    button_key = f"button_{msg_id}_{j}"
                    if cols[j].button(label, key=button_key):
                        logger.info(f"Button '{label}' (msg {msg_id}) clicked.")
                        st.session_state.button_states[msg_id] = True
                        add_message("user", label, USER_AVATAR)
                        st.session_state.processing_user_input = label
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


# --- Handle User Text Input ---
# (Keep text input logic as before)
prompt = st.chat_input("Taip soalan anda di sini...", key="chat_input")
if prompt:
    logger.info(f"Received text input: '{prompt}'")
    add_message("user", prompt, USER_AVATAR)
    st.session_state.button_states = {k: True for k in st.session_state.button_states}
    st.session_state.last_assistant_message_id_with_suggestions = -1
    st.session_state.processing_user_input = prompt
    st.rerun()

# --- Generate and Display Assistant Response Logic ---
# (Keep response generation logic, including fallback, exactly as before)
# --- Generate and Display Assistant Response Logic ---
if st.session_state.get("processing_user_input"):

    user_input_to_process = st.session_state.processing_user_input
    st.session_state.processing_user_input = None # Clear flag immediately
    logger.info(f"Processing input: '{user_input_to_process}'")

    # Generate suggestions based on the USER'S input first
    new_suggestions = generate_contextual_suggestions(user_input_to_process)

    # Display thinking state using chat_message context
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR_URL):
        response_placeholder = st.empty()
        response_placeholder.markdown("...") # Thinking indicator

        final_assistant_content = "Maaf, ralat memproses." # Default
        source_docs = []
        processing_time = 0

        if not qa_chain:
            final_assistant_content = "Maaf, sistem QA tidak aktif."
            logger.error("QA Chain not available.")
            st.error(final_assistant_content) # Display error directly
        else:
            try:
                start_time = time.time()
                logger.info("Invoking RAG chain...")
                result = qa_chain.invoke({"query": user_input_to_process})
                end_time = time.time()
                processing_time = end_time - start_time

                generated_answer_raw = result.get('result', "")
                source_docs = result.get('source_documents', []) # Get sources regardless of answer quality
                logger.info(f"Raw LLM output snippet: {generated_answer_raw[:100]}")
                logger.info(f"Retrieved {len(source_docs)} source documents.")

                # --- Strict Check for Generation Failure ---
                cleaned_answer = clean_llm_output(generated_answer_raw) # Clean first

                generation_failed = False # Assume success initially
                if not cleaned_answer:
                    generation_failed = True
                    logger.warning("Generation failed: Cleaned answer is empty.")
                elif cleaned_answer.startswith("Maaf,"):
                     generation_failed = True
                     logger.warning("Generation failed: Output starts with 'Maaf,'.")
                elif len(cleaned_answer) < (len(user_input_to_process) + 5) and cleaned_answer.lower() in user_input_to_process.lower():
                     # Check if it's basically just echoing the input
                     generation_failed = True
                     logger.warning("Generation failed: Output is likely an echo of the input.")
                elif cleaned_answer.lower().startswith("konteks yang diberi") or cleaned_answer.lower().startswith("gunakan konteks"):
                     # Check if it's repeating the prompt
                     generation_failed = True
                     logger.warning("Generation failed: Output repeats prompt instructions.")
                # Add any other specific failure patterns you observe

                # --- Determine Final Content ---
                if generation_failed and source_docs:
                    # FAILURE + Sources Found => Use Fallback
                    fallback_texts = []
                    for i, doc in enumerate(source_docs[:1]): # Limit fallback display
                         clean_source = re.sub(r'\s+', ' ', doc.page_content).strip()
                         if len(clean_source) > 600: clean_source = clean_source[:600] + "..."
                         fallback_texts.append(f"*{clean_source}*")
                    final_assistant_content = f"Berikut adalah maklumat berkaitan yang ditemui:\n\n---\n" + "\n\n---\n".join(fallback_texts)
                    logger.info("Displaying fallback from source(s).")

                elif generation_failed: # FAILURE + No Sources Found
                    final_assistant_content = "Maaf, tiada maklumat relevan dijumpai untuk menjawab soalan itu."
                    logger.warning("Generation failed and no relevant source docs retrieved.")
                else: # SUCCESS => Use Cleaned LLM Output
                    final_assistant_content = cleaned_answer
                    logger.info("Displaying cleaned LLM generated response.")

            except Exception as e:
                logger.error(f"Error during RAG chain execution: {str(e)}", exc_info=True)
                final_assistant_content = "Maaf, ralat teknikal semasa memproses."
                source_docs = [] # Ensure no sources shown on error

        # --- Display Final Response & Sources in UI ---
        with response_placeholder.container():
             st.markdown(final_assistant_content) # Display the final text (generated or fallback)
             if source_docs: # Show sources if they were retrieved, regardless of fallback state
                  with st.expander("Lihat Sumber Rujukan", expanded=False):
                       for k, doc in enumerate(source_docs):
                           source_name = os.path.basename(doc.metadata.get('source', f'Dokumen {k+1}'))
                           st.markdown(f"""<div class="source-box"><strong>{source_name}</strong><pre>{doc.page_content}</pre></div>""", unsafe_allow_html=True)
                       if processing_time > 0:
                           st.caption(f"Masa diambil: {processing_time:.2f} saat")

    # --- Append final message AFTER displaying & processing ---
    # This should now have the correct final_assistant_content
    add_message("assistant", final_assistant_content, ASSISTANT_AVATAR_URL, new_suggestions)

    # --- Rerun to update ---
    st.rerun()


# --- Sidebar Content (MODIFIED TO ADD THEME INFO) ---
with st.sidebar:
    st.title("ℹ️ Info Bot")
    st.markdown("**Bot QA E-dagang BM**")
    st.image(ASSISTANT_AVATAR_URL, width=80)
    st.markdown("Bot ini menjawab soalan polisi berdasarkan pangkalan data yang disediakan.")
    st.markdown("---")
    st.markdown("#### ⚙️ Teknologi")
    st.markdown(f"""
    - **Arsitektur**: RAG (LangChain)
    - **Embeddings**: `{os.path.basename(EMBEDDING_MODEL_NAME)}`
    - **Vector Store**: FAISS (Lokal)
    - **LLM**: `{os.path.basename(LLM_CHECKPOINT)}`
    - **UI**: Streamlit
    """)
    st.markdown("---")
    # --- MODIFICATION: Added Theme Info ---
    st.markdown("#### 🎨 Tetapan Tema")
    st.info("Aplikasi ini akan mengikut tema (Light/Dark) sistem atau pelayar web anda. Anda boleh menetapkannya secara manual dalam menu 'Settings' Streamlit (ikon gear atau '...' di penjuru atas kanan).", icon="💡")
    st.markdown("---")
    # --- END MODIFICATION ---
    st.caption("Pastikan fail index FAISS wujud.")


# --- Footer ---
# (Keep Footer block as before)
st.markdown("""
<div class="footer">
    Project 3: Malay QA Bot with RAG | © Amirulhazym 2025
</div>
""", unsafe_allow_html=True)
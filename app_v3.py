# --- app_v3.py (Modern UI/UX - Responsive - Shopee Flow Inspired) ---
import streamlit as st
import time
import torch
import random
# Use updated imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    # print("Using langchain_huggingface imports.") # Optional print
except ImportError:
    # print("WARNING: langchain-huggingface not found, falling back...")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.llms import HuggingFacePipeline
    except ImportError: print("!!! ERROR: Core LangChain components not found."); raise
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import re

# --- Page Config (MUST be the FIRST Streamlit command) ---
# Centered layout usually works well for chat on mobile/desktop
# Wide layout can also work if content inside is constrained
st.set_page_config(page_title="Bantuan E-Dagang", page_icon="🛍️", layout="centered")

# --- Constants ---
# Ensure these paths and names are correct for your setup
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_CHECKPOINT = "google/mt5-small"
ASSISTANT_AVATAR_URL = "https://cdn-icons-png.flaticon.com/512/6134/6134346.png" # Example Bot Avatar URL
USER_AVATAR = "👤" # Standard emoji
CACHE_DIR_ST = os.path.join(os.getcwd(), ".hf_cache_st")
os.makedirs(CACHE_DIR_ST, exist_ok=True)

# Predefined Suggestions (Refined examples)
SUGGESTIONS = {
    "pemulangan": ["Apakah Status Pemulangan'?", "Bagaimana jika barang rosak?", "Berapa lama proses bayaran balik?", "Perlu hantar balik barang?"],
    "pembayaran": ["Boleh guna ShopeePay?", "Bagaimana bayar ansuran?", "Ada caj tersembunyi?", "Kenapa pembayaran gagal?"],
    "penghantaran": ["Berapa lama tempoh penghantaran?", "Boleh tukar alamat?", "Bagaimana jejak pesanan saya?", "Kurier apa yang digunakan?"],
    "pembatalan": ["Boleh batal jika sudah bayar?", "Bagaimana dapat refund lepas batal?", "Kenapa butang batal tiada?"],
    "umum": ["Cara hubungi Khidmat Pelanggan?", "Promosi terkini?", "Adakah produk ini original?", "Maklumat lanjut tentang [Topik]?"] # Default suggestions
}
DEFAULT_SUGGESTIONS = SUGGESTIONS["umum"]

# --- Function to Clean LLM Output ---
def clean_llm_output(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<pad>', '', text)
    cleaned_text = text.strip()
    # If result is empty or only punctuation after cleaning, return a generic failure message
    if not cleaned_text or all(c in ' .,;:!?()[]{}<>/\\"\'`~#@$%^&*-_=+|\t\n' for c in cleaned_text):
         return "Maaf, saya tidak dapat memberikan jawapan yang jelas berdasarkan maklumat ini."
    return cleaned_text

# --- Function to Get Suggestions ---
def get_suggestions(last_assistant_message):
    if not isinstance(last_assistant_message, str): return DEFAULT_SUGGESTIONS[:3]
    last_assistant_message_lower = last_assistant_message.lower()
    matched_keys = []
    # Simple keyword matching (can be improved with NLP later)
    if any(k in last_assistant_message_lower for k in ["pulang", "refund", "pemulangan", "balik"]): matched_keys.extend(SUGGESTIONS["pemulangan"])
    if any(k in last_assistant_message_lower for k in ["bayar", "payment", "pembayaran", "ansuran"]): matched_keys.extend(SUGGESTIONS["pembayaran"])
    if any(k in last_assistant_message_lower for k in ["hantar", "shipping", "penghantaran", "kurier", "jejak"]): matched_keys.extend(SUGGESTIONS["penghantaran"])
    if any(k in last_assistant_message_lower for k in ["batal", "cancel", "pembatalan"]): matched_keys.extend(SUGGESTIONS["pembatalan"])

    if not matched_keys: matched_keys.extend(DEFAULT_SUGGESTIONS)
    unique_suggestions = list(dict.fromkeys(matched_keys)) # Remove duplicates
    # Try to return diverse suggestions, limit to 3-4
    return random.sample(unique_suggestions, min(len(unique_suggestions), 3))

# --- Cached Loading Functions ---
# These functions load heavy resources once and cache them
@st.cache_resource
def load_embeddings_model():
    # print(">> (Cache) Loading embedding model...") # Reduce console noise
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device},
            cache_folder=CACHE_DIR_ST
        )
        print(f">> Embedding model ready on {device}.")
        return embed_model
    except Exception as e: st.error(f"Ralat memuatkan model embedding: {e}"); st.stop()

@st.cache_resource
def load_faiss_index(_embeddings):
    # print(f">> (Cache) Loading FAISS index from: {INDEX_SAVE_PATH}...")
    if not _embeddings: st.error("Embeddings needed for FAISS."); return None
    if not os.path.exists(INDEX_SAVE_PATH): st.error(f"Index FAISS tidak dijumpai: '{INDEX_SAVE_PATH}'. Jalankan reindex.py."); return None
    try:
        vector_store = FAISS.load_local(INDEX_SAVE_PATH, _embeddings, allow_dangerous_deserialization=True)
        print(f">> FAISS index ready ({vector_store.index.ntotal} vectors).")
        return vector_store
    except Exception as e: st.error(f"Ralat memuatkan index FAISS: {e}"); return None

@st.cache_resource
def load_llm_qa_pipeline():
    # print(f">> (Cache) Loading LLM pipeline: {LLM_CHECKPOINT}...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=150, device=device)
        llm_pipe = HuggingFacePipeline(pipeline=pipe)
        print(f">> LLM pipeline ready on {'CPU' if device==-1 else 'GPU'}.")
        return llm_pipe
    except Exception as e: st.error(f"Ralat memuatkan LLM pipeline: {e}"); st.stop()

# --- Load Resources & Create Chain ---
# Use placeholders while loading
with st.spinner("Memuatkan model AI... 🧠"):
    embeddings_model = load_embeddings_model()
    vector_store = load_faiss_index(embeddings_model)
    llm_pipeline = load_llm_qa_pipeline()

# Define Custom Prompt
prompt_template_text = """Gunakan konteks berikut untuk menjawab soalan di akhir. Jawab hanya berdasarkan konteks yang diberikan. Jika jawapan tiada dalam konteks, nyatakan "Maaf, maklumat tiada dalam pangkalan data.". Jawab dalam Bahasa Melayu.

Konteks:
{context}

Soalan: {question}
Jawapan Membantu:"""
PROMPT = PromptTemplate(template=prompt_template_text, input_variables=["context", "question"])

# Create QA Chain
qa_chain = None
if vector_store and llm_pipeline and PROMPT and embeddings_model:
    try:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 10})
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(llm=llm_pipeline, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
        print(">> QA Chain ready.")
    except Exception as e: st.error(f"Ralat mencipta QA chain: {e}")
else:
    st.error("Komponen RAG tidak dapat dimuatkan. Sila semak console log.")
    # Consider st.stop() here if the chain is absolutely essential for app function

# --- Inject Custom CSS ---
st.markdown("""
<style>
    /* --- Base & Layout --- */
    .stApp { background-color: #f0f2f5; /* Light grey background */ }
    /* Center content vertically and horizontally */
    .main .block-container {
        max-width: 600px; /* Adjust max width for chat bubble feel */
        margin: auto;
        padding: 1rem 1rem 6rem 1rem; /* More bottom padding for fixed input */
        box-sizing: border-box;
        background-color: #ffffff; /* White background for chat area */
        border-radius: 10px; /* Rounded corners for chat area */
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Subtle shadow */
        min-height: calc(100vh - 40px); /* Try to fill height, leave space */
        display: flex;
        flex-direction: column;
    }
    /* Container for messages to allow scrolling */
     div.stChatMessage { display: flex; flex-direction: column; } /* Needed for msg bubbles */
     div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] {
        flex-grow: 1; /* Allows this container to fill space */
        overflow-y: auto; /* Enable vertical scroll */
        padding-right: 10px; /* Prevent scrollbar overlap */
     }

    /* --- Header --- */
    .chat-header {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); /* Shades of Blue */
        color: white;
        padding: 12px 18px;
        border-radius: 8px 8px 0 0; /* Match container top */
        display: flex;
        align-items: center;
        margin: -1rem -0.5rem 1rem -0.5rem; /* Use negative margin to span edges */
        position: sticky; /* Keep header visible */
        top: 0; /* Stick to top */
        z-index: 100; /* Ensure header is above scrolling content */
    }
    .chat-header img.avatar { width: 36px; height: 36px; border-radius: 50%; margin-right: 10px; }
    .chat-header .title { font-weight: 600; font-size: 1.05em; margin-bottom: 1px; }
    .chat-header .subtitle { font-size: 0.8em; opacity: 0.9; }

    /* --- Chat Messages --- */
    div[data-testid="stChatMessage"] {
        padding: 10px 14px;
        border-radius: 18px;
        margin-bottom: 8px;
        width: fit-content;
        max-width: 85%;
        line-height: 1.5;
        border: 1px solid #E5E7EB; /* Light border for assistant */
        box-shadow: 0 1px 1px rgba(0,0,0,0.04);
    }
    /* Assistant messages (left aligned) */
    div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #F9FAFB; /* Very light grey */
        color: #374151; /* Darker grey text */
        margin-right: auto;
    }
    /* User messages (right aligned) */
    div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
        background-color: #3B82F6; /* Primary Blue */
        color: white;
        margin-left: auto;
        margin-right: 0;
        border: none;
    }
    div[data-testid="stChatMessage"] p { margin-bottom: 0.3rem; }

    /* --- Suggestion Buttons Container & Buttons --- */
    .suggestion-container {
        padding-top: 5px;
        padding-left: 40px; /* Indent buttons */
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 10px;
    }
    .suggestion-container .stButton>button {
        background-color: #EFF6FF; /* Lightest Blue */
        color: #3B82F6; /* Primary Blue */
        border: 1px solid #BFDBFE; /* Light Blue border */
        border-radius: 16px;
        padding: 5px 12px;
        font-size: 0.85em;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .suggestion-container .stButton>button:hover { background-color: #DBEAFE; border-color: #93C5FD; }

    /* --- Chat Input --- */
    div[data-testid="stChatInput"] {
         background-color: #f0f2f5; /* Match app background */
         border-top: 1px solid #E5E7EB;
         padding: 0.75rem 1rem;
         position: fixed; /* Fix at bottom */
         bottom: 0;
         left: 0; right: 0; margin: auto; /* Center */
         max-width: 800px; /* Match content width */
         width: 100%;
         box-sizing: border-box;
         z-index: 100; /* Above content */
    }
    div[data-testid="stChatInput"] textarea { border-radius: 18px; border: 1px solid #D1D5DB; background-color: #fff; }
    div[data-testid="stChatInput"] button { /* Style send button */ background-color: #2563EB; svg {fill: white;} } /* Blue send */
    div[data-testid="stChatInput"] button:hover { background-color: #1D4ED8; }


    /* --- Hide Streamlit UI Elements --- */
    header[data-testid="stHeader"], footer, #MainMenu, .stDeployButton { display: none !important; visibility: hidden !important; }
    /* Adjust top padding of main area to account for custom fixed header */
    .main .block-container { padding-top: 70px !important; } /* Adjust based on your header height */

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
if "messages" not in st.session_state: st.session_state.messages = []
if "last_assistant_message_id_with_suggestions" not in st.session_state: st.session_state.last_assistant_message_id_with_suggestions = -1
if "processing_user_input" not in st.session_state: st.session_state.processing_user_input = None


# --- Function to add message ---
def add_message(role, content, avatar=None, suggestions=None):
    message_id = len(st.session_state.messages)
    msg = {"role": role, "content": content, "id": message_id}
    if avatar: msg["avatar"] = avatar
    if suggestions:
        msg["suggestions"] = suggestions
        st.session_state.last_assistant_message_id_with_suggestions = message_id
    st.session_state.messages.append(msg)

# --- Add initial assistant message ---
if not st.session_state.messages:
     add_message("assistant", "Salam! 👋 Ada apa yang boleh saya bantu? Sila tanya soalan atau pilih topik.", ASSISTANT_AVATAR_URL, DEFAULT_SUGGESTIONS[:3])


# --- Display chat area ---
# Container for messages to allow positioning input at bottom
chat_container = st.container()
with chat_container:
    # Display messages from history
    for message in st.session_state.messages:
        msg_id = message["id"]
        is_last_assistant = (message["role"] == "assistant" and msg_id == st.session_state.last_assistant_message_id_with_suggestions)
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Display suggestions only AFTER the last message IF it's the designated assistant message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
         last_msg = st.session_state.messages[-1]
         last_msg_id = last_msg["id"]
         if "suggestions" in last_msg and last_msg_id == st.session_state.last_assistant_message_id_with_suggestions:
              st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
              suggestions_to_show = last_msg["suggestions"][:4] # Limit visible suggestions
              for j, label in enumerate(suggestions_to_show):
                   button_key = f"button_{last_msg_id}_{j}"
                   if st.button(label, key=button_key):
                       add_message("user", label, USER_AVATAR)
                       st.session_state.last_assistant_message_id_with_suggestions = -1 # Hide suggestions
                       st.session_state.processing_user_input = label # Flag for processing
                       st.rerun() # Rerun to show user message & trigger processing
              st.markdown('</div>', unsafe_allow_html=True)


# --- Handle user text input at the bottom ---
if prompt := st.chat_input("Taip soalan anda di sini..."):
    add_message("user", prompt, USER_AVATAR)
    st.session_state.last_assistant_message_id_with_suggestions = -1 # Hide suggestions on new input
    st.session_state.processing_user_input = prompt # Flag for processing
    # Streamlit reruns automatically

# --- Generate Response Logic ---
if st.session_state.processing_user_input:
    user_input_to_process = st.session_state.processing_user_input
    st.session_state.processing_user_input = None # Clear flag

    # Add assistant placeholder message immediately
    response_id = len(st.session_state.messages)
    add_message("assistant", "...", ASSISTANT_AVATAR_URL) # Add placeholder

    # Use the placeholder created by add_message implicitly via st.chat_message context
    with st.spinner("Sedang berfikir... 🤔"): # Show spinner during processing
        full_response = "Maaf, ralat memproses permintaan." # Default error response
        source_docs = []
        if not qa_chain:
            full_response = "Maaf, sistem RAG tidak bersedia."
        else:
            try:
                start_time = time.time()
                result = qa_chain.invoke({"query": user_input_to_process})
                end_time = time.time()
                processing_time = end_time - start_time

                generated_answer_raw = result.get('result', "Maaf, ralat.")
                source_docs = result.get('source_documents', [])

                # Apply fallback/cleaning logic
                if "<extra_id_" in generated_answer_raw and source_docs:
                    fallback_content = source_docs[0].page_content
                    fallback_content = re.sub(r'\s+', ' ', fallback_content).strip()
                    full_response = f"Jawapan tepat tidak jelas, berikut maklumat berkaitan:\n\n---\n_{fallback_content[:800]}_"
                elif "<extra_id_" in generated_answer_raw:
                    full_response = "Maaf, saya tidak pasti jawapannya."
                else:
                    full_response = clean_llm_output(generated_answer_raw)

                # Add source info expander content here maybe? Or handle below.
                # For simplicity, we just update the content of the existing message

            except Exception as e:
                st.error(f"Ralat semasa memproses RAG: {e}")
                full_response = "Maaf, berlaku ralat teknikal."

        # Generate new suggestions based on the response
        new_suggestions = get_suggestions(full_response)

        # Update the placeholder message with the actual response and suggestions
        st.session_state.messages[response_id]["content"] = full_response
        st.session_state.messages[response_id]["suggestions"] = new_suggestions
        # Mark this new message as the one with suggestions
        st.session_state.last_assistant_message_id_with_suggestions = response_id

        # Rerun to display the final assistant message and its suggestions
        st.rerun()
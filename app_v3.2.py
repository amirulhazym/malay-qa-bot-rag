# --- app.py (Chat UI Enhanced & Functional) ---
import streamlit as st
import time
import torch
# Ensure correct, newer import paths if using latest langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
# Older import path, might need update depending on langchain version
# from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import re # Import regex for cleaning

# --- Page Config ---
st.set_page_config(page_title="Bot Soal Jawab BM", page_icon="🇲🇾", layout="centered")

# --- Constants ---
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_CHECKPOINT = "google/mt5-small"
ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "👤"
HEADER_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/194/194938.png"

# --- Function to Clean LLM Output ---
def clean_llm_output(text):
    """Removes common unwanted tokens like <extra_id_*> and <pad>."""
    if not isinstance(text, str): # Handle potential non-string input
        return ""
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<pad>', '', text)
    # Add more specific cleaning if needed
    # Example: remove leading/trailing whitespace after cleaning tokens
    cleaned_text = text.strip()
    # If the result is just punctuation or seems empty, return a default
    if not cleaned_text or all(c in ' .,;:!?' for c in cleaned_text):
        return "Maaf, saya tidak dapat memberikan jawapan yang jelas berdasarkan maklumat ini."
    return cleaned_text

# --- Cached Loading Functions (Keep these) ---
@st.cache_resource
def load_embeddings_model():
    print(">> (Cache) Loading embedding model...")
    # ... (rest of function same as before)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        print(f">> Embedding model loaded on {device}.")
        return embed_model
    except Exception as e:
        st.error(f"Ralat memuatkan model embedding: {e}")
        st.stop()

@st.cache_resource
def load_faiss_index(_embeddings):
    print(f">> (Cache) Loading FAISS index from: {INDEX_SAVE_PATH}...")
    # ... (rest of function same as before)
    if not _embeddings:
         st.error("Tidak dapat memuatkan index FAISS tanpa model embedding.")
         return None
    if not os.path.exists(INDEX_SAVE_PATH):
        st.error(f"Index FAISS tidak dijumpai di {INDEX_SAVE_PATH}. Pastikan ia wujud.")
        return None
    try:
        vector_store = FAISS.load_local(
            INDEX_SAVE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        print(f">> FAISS index loaded. Contains {vector_store.index.ntotal} vectors.")
        return vector_store
    except Exception as e:
        st.error(f"Ralat memuatkan index FAISS: {e}")
        return None

@st.cache_resource
def load_llm_qa_pipeline():
    print(f">> (Cache) Loading LLM pipeline: {LLM_CHECKPOINT}...")
    # ... (rest of function same as before)
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text2text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=100,
            device=device
        )
        llm_pipe = HuggingFacePipeline(pipeline=pipe)
        print(f">> LLM pipeline loaded on device {device}.")
        return llm_pipe
    except Exception as e:
        st.error(f"Ralat memuatkan LLM pipeline: {e}")
        st.stop()

# --- Load Resources & Create Chain (Keep this) ---
embeddings_model = load_embeddings_model()
vector_store = load_faiss_index(embeddings_model)
llm_pipeline = load_llm_qa_pipeline()

qa_chain = None
if vector_store and llm_pipeline:
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_pipeline,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        print(">> QA Chain ready.")
    except Exception as e:
        st.error(f"Ralat mencipta QA chain: {e}")

# --- Inject Custom CSS (Keep this) ---
st.markdown("""
<style>
    /* ... (CSS styles same as before) ... */
    .chat-header { padding: 10px 15px; background-color: #1E3A8A; color: white; border-radius: 10px 10px 0 0; margin-bottom: 10px; display: flex; align-items: center; }
    .chat-header img { width: 40px; height: 40px; border-radius: 50%; margin-right: 10px; }
    .chat-header .title { font-weight: bold; font-size: 1.1em; }
    .chat-header .subtitle { font-size: 0.9em; opacity: 0.8; }
    .stApp > header { background-color: transparent; }
     div[data-testid="stChatMessage"] { margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Custom Header (Keep this) ---
st.markdown(f"""
<div class="chat-header">
    <img src="{HEADER_IMAGE_URL}" alt="Avatar">
    <div>
        <div class="title">Chat Bantuan E-Dagang</div>
        <div class="subtitle">Kami sedia membantu!</div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": "Salam! 👋 Pilih topik atau taip soalan anda di bawah.", "buttons": ["Status Penghantaran", "Polisi Pemulangan", "Cara Pembayaran"], "id": 0}
    ]
# Ensure each message has a unique ID for button state tracking
if not all("id" in msg for msg in st.session_state.messages):
     for i, msg in enumerate(st.session_state.messages):
         msg["id"] = i

# --- Display Chat History ---
# Use a container for the chat history area
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        msg_id = message["id"] # Get unique message ID
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
            # Display buttons if they exist and haven't been used for *this specific message ID*
            if "buttons" in message and not st.session_state.get(f"buttons_used_{msg_id}", False):
                cols = st.columns(len(message["buttons"]))
                for j, label in enumerate(message["buttons"]):
                    button_key = f"button_{msg_id}_{j}" # Key includes message ID
                    if cols[j].button(label, key=button_key):
                        # Add user message simulation
                        st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": label, "id": len(st.session_state.messages)})
                        # Mark buttons for THIS message as used
                        st.session_state[f"buttons_used_{msg_id}"] = True
                        # *** NO st.rerun() here *** - Let Streamlit handle the rerun implicitly
                        st.rerun() # Use experimental rerun ONLY IF needed to force immediate update after button click before input box check


# --- Handle User Input via Chat Input Box ---
if prompt := st.chat_input("Taip mesej anda..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": prompt, "id": len(st.session_state.messages)})
    # *** NO st.rerun() here *** - The script continues below

# --- Generate Response if Last Message is from User ---
# Check if there are messages and the last one is from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_message = st.session_state.messages[-1]["content"]

    # Check if we already generated a response for this user message in this run
    # (Prevents generating response multiple times if script reruns unexpectedly)
    # A simple way is to check if the very last message is from the assistant
    generate_response = True
    if len(st.session_state.messages) > 1 and st.session_state.messages[-2]["role"] == "user" and st.session_state.messages[-1]["role"] == "assistant":
         # This implies a response was just added in this script run
         # Or more robustly, check based on IDs or add a flag
         pass # For now, allow regeneration if needed, can add stricter checks

    if generate_response:
        with st.spinner("Mencari jawapan..."):  # <<< START OF NEW BLOCK
            assistant_response_content = "Maaf, sistem RAG tidak bersedia."  # Default
            source_docs = []
            processing_time = 0
            if not qa_chain:
                st.error("Maaf, sistem RAG tidak bersedia.")
            else:
                try:
                    start_time = time.time()
                    result = qa_chain({"query": last_user_message})
                    end_time = time.time()
                    processing_time = end_time - start_time

                    generated_answer_raw = result.get('result', "Maaf, ralat semasa menjana jawapan.")
                    source_docs = result.get('source_documents', [])

                    # --- YOUR MODIFICATION START ---
                    # Check for placeholder BEFORE cleaning, as cleaning might remove it
                    if "<extra_id_" in generated_answer_raw and source_docs:
                        # Fallback: Show first source if LLM failed but sources found
                        fallback_source_content = source_docs[0].page_content
                        # Basic cleaning for the fallback source as well
                        fallback_source_content = re.sub(r'\s+', ' ', fallback_source_content).strip()  # Replace multiple spaces/newlines
                        assistant_response_content = f"Saya tidak pasti jawapan tepat, tetapi berikut adalah maklumat berkaitan yang ditemui:\n\n---\n_{fallback_source_content}_"  # Italicize source
                        print(">> LLM failed (<extra_id>), falling back to first source.")  # Debugging print
                    elif "<extra_id_" in generated_answer_raw:
                        # LLM failed, no good sources
                        assistant_response_content = "Maaf, saya tidak pasti jawapannya berdasarkan maklumat yang ada."
                        print(">> LLM failed (<extra_id>), no sources to fall back on.")  # Debugging print
                    else:
                        # LLM likely succeeded, clean its output
                        assistant_response_content = clean_llm_output(generated_answer_raw)
                        print(">> LLM generated response, applying cleaning.")  # Debugging print
                    # --- YOUR MODIFICATION END ---

                except Exception as e:
                    st.error(f"Ralat semasa memproses RAG: {e}")
                    assistant_response_content = "Maaf, berlaku ralat semasa mencari jawapan."

            # Display the final answer (potentially the fallback)
            st.markdown(assistant_response_content)

            # Display sources if any were retrieved (even if LLM failed)
            if source_docs:
                with st.expander("Lihat Sumber Rujukan Lengkap", expanded=False):  # Renamed expander
                    for k, doc in enumerate(source_docs):
                        source_name = doc.metadata.get('source', f'Sumber {k+1}')
                        st.caption(f"**{source_name}:**")
                        st.text(doc.page_content)  # Display full source text
                    # Show processing time only if RAG ran successfully
                    if processing_time > 0:
                        st.caption(f"Masa mencari: {processing_time:.2f} saat")
            elif qa_chain:
                st.caption("Tiada sumber rujukan khusus ditemui.")
            # <<< END OF NEW BLOCK

        # Add the generated response to session state
        st.session_state.messages.append({
            "role": "assistant",
            "avatar": ASSISTANT_AVATAR,
            "content": assistant_response_content,  # Store cleaned answer
            # Optionally store sources/time here too if needed for later display logic
            "id": len(st.session_state.messages)
        })
        # NOTE: We might need ONE rerun *here* after adding the assistant message
        # to ensure it displays correctly before the next input waits. Test without first.
        st.rerun()  # Add this if the assistant response doesn't show up immediately
# --- app.py (Chat UI Enhanced Version) ---
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
ASSISTANT_AVATAR = "🤖" # Feel free to use a URL to an image instead
USER_AVATAR = "👤"
HEADER_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/194/194938.png" # Example avatar for header

# --- Function to Clean LLM Output ---
def clean_llm_output(text):
    """Removes common unwanted tokens like <extra_id_*> and <pad>."""
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<pad>', '', text)
    return text.strip()

# --- Cached Loading Functions (Keep these essential functions) ---

@st.cache_resource
def load_embeddings_model():
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
        st.error(f"Ralat memuatkan model embedding: {e}")
        st.stop()

@st.cache_resource
def load_faiss_index(_embeddings):
    print(f">> (Cache) Loading FAISS index from: {INDEX_SAVE_PATH}...")
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
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT)
        device = 0 if torch.cuda.is_available() else -1
        # Limit max_length for the pipeline if needed, check model's capability
        pipe = pipeline(
            "text2text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=100, # Slightly reduced max tokens
            # temperature=0.7,
            # early_stopping=True, # Optional: stop generation earlier
            device=device
        )
        llm_pipe = HuggingFacePipeline(pipeline=pipe)
        print(f">> LLM pipeline loaded on device {device}.")
        return llm_pipe
    except Exception as e:
        st.error(f"Ralat memuatkan LLM pipeline: {e}")
        st.stop()

# --- Load Resources ---
embeddings_model = load_embeddings_model()
vector_store = load_faiss_index(embeddings_model)
llm_pipeline = load_llm_qa_pipeline()

# --- Create QA Chain ---
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

# --- Inject Custom CSS for Header (Optional, basic styling) ---
# Keep this minimal to avoid breaking Streamlit updates
st.markdown("""
<style>
    /* Basic styling for a header-like area */
    .chat-header {
        padding: 10px 15px;
        background-color: #1E3A8A; /* Dark Blue */
        color: white;
        border-radius: 10px 10px 0 0;
        margin-bottom: 10px; /* Space below header */
        display: flex;
        align-items: center;
    }
    .chat-header img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }
    .chat-header .title {
        font-weight: bold;
        font-size: 1.1em;
    }
    .chat-header .subtitle {
        font-size: 0.9em;
        opacity: 0.8;
    }
    /* Style Streamlit's main block slightly */
    .stApp > header {
        background-color: transparent; /* Hide default header */
    }
    /* Ensure chat messages container has some padding */
     div[data-testid="stChatMessage"] {
         margin-bottom: 10px;
     }

</style>
""", unsafe_allow_html=True)

# --- Custom Header ---
# Using markdown with unsafe_allow_html to structure the header
st.markdown(f"""
<div class="chat-header">
    <img src="{HEADER_IMAGE_URL}" alt="Avatar">
    <div>
        <div class="title">Chat Bantuan E-Dagang</div>
        <div class="subtitle">Kami sedia membantu!</div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Initialize Chat History and State ---
if "messages" not in st.session_state:
    # Start with initial greeting and quick replies simulation
    st.session_state.messages = [
        {"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": "Salam! 👋 Pilih topik atau taip soalan anda di bawah.", "buttons": ["Status Penghantaran", "Polisi Pemulangan", "Cara Pembayaran"]}
    ]
if "buttons_shown" not in st.session_state:
     st.session_state.buttons_shown = True # Flag to show initial buttons only once

# --- Display Chat History ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])
        # Display buttons associated with this message, if any, and if they haven't been used
        if "buttons" in message and st.session_state.get(f"buttons_used_{i}", False) is False:
             cols = st.columns(len(message["buttons"]))
             for j, label in enumerate(message["buttons"]):
                 # Add a unique key based on message index and button index
                 button_key = f"button_{i}_{j}"
                 if cols[j].button(label, key=button_key):
                     # When button is clicked:
                     # 1. Add user message simulating the button click
                     st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": label})
                     # 2. Mark these buttons as used
                     st.session_state[f"buttons_used_{i}"] = True
                     # 3. Rerun the script to process the new user message
                     st.rerun()

# --- Handle User Input ---
if prompt := st.chat_input("Taip mesej anda..."):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": prompt})

    # 2. Generate and display assistant response using RAG
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if not qa_chain:
            st.error("Maaf, sistem RAG tidak bersedia.")
            assistant_response_content = "Maaf, sistem RAG tidak bersedia."
        else:
            with st.spinner("Sedang mencari jawapan..."):
                try:
                    start_time = time.time()
                    result = qa_chain({"query": prompt})
                    end_time = time.time()

                    generated_answer = result.get('result', "Maaf, ralat semasa menjana jawapan.")
                    # Clean the output
                    cleaned_answer = clean_llm_output(generated_answer)
                    if not cleaned_answer: # If cleaning results in empty string
                        cleaned_answer = "Maaf, saya tidak pasti jawapannya berdasarkan maklumat yang ada."

                    st.markdown(cleaned_answer) # Display the cleaned answer

                    # Optionally display sources
                    source_docs = result.get('source_documents', [])
                    if source_docs:
                        with st.expander("Lihat Sumber Rujukan", expanded=False):
                            for k, doc in enumerate(source_docs):
                                source_name = doc.metadata.get('source', f'Sumber {k+1}')
                                # Use code block for better readability of source content
                                st.info(f"**{source_name}:**\n```\n{doc.page_content}\n```")
                            st.caption(f"Masa mencari: {end_time - start_time:.2f} saat")
                    assistant_response_content = cleaned_answer # Store only the answer in history for now

                except Exception as e:
                    st.error(f"Ralat semasa memproses RAG: {e}")
                    assistant_response_content = "Maaf, berlaku ralat semasa mencari jawapan."

    # 3. Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": assistant_response_content})

    # 4. Rerun to display the latest messages immediately
    st.rerun()
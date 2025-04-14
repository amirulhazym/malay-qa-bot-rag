# --- app_v3.py (Shopee-Style UI & Flow) ---
import streamlit as st
import time
import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import re

# --- Page Config ---
st.set_page_config(page_title="Bot Bantuan BM", page_icon="🇲🇾", layout="centered")

# --- Constants ---
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index"
EMBEDDING_MODEL_NAME = "mesolitica/mistral-embedding-191m-8k-contrastive"
# --- Use the local fine-tuned model ---
LLM_CHECKPOINT = "./malay-qa-model-finetuned" # <-- CHANGED TO LOCAL MODEL
ASSISTANT_AVATAR = "🤖" # Consider changing to Shopee-like avatar if desired
USER_AVATAR = "👤"
HEADER_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/194/194938.png" # Keep or change

# --- Function to Clean LLM Output (Keep) ---
def clean_llm_output(text):
    """Removes common unwanted tokens like <extra_id_*> and <pad>."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<pad>', '', text)
    cleaned_text = text.strip()
    if not cleaned_text or all(c in ' .,;:!?' for c in cleaned_text):
        # More generic fallback if LLM fails *even with fine-tuned model*
        return "Maaf, saya tidak dapat memproses jawapan buat masa ini."
    return cleaned_text

# --- Predefined Q&A ---
# Map questions (button labels) to predefined answers or actions
# Using Malay based on image context
PREDEFINED_QUESTIONS = {
    "Status Pemulangan/Bayaran Balik": "Untuk menyemak status pemulangan atau bayaran balik anda, sila pergi ke bahagian 'Pesanan Saya' dan pilih item yang berkenaan.",
    "Percepatkan Penghantaran Pakej": "Maaf, kelajuan penghantaran bergantung pada perkhidmatan kurier. Anda boleh menjejaki pakej anda dalam aplikasi.",
    "Terma Pembayaran SPayLater": "Terma SPayLater termasuk kitaran bil bulanan dan caj lewat bayar jika berkenaan. Sila rujuk aplikasi Shopee untuk butiran penuh.",
    "Kenapa tak boleh bayar guna ShopeePay?": "Sila pastikan baki ShopeePay anda mencukupi dan akaun anda aktif. Jika masalah berterusan, hubungi khidmat pelanggan Shopee.",
    "Lain-lain Soalan Lazim": "Anda boleh rujuk Pusat Bantuan Shopee untuk senarai penuh soalan lazim.",
    # Add more questions and answers as needed
}

# --- Cached Loading Functions (Keep, but update LLM loading) ---
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
def load_llm_qa_pipeline(model_path): # Takes path now
    print(f">> (Cache) Loading LLM pipeline from local path: {model_path}...")
    if not os.path.isdir(model_path):
         st.error(f"Direktori model LLM tidak dijumpai: {model_path}")
         return None
    try:
        # Ensure the local model has the necessary config files (config.json, etc.)
        llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        device = 0 if torch.cuda.is_available() else -1 # Use GPU if available
        pipe = pipeline(
            "text2text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=150, # Increased slightly
            device=device
        )
        llm_pipe = HuggingFacePipeline(pipeline=pipe)
        print(f">> LLM pipeline loaded from {model_path} on device {device}.")
        return llm_pipe
    except Exception as e:
        st.error(f"Ralat memuatkan LLM pipeline dari {model_path}: {e}")
        st.stop() # Stop if fine-tuned model fails to load

# --- Load Resources ---
embeddings_model = load_embeddings_model()
vector_store = load_faiss_index(embeddings_model)
# --- Load fine-tuned LLM ---
llm_pipeline = load_llm_qa_pipeline(LLM_CHECKPOINT)

# --- Define Prompt Template (Still needed for RAG fallback) ---
prompt_template_text = """Gunakan konteks berikut untuk menjawab soalan di akhir. Jawab hanya berdasarkan konteks yang diberikan dalam Bahasa Melayu. Jika jawapan tiada dalam konteks, nyatakan "Maaf, maklumat tiada dalam pangkalan data.".

Konteks:
{context}

Soalan: {question}
Jawapan Membantu:"""

PROMPT = PromptTemplate(
    template=prompt_template_text, input_variables=["context", "question"]
)

# --- Create QA Chain (Only if resources loaded successfully) ---
qa_chain = None
if vector_store and llm_pipeline and PROMPT:
    try:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 10}
        )
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_pipeline,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        print(">> QA Chain ready with Fine-Tuned Model and Custom Prompt.")
    except Exception as e:
        st.error(f"Ralat mencipta QA chain: {e}")
        # App can continue but RAG won't work
else:
    st.warning("Sistem RAG tidak dapat dimulakan sepenuhnya. Carian mungkin tidak berfungsi.")

# --- Inject Custom CSS (Keep or modify) ---
st.markdown("""
<style>
    .stButton>button { width: 100%; text-align: left; margin-bottom: 5px; } /* Style suggested question buttons */
    .chat-header { padding: 10px 15px; background-color: #1E3A8A; color: white; border-radius: 10px 10px 0 0; margin-bottom: 10px; display: flex; align-items: center; }
    .chat-header img { width: 40px; height: 40px; border-radius: 50%; margin-right: 10px; }
    .chat-header .title { font-weight: bold; font-size: 1.1em; }
    .chat-header .subtitle { font-size: 0.9em; opacity: 0.8; }
    .stApp > header { background-color: transparent; }
    div[data-testid="stChatMessage"] { margin-bottom: 10px; }
    /* Container for suggested questions */
    .suggested-questions-container {
        padding: 15px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .suggested-questions-container h4 { margin-top: 0; margin-bottom: 10px; color: #555; }
</style>
""", unsafe_allow_html=True)

# --- Custom Header (Keep) ---
st.markdown(f"""
<div class="chat-header">
    <img src="{HEADER_IMAGE_URL}" alt="Avatar">
    <div>
        <div class="title">Chat Bantuan E-Dagang</div>
        <div class="subtitle">Kami sedia membantu!</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Initialize Chat History & State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": "Salam! 👋 Bagaimana saya boleh bantu anda hari ini?"}
    ]
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True # Show suggestions initially

# --- Function to Handle Response Generation ---
def generate_response(user_query):
    # 1. Check if query matches a predefined question
    if user_query in PREDEFINED_QUESTIONS:
        return PREDEFINED_QUESTIONS[user_query], [] # Return predefined answer, no sources

    # 2. If no predefined match, use RAG chain (if available)
    elif qa_chain:
        try:
            with st.spinner("Mencari jawapan dalam pangkalan data..."):
                start_time = time.time()
                result = qa_chain.invoke({"query": user_query})
                end_time = time.time()
                processing_time = end_time - start_time
                print(f">> RAG processing time: {processing_time:.2f}s")

                generated_answer_raw = result.get('result', "")
                source_docs = result.get('source_documents', [])

                # Clean the output from the fine-tuned model
                assistant_response_content = clean_llm_output(generated_answer_raw)

                # Add source info if available
                if source_docs:
                     # Simple source indication
                     assistant_response_content += "\n\n_(Sumber dari pangkalan data)_"

                return assistant_response_content, source_docs # Return RAG answer and sources

        except Exception as e:
            st.error(f"Ralat semasa memproses RAG: {e}")
            return "Maaf, berlaku ralat semasa mencari jawapan.", []
    else:
        # Fallback if RAG chain isn't ready
        return "Maaf, saya tidak dapat mencari jawapan dalam pangkalan data buat masa ini.", []

# --- Display Chat History ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

# --- Display Suggested Questions ---
suggestions_container = st.container()
if st.session_state.show_suggestions:
    with suggestions_container:
        st.markdown('<div class="suggested-questions-container">', unsafe_allow_html=True)
        st.markdown("<h4>Anda mungkin ingin bertanya:</h4>", unsafe_allow_html=True)
        for question in PREDEFINED_QUESTIONS.keys():
            button_key = f"suggest_{question}"
            if st.button(question, key=button_key):
                # Add user message (the question)
                st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": question})
                # Get and add predefined assistant response
                response_text, _ = generate_response(question) # Ignore sources for predefined
                st.session_state.messages.append({"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": response_text})
                # Hide suggestions after a button is clicked (optional)
                st.session_state.show_suggestions = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# --- Handle User Input via Chat Input Box ---
if prompt := st.chat_input("Taip mesej anda..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "avatar": USER_AVATAR, "content": prompt})
    # Hide suggestions when user types
    st.session_state.show_suggestions = False

    # Generate and add assistant response (could be predefined or RAG)
    response_text, source_docs = generate_response(prompt) # Use the function
    assistant_message = {"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": response_text}
    # We could potentially add sources to the message dict if needed later
    st.session_state.messages.append(assistant_message)

    # Rerun to display the new messages and hide suggestions
    st.rerun()

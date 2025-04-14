import streamlit as st
import time
import torch
import datetime
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os

# --- Page Configuration ---
st.set_page_config(page_title="Bot Soal Jawab BM", page_icon="🇲🇾", layout="centered")

# --- Constants ---
INDEX_SAVE_PATH = "faiss_malay_ecommerce_kb_index"
EMBEDDING_MODEL_NAME = "mesolitica/mistral-embedding-191m-8k-contrastive"
LLM_CHECKPOINT = "google/mt5-base"
ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "👤"
HEADER_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/194/194938.png"

# --- Function to Clean LLM Output ---
def clean_llm_output(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<pad>', '', text)
    cleaned_text = text.strip()
    if not cleaned_text or all(c in ' .,;:!?' for c in cleaned_text):
        return "Maaf, saya tidak dapat memberikan jawapan yang jelas berdasarkan maklumat ini."
    return cleaned_text

# --- Cached Loading Functions ---
@st.cache_resource
def load_embeddings_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': device})
        return embed_model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()

@st.cache_resource
def load_faiss_index(_embeddings):
    if not _embeddings:
        st.error("Cannot load FAISS index without embedding model.")
        return None
    if not os.path.exists(INDEX_SAVE_PATH):
        st.error(f"FAISS index not found at {INDEX_SAVE_PATH}. Ensure it exists.")
        return None
    try:
        vector_store = FAISS.load_local(INDEX_SAVE_PATH, _embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

@st.cache_resource
def load_llm_qa_pipeline():
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=100, device=device)
        llm_pipe = HuggingFacePipeline(pipeline=pipe)
        return llm_pipe
    except Exception as e:
        st.error(f"Error loading LLM pipeline: {e}")
        st.stop()

# --- Load Resources & Create Chain ---
embeddings_model = load_embeddings_model()
vector_store = load_faiss_index(embeddings_model)
llm_pipeline = load_llm_qa_pipeline()

# --- Define Custom Prompt Template ---
prompt_template_text = """Gunakan konteks berikut untuk menjawab soalan di akhir. Jawab hanya berdasarkan konteks yang diberikan. Jika jawapan tiada dalam konteks, nyatakan "Maaf, maklumat tiada dalam pangkalan data.".

Konteks:
{context}

Soalan: {question}
Jawapan Membantu:"""

PROMPT = PromptTemplate(template=prompt_template_text, input_variables=["context", "question"])

qa_chain = None
if vector_store and llm_pipeline and PROMPT:
    try:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 10})
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(llm=llm_pipeline, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")

# --- Inject Custom CSS ---
st.markdown("""
<style>
    .chat-header { padding: 10px 15px; background-color: #1E3A8A; color: white; border-radius: 10px 10px 0 0; margin-bottom: 10px; display: flex; align-items: center; }
    .chat-header img { width: 40px; height: 40px; border-radius: 50%; margin-right: 10px; }
    .chat-header .title { font-weight: bold; font-size: 1.1em; }
    .chat-header .subtitle { font-size: 0.9em; opacity: 0.8; }
    .stApp > header { background-color: transparent; }
    div[data-testid="stChatMessage"] { margin-bottom: 10px; }
    .stChatMessage--assistant { background-color: #FFDAB9; border-radius: 10px; padding: 10px; margin-bottom: 10px; max-width: 70%; margin-right: auto; }
    .stChatMessage--user { background-color: #F0F0F0; border-radius: 10px; padding: 10px; margin-bottom: 10px; max-width: 70%; margin-left: auto; }
    .stButton > button { background-color: #F0F0F0; color: #333; border: none; border-radius: 20px; padding: 8px 16px; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# --- Custom Header ---
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
        {"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": "Salam! 👋 Pilih topik atau taip soalan anda di bawah.", "buttons": ["Status Penghantaran →", "Polisi Pemulangan →", "Cara Pembayaran →"], "id": 0, "timestamp": datetime.datetime.now().strftime("%H:%M")}
    ]

# Ensure each message has a unique ID
if not all("id" in msg for msg in st.session_state.messages):
    for i, msg in enumerate(st.session_state.messages):
        msg["id"] = i

# --- Display Chat History ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        msg_id = message["id"]
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(message["timestamp"])
            if "buttons" in message and not st.session_state.get(f"buttons_used_{msg_id}", False):
                cols = st.columns(len(message["buttons"]))
                for j, label in enumerate(message["buttons"]):
                    if cols[j].button(label, key=f"button_{msg_id}_{j}"):
                        st.session_state.messages.append({
                            "role": "user",
                            "avatar": USER_AVATAR,
                            "content": label,
                            "timestamp": datetime.datetime.now().strftime("%H:%M"),
                            "id": len(st.session_state.messages)
                        })
                        st.session_state[f"buttons_used_{msg_id}"] = True
                        st.rerun()

# --- Handle User Input ---
if prompt := st.chat_input("Taip mesej anda..."):
    st.session_state.messages.append({
        "role": "user",
        "avatar": USER_AVATAR,
        "content": prompt,
        "timestamp": datetime.datetime.now().strftime("%H:%M"),
        "id": len(st.session_state.messages)
    })

# --- Generate Assistant Response ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_message = st.session_state.messages[-1]["content"]
    with st.spinner("Mencari jawapan..."):
        assistant_response_content = "Maaf, sistem RAG tidak bersedia."
        source_docs = []
        processing_time = 0
        if qa_chain:
            try:
                start_time = time.time()
                result = qa_chain({"query": last_user_message})
                end_time = time.time()
                processing_time = end_time - start_time
                generated_answer_raw = result.get('result', "Maaf, ralat semasa menjana jawapan.")
                source_docs = result.get('source_documents', [])
                if "<extra_id_" in generated_answer_raw and source_docs:
                    fallback_source_content = source_docs[0].page_content
                    fallback_source_content = re.sub(r'\s+', ' ', fallback_source_content).strip()
                    assistant_response_content = f"Saya tidak pasti jawapan tepat, tetapi berikut adalah maklumat berkaitan yang ditemui:\n\n---\n_{fallback_source_content}_"
                elif "<extra_id_" in generated_answer_raw:
                    assistant_response_content = "Maaf, saya tidak pasti jawapannya berdasarkan maklumat yang ada."
                else:
                    assistant_response_content = clean_llm_output(generated_answer_raw)
            except Exception as e:
                st.error(f"Error processing RAG: {e}")
                assistant_response_content = "Maaf, berlaku ralat semasa mencari jawapan."

        # Add related topic buttons based on keywords
        related_topics = {
            "penghantaran": ["Polisi Penghantaran →", "Jejak Penghantaran →"],
            "pemulangan": ["Polisi Pemulangan →", "Permintaan Pemulangan →"],
            "pembayaran": ["Kaedah Pembayaran →", "Status Pembayaran →"]
        }
        buttons = []
        for keyword, topics in related_topics.items():
            if keyword in last_user_message.lower():
                buttons = topics
                break

        # Append assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "avatar": ASSISTANT_AVATAR,
            "content": assistant_response_content,
            "buttons": buttons if buttons else None,
            "timestamp": datetime.datetime.now().strftime("%H:%M"),
            "id": len(st.session_state.messages)
        })

        # Display the response
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(assistant_response_content)
            st.caption(st.session_state.messages[-1]["timestamp"])
            if buttons:
                cols = st.columns(len(buttons))
                for j, label in enumerate(buttons):
                    if cols[j].button(label, key=f"button_{st.session_state.messages[-1]['id']}_{j}"):
                        st.session_state.messages.append({
                            "role": "user",
                            "avatar": USER_AVATAR,
                            "content": label,
                            "timestamp": datetime.datetime.now().strftime("%H:%M"),
                            "id": len(st.session_state.messages)
                        })
                        st.session_state[f"buttons_used_{st.session_state.messages[-1]['id']}"] = True
                        st.rerun()

        # Display sources
        if source_docs:
            with st.expander("Lihat Sumber Maklumat", expanded=False):
                for k, doc in enumerate(source_docs):
                    source_name = doc.metadata.get('source', f'Sumber {k+1}')
                    st.markdown(f"**{source_name}:**")
                    st.text(doc.page_content[:200] + "...")
                if processing_time > 0:
                    st.caption(f"Masa mencari: {processing_time:.2f} saat")
        elif qa_chain:
            st.caption("Tiada sumber rujukan khusus ditemui.")

# --- Reset Chat Button ---
if st.button("Mulakan Semula"):
    st.session_state.messages = [
        {"role": "assistant", "avatar": ASSISTANT_AVATAR, "content": "Salam! 👋 Pilih topik atau taip soalan anda di bawah.", "buttons": ["Status Penghantaran →", "Polisi Pemulangan →", "Cara Pembayaran →"], "id": 0, "timestamp": datetime.datetime.now().strftime("%H:%M")}
    ]
    st.rerun()
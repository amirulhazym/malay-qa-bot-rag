# Full, Upgraded Code for: v2_multilingual_api/frontend/app.py

import streamlit as st
import requests
import json
import time

# --- API Configuration ---
BACKEND_BASE_URL = "http://127.0.0.1:8000"
ASK_URL = f"{BACKEND_BASE_URL}/api/ask"
SUGGEST_URL = f"{BACKEND_BASE_URL}/api/suggest_questions"

# --- Page Configuration ---
st.set_page_config(
    page_title="AuraCart AI Assistant",
    page_icon="🛒",
    layout="centered"
)

# --- Custom CSS for Modern Dark Mode UI ---
st.markdown("""
<style>
    /* Main app background */
    body {
        background-color: #0E1117; /* Streamlit's default dark background */
    }
    .stApp {
        background-color: #0E1117;
    }

    /* Main chat window container */
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        background-color: #1E1E1E; /* A slightly lighter dark for the container */
        border-radius: 15px;
        border: 1px solid #31333F;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat bubbles */
    .st-emotion-cache-1c7y2kd {
        border-radius: 20px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
        max-width: 85%;
    }
    
    /* User chat bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] p) {
        /* This is a trick to target the user message based on its structure */
        /* background-color: #007BFF; */ /* A nice blue for user messages */
        /* color: white; */
    }
    
    /* Assistant chat bubble */
    div[data-testid="stChatMessage"]:not(:has(div[data-testid="stMarkdownContainer"] p)) {
       /* background-color: #31333F; */ /* A grey for assistant messages */
    }

    /* Hiding Streamlit's default hamburger menu and footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Styling for the suggestion buttons */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #4A4A4A;
        background-color: #262730;
        color: #FAFAFA;
        transition: all 0.2s ease-in-out;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #31333F;
        border-color: #007BFF;
        color: #FFFFFF;
    }

    /* Source expander styling */
    .stExpander {
        border: none !important;
        box-shadow: none !important;
    }
    .stExpander>div:first-child {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Temporary: Sidebar for Demo Mode - Toggle of Source Logs ---
st.sidebar.title("👨‍💻 Demo Controls")
st.sidebar.info("Use this control to change the demo experience.")
is_dev_mode = st.sidebar.toggle(
    "Show Developer Info", 
    value=True, 
    help="Turn this on to see the retrieval sources for each answer. Turn it off for a clean end-user view."
)

# --- Helper Functions ---
def get_suggestions(history):
    """Calls the backend to get suggested questions."""
    try:
        api_payload = {"history": history}
        response = requests.post(SUGGEST_URL, json=api_payload, timeout=5)
        response.raise_for_status()
        return response.json().get("suggestions", [])
    except requests.exceptions.RequestException:
        return ["How can I return an item?", "What are my payment options?", "How do I track my delivery?"]

# --- Main App Logic ---

# Create the main chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    st.title("🛒 AuraCart AI Assistant")
    st.caption("Your friendly AI guide to AuraCart.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = ["What is the return policy?", "How long does shipping take?", "How do I contact support?"]

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if is_dev_mode and "sources" in message and message["sources"]:
                with st.expander("🔍 Show Retrieval Sources (Developer View)"):
                    for i, source in enumerate(message["sources"]):
                        st.info(f"**Source {i+1}:** `{source.get('source', 'Unknown')}`\n\n---\n\n{source.get('content', 'No content available.')}")

    # Display suggestion buttons
    if st.session_state.suggestions:
        # Create 3 columns for the buttons
        cols = st.columns(3)
        for i, suggestion in enumerate(st.session_state.suggestions[:3]):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.prompt = suggestion
                    st.session_state.suggestions = [] # Clear suggestions after one is clicked
                    st.rerun()

    # Handle new user input, checking both text input and button clicks
    prompt_from_input = st.chat_input("Ask about returns, shipping, or anything else!")
    prompt = st.session_state.get('prompt', prompt_from_input)

    if prompt:
        st.session_state.pop('prompt', None)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Immediately rerun to show the user's message
        st.rerun()

    # If the last message was from the user, get the bot's response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Aura is typing..."):
                history_for_api = [
                    {"user": m["content"]} if m["role"] == "user" else {"bot": m["content"]}
                    for m in st.session_state.messages
                ]
                api_payload = {"question": st.session_state.messages[-1]["content"], "history": history_for_api[:-1]}
                
                try:
                    response = requests.post(ASK_URL, json=api_payload, timeout=30)
                    response.raise_for_status()
                    bot_response_data = response.json()
                    bot_response_text = bot_response_data.get("answer", "I'm sorry, something went wrong.")
                    sources = bot_response_data.get("sources", [])
                    st.session_state.messages.append({"role": "assistant", "content": bot_response_text, "sources": sources})
                    st.session_state.suggestions = get_suggestions(history_for_api)

                except requests.exceptions.RequestException as e:
                    error_message = f"Sorry, I couldn't connect to the Aura AI service. (Error: {e})"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.session_state.suggestions = []
        
        # Rerun to display the bot's response and new suggestions
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
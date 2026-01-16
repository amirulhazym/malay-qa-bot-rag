import streamlit as st
import time
from graph import get_app

# Page Config
st.set_page_config(page_title="MaiStorage V3 Agentic RAG", layout="wide")

# Sidebar
with st.sidebar:
    st.title("MaiStorage V3")
    st.caption("Agentic RAG Upgrade")
    st.markdown("---")
    
    # API Keys (Optional if .env is set, but good for UI control)
    st.subheader("Configuration")
    st.info("System is using keys from .env")

# Main Interface
st.header("MaiStorage Agentic Interface")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "app" not in st.session_state:
    # Initialize the graph once
    with st.spinner("Initializing Agent Brain..."):
        st.session_state.app = get_app()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Agent Execution
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # UI Status Container
        with st.status("Agent Reasoning Process", expanded=True) as status:
            
            # Prepare inputs
            inputs = {"question": prompt, "web_search": False}
            
            # Run the graph stream
            try:
                for output in st.session_state.app.stream(inputs):
                    for key, value in output.items():
                        
                        if key == "retrieve":
                            status.write("🔍 Retriever Active: Fetching documents...")
                            docs = value.get("documents", [])
                            st.caption(f"Found {len(docs)} relevant chunks.")
                            time.sleep(0.3)
                            
                        elif key == "grade_documents":
                            status.write("👩‍🏫 Grader Active: Checking relevance...")
                            if value.get("web_search"):
                                status.warning("⚠️ Low Relevance Detected -> Triggering Web Search")
                            else:
                                status.success("✅ Documents are relevant.")
                            time.sleep(0.3)
                            
                        elif key == "web_search_node":
                            status.write("🌐 Web Search Active: Consulting Tavily...")
                            time.sleep(0.3)
                            
                        elif key == "generate":
                            status.write("🤖 Generator Active: Synthesizing answer...")
                            full_response = value.get("generation", "")

                status.update(label="Agent Finished", state="complete", expanded=False)
                
                # Display Final Answer
                response_placeholder.markdown(full_response)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                status.update(label="Error", state="error")

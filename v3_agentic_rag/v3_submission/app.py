import streamlit as st
import time
from backend import run_agent

# Page Config
st.set_page_config(page_title="MaiStorage V3 Agentic Upgrade", layout="wide")

# Sidebar
with st.sidebar:
    st.title("MaiStorage V3 Agentic Upgrade")
    st.markdown("---")
    enable_search = st.checkbox("Enable Internet Search (Tavily)", value=True)
    st.markdown("### System Status")
    st.success("Hybrid Retriever: Active")
    st.success("Gemini 2.0 Flash: Active")

st.header("V3 Agentic RAG Interface")

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask about the policies..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Visualization of Thought Process
        with st.status("Agent Reasoning Process", expanded=True) as status:
            
            # Run the agent stream
            for step_output in run_agent(prompt):
                
                # Check which node just finished
                for node_name, state_update in step_output.items():
                    
                    if node_name == "retrieve":
                        status.write("Fetching from Vector DB (Chroma + BM25)...")
                        num_docs = len(state_update.get("documents", []))
                        st.caption(f"Found {num_docs} potential fragments.")
                        time.sleep(0.5) # UX pause
                        
                    elif node_name == "grade_documents":
                        status.write("Grading document relevance...")
                        if state_update.get("web_search"):
                            status.warning("⚠️ Knowledge Gap Detected -> Searching Web...")
                        else:
                            status.info("Documents are sufficient.")
                        time.sleep(0.5)

                    elif node_name == "web_search":
                        status.write("Running Tavily Search...")
                        # We can look at the docs added in this step if needed
                        time.sleep(0.5)
                        
                    elif node_name == "generate":
                        status.write("Synthesizing final answer...")
                        full_response = state_update.get("generation", "")

            status.update(label="Agent Finished", state="complete", expanded=False)

        # Show Final Answer
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

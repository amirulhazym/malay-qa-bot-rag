# MaiStorage V3 Agentic RAG - Project Status Checkpoint

**Date:** 16 January 2026  
**Version:** V3.0 (Prototype)  
**Architecture:** Agentic RAG (Router + Corrective Strategy)

---

## 🚀 Project Overview
This document serves as a technical checkpoint for the "MaiStorage V3" upgrade. The system has been successfully migrated from a basic RAG implementation to an **Agentic Workflow** using **LangGraph**. It features a self-correcting retrieval mechanism, hybrid search, and a "Glass-Box" UI that visualizes the AI's reasoning process.

---

## 🏗️ System Architecture

### 1. Data Pipeline & Retrieval (`app/utils.py`)
- **Source Data:** Converted existing Markdown knowledge base (`v2_multilingual`) into PDFs for standardized ingestion.
- **Ingestion Logic:** 
  - **Loader:** `PyPDFLoader` via `DirectoryLoader`.
  - **Splitter:** `RecursiveCharacterTextSplitter` (Chunk Size: 1000, Overlap: 200).
- **Retrieval Strategy:** **Hybrid Search (EnsembleRetriever)**
  - **Sparse:** BM25 (Keyword matching for specific policy terms).
  - **Dense:** ChromaDB (Semantic search with `text-embedding-004`).
  - **Weights:** 50% BM25 / 50% Vector.

### 2. The Agentic Brain (`app/graph.py`)
Built on **LangGraph**, the agent follows a **"Retrieve-Grade-Generate"** flow with a corrective web search loop.

- **State Management:** `AgentState` tracks the question, retrieved documents, generation, and search triggers.
- **Workflow Nodes:**
  1.  **`retrieve`**: Fetches top 5 documents from the Hybrid Retriever.
  2.  **`grade_documents`**: Uses Gemini to score document relevance (Binary Yes/No).
  3.  **`web_search_node`**: Triggered if documents are graded as irrelevant. Uses **Tavily API** to bridge knowledge gaps.
  4.  **`generate`**: Synthesizes the final answer using valid context (Local or Web) with strict citation rules.

- **LLM Configuration:** 
  - **Model:** `gemini-3-flash-preview` (Prioritized for performance, note rate limits).
  - **Fallback:** `gemini-1.5-flash` (recommended if Free Tier `429` errors persist).

### 3. User Interface (`app/main.py`)
- **Framework:** Streamlit.
- **Features:**
  - **Agent Reasoning:** A collapsible `st.status` box showing real-time decisions (e.g., "⚠️ Knowledge Gap Detected -> Searching Web...").
  - **Chat History:** Persistent session state.
  - **Streaming:** Real-time token streaming from the Agent.

---

## ✅ Completed Tasks & Fixes

### Setup & Infrastructure
- [x] **Folder Structure:** Implemented professional separation of concerns (`app/` package, `data/` folder).
- [x] **Dependencies:** Locked versions in `requirements.txt` to avoid conflicts (resolved `langchain-chroma` vs `chromadb` conflict).
- [x] **Format Conversion:** Created `convert_docs.py` to auto-convert legacy `.md` knowledge base files to `.pdf`.

### Bug Fixes
- [x] **Import Errors:** Fixed `ModuleNotFoundError` by correcting import paths for Streamlit's execution context (Sibling imports vs Package imports).
- [x] **Graph Logic:** Resolved `ValueError` where the state key `"web_search"` conflicted with the node name (Renamed node to `web_search_node`).
- [x] **Index Corruption:** Created `force_reindex.py` to wipe corrupted/legacy vector stores and force a fresh build of the Hybrid Index.

---

## ⚠️ Known Constraints & Next Steps

1.  **Rate Limits (Gemini 3 Flash Preview):**
    - The current model `gemini-3-flash-preview` has a strict free-tier limit (5 RPM).
    - **Mitigation:** If `429 ResourceExhausted` occurs, wait 10-15 seconds.
2.  **Telemetry Noise:**
    - ChromaDB emits standard telemetry warnings in the console. These are harmless but can be silenced via environment variables (`CHROMADB_TELEMETRY_ENABLED=false`).

---

## 🛠️ How to Run

1.  **Environment Variables:** Ensure `.env` contains:
    ```env
    GOOGLE_API_KEY=your_key
    TAVILY_API_KEY=your_key
    ```
2.  **Start Application:**
    ```powershell
    v3env\Scripts\streamlit run app/main.py
    ```

---

**Status:** READY FOR DEMO

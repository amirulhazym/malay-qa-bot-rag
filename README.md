# Multilingual RAG Question-Answering Bot for E-commerce 🛒

[![Hugging Face App (V2)](https://img.shields.io/badge/Hugging_Face-Live_Demo-FFD21E?logo=huggingface)](https://huggingface.co/spaces/amirulhazym/multilingual-auracart-ai-assistant)
[![FastAPI Backend (V2)](https://img.shields.io/badge/FastAPI-Backend_API-05998B?logo=fastapi)](https://huggingface.co/spaces/amirulhazym/multilingual-auracart-ai-assistant)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-blueviolet?logo=langchain)](https://www.langchain.com/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This repository showcases the evolution of a RAG-based chatbot, demonstrating a clear progression from a self-hosted, monolithic prototype (V1) to a professional, API-driven, and globally deployed conversational AI (V2). The primary objective was to master and implement state-of-the-art techniques in LLM application development, including advanced RAG pipelines, decoupled architectures, and robust deployment strategies as part of the intensive **G-v5.6-Go** training program.

The initial system (**V1**) was a Malay-only chatbot built to validate the core RAG concept using a self-hosted LLM and a local FAISS index. After identifying key limitations in generation quality and retrieval accuracy, the project was upgraded.

The final system (**V2**), "AuraCart AI Assistant," is a fully multilingual chatbot capable of answering questions about a fictional e-commerce platform in both English and Malay. It leverages a powerful **retrieve-and-re-rank** pipeline with cloud services to ensure high-quality, factually grounded answers, and is deployed as a resilient two-process application in a custom Docker environment on Hugging Face Spaces.

## ✨ Live Demo (V2)

- **🚀 Interactive Web App (Hugging Face Spaces):** [**https://huggingface.co/spaces/amirulhazym/multilingual-auracart-ai-assistant**](https://huggingface.co/spaces/amirulhazym/multilingual-auracart-ai-assistant)

*(A GIF of the final Streamlit application in action, showing multilingual queries.)*
![AuraCart AI Assistant Demo](https://i.imgur.com/YOUR_GIF_ID.gif) 
<!-- You can create a GIF using tools like ScreenToGif and upload it to a site like Imgur to get a link -->

## 🏗️ Architectural Evolution: From V1 Prototype to V2 Production

This flowchart illustrates the project's journey, highlighting the limitations of the V1 prototype that directly motivated the architectural upgrade to the V2 system.

```mermaid
flowchart TD
    subgraph "V1: Self-Hosted Monolithic Prototype (`app.py`)"
        A[User] -- Interacts with --> B{Streamlit UI & Backend Logic};
        B -- Runs Internal RAG --> C[RAG Pipeline];
        C -- Retrieves from --> D[Local FAISS Index];
        C -- Generates with --> E[Local FLAN-T5-Small];
        E -- Returns Answer/Fallback --> B;
    end

    B --> F{"<div style='text-align: left; font-weight:bold;'>Limitations Identified</div><div style='text-align: left;'>- <b>LLM Generation Failure:</b> Local model (FLAN-T5) often failed, requiring a fallback.<br/>- <b>Retrieval Inconsistency:</b> Basic similarity search was unreliable for nuanced queries.<br/>- <b>Monolithic Design:</b> UI and AI logic tightly coupled, hard to scale.<br/>- <b>Single Language:</b> Only supported Malay.</div>"};

    F -- Decision: <br/>Upgrade to Professional API Architecture --> G;

    subgraph "V2: Decoupled API Architecture"
        G(User) -- Interacts with --> H["Frontend<br/>(Streamlit)"];
        H -- API Request --> I["Backend<br/>(FastAPI on Docker)"];
        I -- Returns Answer --> H;
    end

    subgraph "V2: Advanced RAG Pipeline (Backend Logic)"
        I -- Triggers --> J(1. Retrieve: Query Pinecone);
        J -- Returns Top 20 Candidates --> K(2. Re-rank: Cross-Encoder);
        K -- Returns Top 4 Contexts --> L(3. Generate: Prompt Gemini);
        L -- Returns Final Answer --> I;
    end

    subgraph "External Cloud Services"
        J --> M[Pinecone Vector Database];
        L --> N[Google Gemini API];
    end

    style Z fill:#FFCCCC,stroke:#A00,stroke-width:2px
```

## ⭐ Core Features (V2)

- **Decoupled Architecture**: A robust FastAPI backend serves the AI logic, while a responsive Streamlit frontend provides the user interface.
- **Advanced RAG Pipeline**: Implements a state-of-the-art Retrieve-and-Re-rank strategy, using a bi-encoder (paraphrase-multilingual-mpnet-base-v2) for fast retrieval from Pinecone and a Cross-Encoder for high-precision re-ranking.
- **State-of-the-Art Generative AI**: Leverages the Google Gemini API (gemini-2.5-flash-lite) for fluent, accurate, and synthesized answers.
- **Multilingual & Conversational**: Natively supports both English and Malay and maintains conversation history for natural, multi-turn dialogue.
- **Transparent & Debuggable**: Features a live "Developer Mode" toggle in the UI (not shown in GIF, but implemented) that reveals the exact retrieval sources used to generate an answer.
- **Robust Docker Deployment**: The entire two-process application is containerized with Docker and deployed to Hugging Face Spaces for a consistent runtime environment.

## 🛠️ Technology Stack Comparison

| Category | V1 (Self-Hosted Prototype) | V2 (Professional API) |
|----------|----------------------------|----------------------|
| AI & ML | Python, LangChain, Sentence Transformers (MiniLM-L12-v2) | Python, LangChain, Sentence Transformers (Bi- & Cross-encoder), unstructured |
| Backend & UI | Streamlit (Monolithic) | FastAPI (Backend) + Streamlit (Frontend) |
| Vector Store | FAISS (Local File) | Pinecone (Cloud Vector DB) |
| Generative Model | Google flan-t5-small (Self-Hosted) | Google Gemini `gemini-2.5-flash-lite-preview-06-17` (API Integration) |
| Deployment | Hugging Face Spaces (Standard SDK) | Docker, Hugging Face Spaces (Custom Docker) |

## 📁 Project Structure

```
malay-qa-bot-rag/
├── .git/
├── knowledge_base/
│   ├── v1_malay/                 # Malay-only .txt files for V1
│   └── v2_multilingual/          # English & Malay .md files for V2
├── v1_malay_selfhosted/          # Archived V1 monolithic Streamlit app
├── v2_multilingual_api/
│   ├── backend/
│   │   ├── main.py               # The FastAPI backend server
│   │   └── index_knowledge_base.py # Script to populate Pinecone
│   └── frontend/
│       └── app.py                # The Streamlit frontend UI
├── .env                          # Local secrets (in .gitignore)
├── .gitignore
├── Dockerfile                    # Blueprint for the V2 production container
├── requirements.txt              # V2 Python dependencies
├── setup.sh                      # Startup script for the V2 Docker container
└── README.md                     # This file
```

## 🚀 Local Setup & Usage (V2)

### Prerequisites
- Git, Python 3.10+, Docker Desktop

### Installation & Local Workflow

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amirulhazym/malay-qa-bot-rag.git
   cd malay-qa-bot-rag
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv p3env
   .\p3env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up API Keys:** Create a `.env` file in the root directory and add your keys:
   ```
   PINECONE_API_KEY="your-pinecone-key"
   GEMINI_API_KEY="your-gemini-key"
   ```

5. **Populate the Vector Database:** Run the indexing script once:
   ```bash
   python v2_multilingual_api/backend/index_knowledge_base.py
   ```

6. **Run the Application:**
   - Terminal 1 (Backend): `uvicorn v2_multilingual_api.backend.main:app --reload`
   - Terminal 2 (Frontend): `streamlit run v2_multilingual_api/frontend/app.py`

## 💡 Key Challenges & Learnings

- **Architecture Migration**: Successfully refactored the monolithic V1 prototype into a professional, decoupled microservice architecture (V2), demonstrating an understanding of scalable and maintainable system design.
- **LLM Performance Bottlenecks**: Directly experienced and addressed the limitations of small, self-hosted LLMs (mt5-small, flan-t5-small), which consistently failed at answer synthesis. This led to the implementation of a robust fallback mechanism in V1 and the strategic decision to switch to a powerful API-based model (Gemini) in V2 to achieve high-quality generation.
- **Advanced Retrieval Pipeline**: Moved beyond basic similarity search by implementing a state-of-the-art Retrieve-and-Re-rank pipeline in V2. This process uses a fast bi-encoder for initial candidate fetching and a more accurate Cross-Encoder for precise re-ranking, significantly improving the quality of context provided to the LLM.
- **Deployment Complexity**: Solved real-world deployment challenges on Hugging Face Spaces by moving from the standard SDK to a custom Docker container. This involved creating a Dockerfile and a setup.sh startup script to manage the two-process (backend/frontend) application, a critical skill for deploying non-trivial applications.
- **Git & Environment Management**: Navigated and resolved numerous real-world Git issues, including proxy configurations, fixing "unrelated histories," and managing a clean repository structure. Mastered Python environment management (venv, requirements.txt) to overcome persistent library conflicts between Streamlit and PyTorch.

## 🔮 Future Enhancements

- **Robust Evaluation**: Implement a formal evaluation pipeline using frameworks like RAGAs to scientifically measure context retrieval precision, answer faithfulness, and overall relevance.
- **Advanced Retrieval**: Enhance the RAG pipeline with Hybrid Search (sparse + dense vectors) in Pinecone to improve performance on queries with specific keywords or product codes.
- **MLOps Automation**: Create an automated script (update_knowledge_base.py) that can be triggered via a CI/CD pipeline (e.g., GitHub Actions) to periodically scan for changes in the knowledge base and update the Pinecone index automatically.

## 👤 Author

**Amirulhazym**
- LinkedIn: [linkedin.com/in/amirulhazym](https://linkedin.com/in/amirulhazym)
- GitHub: [github.com/amirulhazym](https://github.com/amirulhazym)
- Portfolio: [amirulhazym.framer.ai](https://amirulhazym.framer.ai)

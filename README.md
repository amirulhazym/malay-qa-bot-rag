---
title: AuraCart AI Assistant
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
pinned: false
---

# Advanced Conversational AI for E-Commerce: V1 vs. V2

This repository showcases the evolution of a RAG-based chatbot, demonstrating a clear progression from a self-hosted prototype (v1) to a professional, API-driven conversational AI (v2).

---

## V2: The Professional Multilingual Assistant (API-Driven)

This version represents a modern, production-ready system built on a decoupled architecture with full support for English and Malay.

*   **Architecture (`/v2_multilingual_api`):**
    *   **Backend:** A secure FastAPI server implementing an advanced **Retrieve-and-Re-rank** pipeline. It uses multilingual models, Pinecone for vector storage, and the Google Gemini API for generation.
    *   **Frontend:** A responsive Streamlit chat interface.
*   **Knowledge Base:** `/knowledge_base/v2_multilingual`

---

## V1: The Self-Hosted Prototype (Malay-Only)

This version is the original project, serving as a technical baseline.

*   **Architecture (`/v1_malay_selfhosted`):** A single Streamlit application using a self-hosted `mt5-small` model and a local FAISS index.
*   **Knowledge Base:** `/knowledge_base/v1_malay`


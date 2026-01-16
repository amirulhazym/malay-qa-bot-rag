import os
import pickle
from typing import List, TypedDict, Literal
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

# Load env
load_dotenv()

# --- 1. SETUP AGENT STATE ---
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: bool

# --- 2. INITIALIZE COMPONENTS ---

# LLM
# Using 1.5-flash as the stable endpoint, but prompted as 2.0-flash logic
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Retriever (Hybrid: Chroma + BM25)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

try:
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Load BM25
    with open("bm25_retriever.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
except Exception as e:
    print(f"Warning: Retrievers could not be loaded. Run setup_vector_db.py first. Error: {e}")
    retriever = None

# Tavily
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- 3. DEFINE NODES ---

def retrieve(state: AgentState):
    """
    Retrieve documents from the hybrid retriever.
    """
    print("---RETRIEVE---")
    question = state["question"]
    if retriever:
        documents = retriever.invoke(question)
    else:
        documents = []
    return {"documents": documents, "question": question}

def grade_documents(state: AgentState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    # If no docs found, force web search
    if not documents:
        return {"documents": [], "web_search": True}

    # Grading prompt
    system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    filtered_docs = []
    web_search = False
    
    for d in documents:
        score_prompt = f"Retrieved document: \n\n {d.page_content} \n\n User question: {question}"
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=score_prompt)]
        
        response = llm.invoke(messages)
        grade = response.content.strip().lower()
        
        if "yes" in grade:
            filtered_docs.append(d)
    
    # If we filtered out too many docs, enable web search
    if len(filtered_docs) == 0:
        web_search = True
    
    return {"documents": filtered_docs, "web_search": web_search}

def web_search_node(state: AgentState):
    """
    Web search based on the question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    # Tavily Search
    results = tavily.search(query=question, max_results=3)
    
    web_results = []
    for result in results.get("results", []):
        content = result.get("content")
        url = result.get("url")
        doc = Document(page_content=content, metadata={"source": url})
        web_results.append(doc)
    
    # Append to existing documents
    documents.extend(web_results)
    
    return {"documents": documents}

def generate(state: AgentState):
    """
    Generate answer using RAG.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Context
    context_str = ""
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        context_str += f"[Source ID: {i+1}] (Source: {source})\nContent: {doc.page_content}\n\n"
    
    system_prompt = """You are a helpful assistant. 
    Cite sources using [Source ID] (e.g. [Source ID: 1]). 
    If using web search, cite the URL explicitly in the text.
    Answer the user question based ONLY on the provided context.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context_str}")
    ]
    
    response = llm.invoke(messages)
    return {"generation": response.content}

# --- 4. BUILD GRAPH ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate)

# Add Edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

def decide_to_generate(state):
    if state["web_search"]:
        return "web_search"
    else:
        return "generate"

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

def run_agent(question: str):
    """
    Generator function to stream events to the UI
    """
    inputs = {"question": question, "web_search": False, "documents": []}
    
    # Stream the graph updates
    for output in app.stream(inputs):
        yield output

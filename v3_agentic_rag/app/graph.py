from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

from state import AgentState
from utils import setup_retriever

# --- Configuration ---
LLM_MODEL = "gemini-2.5-flash-lite" 

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

# Initialize Tools
web_search_tool = TavilySearchResults(k=3)

# Initialize Retriever (Global for graph usage)
# Note: In a production app, we might inject this or lazy load it.
retriever = setup_retriever()

# --- Nodes ---

def retrieve(state: AgentState):
    """
    Retrieve documents from vectorstore
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
    Determines whether the retrieved documents are relevant to the question
    """
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
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
    
    # Fallback: if no docs found or all filtered out, trigger web search
    if not filtered_docs:
        print("---DECISION: ALL DOCS IRRELEVANT -> WEB SEARCH---")
        web_search = True
    else:
        print("---DECISION: DOCS RELEVANT---")
        
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state: AgentState):
    """
    Web search based on the question
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    docs = web_search_tool.invoke({"query": question})
    
    web_results = []
    if docs:
        for d in docs:
            content = d.get("content")
            url = d.get("url")
            web_results.append(Document(page_content=content, metadata={"source": url}))
            
    documents.extend(web_results)
    
    return {"documents": documents}

def generate(state: AgentState):
    """
    Generate answer using RAG on retrieved documents
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Format context with Source IDs
    context = ""
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown")
        context += f"[Source ID: {i+1}] (Source: {source})\nContent: {doc.page_content}\n\n"
    
    system_prompt = """You are a helpful assistant. 
    Cite sources using [Source ID] (e.g. [Source ID: 1]). 
    If using web search, cite the URL explicitly in the text.
    Answer the user question based ONLY on the provided context.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}")
    ]
    
    response = llm.invoke(messages)
    return {"generation": response.content}

# --- Graph Definition ---

def decide_to_generate(state):
    """
    Conditional Edge logic
    """
    if state["web_search"]:
        return "web_search_node"
    else:
        return "generate"

def get_app():
    """
    Compiles and returns the LangGraph application
    """
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search_node", web_search)
    workflow.add_node("generate", generate)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search_node": "web_search_node",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

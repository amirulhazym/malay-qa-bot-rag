from typing import List, TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    Represents the state of the agent in the graph.
    """
    question: str
    documents: List[Document]
    generation: str
    web_search: bool

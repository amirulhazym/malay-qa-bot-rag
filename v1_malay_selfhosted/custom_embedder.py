# --- custom_embedder.py ---
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer
import torch
from typing import List
import numpy as np

class MistralDirectEmbeddings(Embeddings):
    """Custom LangChain Embeddings class using Mesolitica's direct .encode()."""
    def __init__(self, model_name: str = "mesolitica/mistral-embedding-191m-8k-contrastive"):
        print(f">> Initializing Custom Embedder: {model_name}")
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f">> Using device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            print(">> Custom embedder model and tokenizer loaded.")
        except Exception as e:
            print(f"!!! ERROR initializing custom embedder: {e}")
            raise # Re-raise the exception

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Helper function to embed a list of texts."""
        if not texts:
            return np.array([])
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=8192 # Use model's max length
            ).to(self.device)

            with torch.no_grad():
                # Assuming model.encode takes tokenized input directly
                embeddings = self.model.encode(inputs['input_ids'], attention_mask=inputs['attention_mask'])

            return embeddings.detach().cpu().numpy()
        except Exception as e:
            print(f"!!! ERROR during custom embedding: {e}")
            # Return empty array or handle error as appropriate
            # Returning empty might cause issues downstream
            # Consider returning None or raising error if needed
            return np.array([])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        print(f">> Custom embed_documents called for {len(texts)} texts.")
        embeddings_np = self._embed(texts)
        # Handle case where embedding failed
        if embeddings_np.size == 0 and len(texts) > 0:
             print("!!! WARNING: embed_documents received empty embeddings.")
             # Return list of empty lists or lists of zeros, matching expected output structure
             return [[0.0] * (self.model.config.hidden_size if hasattr(self.model, 'config') else 768)] * len(texts) # Adjust dimension if needed
        return embeddings_np.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        print(f">> Custom embed_query called for query: '{text[:50]}...'")
        embeddings_np = self._embed([text])
         # Handle case where embedding failed
        if embeddings_np.size == 0:
             print("!!! WARNING: embed_query received empty embeddings.")
             return [0.0] * (self.model.config.hidden_size if hasattr(self.model, 'config') else 768) # Adjust dimension if needed
        return embeddings_np[0].tolist()

# Example Self-Test (optional)
if __name__ == '__main__':
    print("Running custom embedder self-test...")
    embedder = MistralDirectEmbeddings()
    sample_texts = ["Ini ujian.", "Ini adalah ujian kedua."]
    doc_embeddings = embedder.embed_documents(sample_texts)
    query_embedding = embedder.embed_query("Ujian ketiga.")
    print(f"Doc embedding shape: ({len(doc_embeddings)}, {len(doc_embeddings[0]) if doc_embeddings else 'N/A'})")
    print(f"Query embedding shape: ({len(query_embedding)},)")
    print("Self-test finished.")
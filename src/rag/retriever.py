import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from config import config

class DrugInteractionRetriever:
    def __init__(self):
        # Load FAISS index from disk
        print("Loading FAISS index...")
        self.index = faiss.read_index(
            os.path.join(config.FAISS_INDEX_PATH, "index.faiss")
        )
        
        # Load original dataframe from disk
        with open(os.path.join(config.FAISS_INDEX_PATH, "df.pkl"), "rb") as f:
            self.df = pickle.load(f)
        
        # Load the same embedding model used to build the index
        # Must be identical model — otherwise vectors won't match
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        print(f"Retriever ready — {self.index.ntotal} interactions indexed")

    def retrieve(self, query: str, top_k: int = None):
        # Use config value if top_k not specified
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        
        # Convert query text to vector
        query_vector = self.model.encode([query]).astype("float32")
        
        # Search FAISS for top_k closest vectors
        # D = distances, I = indices of matching rows
        D, I = self.index.search(query_vector, top_k)
        
        # Build results list
        results = []
        for distance, idx in zip(D[0], I[0]):
            row = self.df.iloc[idx]
            results.append({
                "drug_1": row["Drug 1"],
                "drug_2": row["Drug 2"],
                "interaction": row["Interaction Description"],
                "text": row["text"],
                # Convert L2 distance to similarity score 0-1
                # Lower distance = higher similarity
                "score": float(1 / (1 + distance))
            })
        
        return results

if __name__ == "__main__":
    retriever = DrugInteractionRetriever()
    
    # Test query
    results = retriever.retrieve("aspirin warfarin interaction")
    
    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Drug 1: {r['drug_1']}")
        print(f"  Drug 2: {r['drug_2']}")
        print(f"  Interaction: {r['interaction'][:100]}...")
        print(f"  Score: {r['score']:.3f}")
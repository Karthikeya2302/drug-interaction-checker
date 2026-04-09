import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from src.rag.loader import load_drug_interactions
from config import config

def build_faiss_index():
    # Load cleaned data
    df = load_drug_interactions()
    
    # Load the embedding model
    # all-MiniLM-L6-v2 converts text into 384-dimensional vectors
    print("Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # Convert all text rows into vectors
    # This is the slow step — 191k rows takes a few minutes
    print("Embedding drug interactions...")
    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        batch_size=64
    )
    
    # Convert to float32 — FAISS requires this format
    embeddings = np.array(embeddings).astype("float32")
    
    # Create FAISS index
    # IndexFlatL2 measures straight line distance between vectors
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add all vectors to the index
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to FAISS index")
    
    # Save the index to disk so we don't rebuild every startup
    os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(config.FAISS_INDEX_PATH, "index.faiss"))
    
    # Save the dataframe alongside so we can retrieve original text later
    with open(os.path.join(config.FAISS_INDEX_PATH, "df.pkl"), "wb") as f:
        pickle.dump(df, f)
    
    print("FAISS index saved successfully")
    return index, df

if __name__ == "__main__":
    build_faiss_index()
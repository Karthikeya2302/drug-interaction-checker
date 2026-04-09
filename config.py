import os
from dotenv import load_dotenv

# Reads your .env file and loads all keys into environment
load_dotenv()

class Config:
    # Your Groq API key — loaded from .env file
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    
    # The LLM model we're using on Groq
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # Sentence transformer model for converting drug names to vectors
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # How many results to fetch from FAISS index per query
    TOP_K_RETRIEVAL: int = 5
    
    # Path to your CSV dataset
    DATA_PATH: str = "data/db_drug_interactions.csv"
    
    # Where to save the FAISS index after building it
    FAISS_INDEX_PATH: str = "faiss_index/"
    
    # MLflow experiment name
    MLFLOW_EXPERIMENT: str = "drug-interaction-checker"

# Single instance — every file imports this one object
config = Config()
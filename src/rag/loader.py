import pandas as pd
from config import config

def load_drug_interactions():
    # Read the CSV file from the path defined in config
    df = pd.read_csv(config.DATA_PATH)
    
    # Print shape so we know how many rows we loaded
    print(f"Loaded {len(df)} drug interactions")
    
    # Make all column names lowercase and strip spaces
    df.columns = [col.strip() for col in df.columns]
    
    # Drop any rows where drug names or description are missing
    df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])
    
    # Remove duplicate pairs
    df = df.drop_duplicates(subset=["Drug 1", "Drug 2"])
    
    # Combine all three columns into one text string
    # This is what gets embedded into FAISS vectors
    df["text"] = (
        "Drug A: " + df["Drug 1"].str.strip() + " | " +
        "Drug B: " + df["Drug 2"].str.strip() + " | " +
        "Interaction: " + df["Interaction Description"].str.strip()
    )
    
    print(f"After cleaning: {len(df)} interactions")
    
    return df

if __name__ == "__main__":
    df = load_drug_interactions()
    print(df["text"].head(3))
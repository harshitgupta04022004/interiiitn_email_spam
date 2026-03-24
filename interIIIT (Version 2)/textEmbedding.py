import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pandas as pd


def doEmbedding(df):
    """
    Generate text embeddings using SentenceTransformer model.
    
    Args:
        df: DataFrame with 'text' column containing concatenated email text
        
    Returns:
        DataFrame with text embeddings added as columns and 'text' column removed
    """
    print("\n   Loading SentenceTransformer model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Prepare texts
    texts = df['text'].fillna("").tolist()
    print(f"   Encoding {len(texts)} text samples...")
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32  # Process in batches for efficiency
    )
    
    # Optional: Normalize embeddings (uncomment if needed)
    # embeddings = normalize(embeddings)
    
    # Create embedding columns
    n_dims = embeddings.shape[1]
    embedding_cols = [f'text_emb_{i}' for i in range(n_dims)]
    
    print(f"   Generated {n_dims}-dimensional embeddings")
    
    # Create DataFrame with embeddings
    text_feats_df = pd.DataFrame(embeddings, index=df.index, columns=embedding_cols)
    
    # Concatenate with original dataframe (remove text column)
    df = pd.concat([df.drop(columns=['text']), text_feats_df], axis=1)
    
    return df

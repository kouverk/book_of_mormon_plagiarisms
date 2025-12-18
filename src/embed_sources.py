"""
Generate embeddings for source texts and save to FAISS index.
Reads from database, embeds with OpenAI, saves FAISS index.
"""

import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

from config import EMBEDDINGS_DIR
from database import get_db

client = OpenAI()
MODEL = "text-embedding-3-large"


def embed_openai(texts: list[str]) -> np.ndarray:
    """Embed texts using OpenAI API."""
    response = client.embeddings.create(
        input=texts,
        model=MODEL
    )
    return np.array([item.embedding for item in response.data])


def main():
    db = get_db()

    # Get all source passages from database
    passages = db.get_source_passages()
    print(f"Found {len(passages)} passages in database")

    if not passages:
        print("No passages found. Run text_parser.py first.")
        return

    texts = [p['text'] for p in passages]
    meta = pd.DataFrame([{
        'id': p['id'],
        'source': p['source'],
        'source_name': p['source_name'],
        'location': p['location'],
        'text': p['text'][:500]  # Truncate for CSV
    } for p in passages])

    print(f"Embedding {len(texts)} passages with {MODEL}...")

    embeddings = []
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        # Truncate very long texts to avoid API limits
        batch = [t[:8000] for t in batch]
        batch_embeds = embed_openai(batch)
        embeddings.append(batch_embeds)

    embeddings = np.vstack(embeddings).astype('float32')

    print(f"Saving FAISS index ({embeddings.shape})...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine
    index.add(embeddings)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(EMBEDDINGS_DIR / "source_index.faiss"))
    meta.to_csv(EMBEDDINGS_DIR / "source_meta.csv", index=False)

    print(f"Saved index to {EMBEDDINGS_DIR / 'source_index.faiss'}")
    print(f"Saved metadata to {EMBEDDINGS_DIR / 'source_meta.csv'}")


if __name__ == "__main__":
    main()

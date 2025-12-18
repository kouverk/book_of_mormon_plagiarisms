"""
Generate embeddings for Book of Mormon verses and save to FAISS index.
Reads from database, embeds with OpenAI, saves FAISS index.
"""

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

    # Get all BOM verses from database
    verses = db.get_all_bom_verses()
    print(f"Found {len(verses)} BOM verses in database")

    if not verses:
        print("No verses found. Run text_parser.py bom first.")
        return

    texts = [v['text'] for v in verses]
    meta = pd.DataFrame([{
        'id': v['id'],
        'book': v['book'],
        'chapter': v['chapter'],
        'verse': v['verse'],
        'reference': f"{v['book']} {v['chapter']}:{v['verse']}",
        'text': v['text'][:500]  # Truncate for CSV
    } for v in verses])

    print(f"Embedding {len(texts)} verses with {MODEL}...")

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
    faiss.write_index(index, str(EMBEDDINGS_DIR / "bom_index.faiss"))
    meta.to_csv(EMBEDDINGS_DIR / "bom_meta.csv", index=False)

    print(f"Saved index to {EMBEDDINGS_DIR / 'bom_index.faiss'}")
    print(f"Saved metadata to {EMBEDDINGS_DIR / 'bom_meta.csv'}")


if __name__ == "__main__":
    main()

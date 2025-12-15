import os
import openai
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "text-embedding-3-large"

def embed_openai(texts):
    response = openai.Embedding.create(
        input=texts,
        model=MODEL
    )
    return np.array([item["embedding"] for item in response["data"]])

def load_sources(source_dir="data/source_texts"):
    texts = []
    meta = []
    for filename in Path(source_dir).glob("*.txt"):
        source_name = filename.stem
        with open(filename, "r") as f:
            paragraphs = [p.strip() for p in f.readlines() if p.strip()]
            for i, para in enumerate(paragraphs):
                texts.append(para)
                meta.append({
                    "source": source_name,
                    "index": i,
                    "text": para
                })
    return texts, pd.DataFrame(meta)

def main():
    texts, meta = load_sources()
    print(f"🔍 Embedding {len(texts)} paragraphs from all source texts...")

    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeds = embed_openai(batch)
        embeddings.append(batch_embeds)

    embeddings = np.vstack(embeddings)

    print("✅ Saving FAISS index and metadata...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, "embeddings/source_index.faiss")
    meta.to_csv("embeddings/source_meta.csv", index=False)

if __name__ == "__main__":
    main()


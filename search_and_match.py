import openai
import faiss
import numpy as np
import pandas as pd
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-large"

def embed_verse(text):
    response = openai.Embedding.create(
        input=[text],
        model=MODEL
    )
    return np.array(response["data"][0]["embedding"]).reshape(1, -1)

def search_top_matches(query_text, k=5):
    index = faiss.read_index("embeddings/source_index.faiss")
    meta = pd.read_csv("embeddings/source_meta.csv")
    
    query_vector = embed_verse(query_text)
    D, I = index.search(query_vector, k)
    
    matches = meta.iloc[I[0]].copy()
    matches["score"] = D[0]
    return matches.sort_values("score")

# 🔬 Sample usage
if __name__ == "__main__":
    verse = input("Paste a Book of Mormon verse:\n> ")
    results = search_top_matches(verse, k=5)
    print("\nTop matches:")
    for _, row in results.iterrows():
        print(f"\n📘 Source: {row['source']} (Paragraph {row['index']})")
        print(f"Text: {row['text']}")
        print(f"Score: {row['score']:.4f}")


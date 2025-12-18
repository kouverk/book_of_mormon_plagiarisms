"""
Search and match BOM verses against source texts using pre-computed embeddings.
No API calls needed - uses FAISS indices for fast similarity search.
"""

import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import EMBEDDINGS_DIR
from database import get_db


def load_indices():
    """Load both FAISS indices and metadata."""
    bom_index = faiss.read_index(str(EMBEDDINGS_DIR / "bom_index.faiss"))
    source_index = faiss.read_index(str(EMBEDDINGS_DIR / "source_index.faiss"))
    bom_meta = pd.read_csv(EMBEDDINGS_DIR / "bom_meta.csv")
    source_meta = pd.read_csv(EMBEDDINGS_DIR / "source_meta.csv")
    return bom_index, source_index, bom_meta, source_meta


def search_verse_matches(verse_idx: int, bom_index, source_index, source_meta, k=5):
    """
    Find top-k source matches for a BOM verse using pre-computed embeddings.
    No API calls - pure vector similarity.
    """
    # Get the BOM verse embedding from the index
    bom_embedding = bom_index.reconstruct(verse_idx).reshape(1, -1)

    # Search source index
    D, I = source_index.search(bom_embedding, k)

    matches = source_meta.iloc[I[0]].copy()
    matches["score"] = D[0]
    return matches


def find_all_matches(min_score=0.4, top_k=5, limit=None):
    """
    Find matches for all BOM verses against source texts.
    Returns DataFrame with all matches above min_score.
    """
    bom_index, source_index, bom_meta, source_meta = load_indices()

    all_matches = []
    num_verses = len(bom_meta) if limit is None else min(limit, len(bom_meta))

    print(f"Searching {num_verses} BOM verses for matches (min_score={min_score})...")

    for idx in tqdm(range(num_verses)):
        verse_info = bom_meta.iloc[idx]
        matches = search_verse_matches(idx, bom_index, source_index, source_meta, k=top_k)

        for _, match in matches.iterrows():
            if match['score'] >= min_score:
                all_matches.append({
                    'bom_id': verse_info['id'],
                    'bom_reference': verse_info['reference'],
                    'bom_text': verse_info['text'],
                    'source': match['source'],
                    'source_location': match['location'],
                    'source_text': match['text'],
                    'score': match['score']
                })

    df = pd.DataFrame(all_matches)
    if len(df) > 0:
        df = df.sort_values('score', ascending=False)

    print(f"Found {len(df)} matches above {min_score}")
    return df


def show_top_matches(n=20, min_score=0.4):
    """Display top n matches across all verses."""
    df = find_all_matches(min_score=min_score, top_k=3)

    if len(df) == 0:
        print("No matches found above threshold")
        return

    print(f"\nTop {n} matches:")
    print("=" * 80)

    for i, (_, row) in enumerate(df.head(n).iterrows()):
        print(f"\n[{i+1}] Score: {row['score']:.3f}")
        print(f"BOM: {row['bom_reference']}")
        print(f"  \"{row['bom_text'][:150]}...\"")
        print(f"Source: {row['source']} - {row['source_location']}")
        print(f"  \"{row['source_text'][:150]}...\"")


def interactive_search():
    """Interactive mode - search by verse reference."""
    from openai import OpenAI
    client = OpenAI()

    source_index = faiss.read_index(str(EMBEDDINGS_DIR / "source_index.faiss"))
    source_meta = pd.read_csv(EMBEDDINGS_DIR / "source_meta.csv")

    print("Interactive search mode. Paste a verse or type 'quit' to exit.")

    while True:
        text = input("\n> ").strip()
        if text.lower() in ('quit', 'exit', 'q'):
            break
        if not text:
            continue

        # Embed the query text
        response = client.embeddings.create(input=[text], model="text-embedding-3-large")
        query_vec = np.array(response.data[0].embedding).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vec)

        # Search
        D, I = source_index.search(query_vec, 5)
        matches = source_meta.iloc[I[0]].copy()
        matches["score"] = D[0]

        print("\nTop matches:")
        for _, row in matches.iterrows():
            print(f"\n  [{row['score']:.3f}] {row['source']} - {row['location']}")
            print(f"  {row['text'][:200]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "top":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            show_top_matches(n=n)
        elif cmd == "all":
            min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
            df = find_all_matches(min_score=min_score)
            df.to_csv("results/matches.csv", index=False)
            print(f"Saved to results/matches.csv")
        elif cmd == "interactive":
            interactive_search()
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: top [n], all [min_score], interactive")
    else:
        print("Book of Mormon Parallel Detection")
        print("Usage:")
        print("  python search_and_match.py top [n]           - Show top n matches")
        print("  python search_and_match.py all [min_score]   - Find all matches, save to CSV")
        print("  python search_and_match.py interactive       - Paste text to search")

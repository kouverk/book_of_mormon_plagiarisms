import json
from utils import *

# Load texts
with open("data/bom_verses.txt") as f:
    bom_verses = [line.strip() for line in f if line.strip()]

with open("data/view_of_hebrews_chunks.txt") as f:
    view_chunks = [line.strip() for line in f if line.strip()]

# Embed source text if needed
source_embeddings = embed_texts(view_chunks)
save_faiss_index(source_embeddings)

# Embed BoM verses
bom_embeddings = embed_texts(bom_verses)

# Load FAISS index
index = load_faiss_index()

results = []

for i, (verse, vec) in enumerate(zip(bom_verses, bom_embeddings)):
    print(f"🔍 Analyzing verse {i + 1}/{len(bom_verses)}")
    top_indices = query_index(index, vec, k=5)
    for j in top_indices:
        candidate = view_chunks[j]
        gpt_result = ask_gpt(verse, candidate)
        results.append({
            "verse": verse,
            "candidate": candidate,
            "gpt_output": gpt_result
        })

# Save results
with open("results/matches.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Matching complete.")

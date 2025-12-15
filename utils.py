"""
Utility functions for textual parallel detection.

Includes:
- Embedding functions (Sentence-BERT, OpenAI)
- FAISS index operations
- Lexical similarity (n-gram, TF-IDF, Jaccard)
- LLM prompting for semantic analysis
"""

import re
import os
import json
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

from config import OPENAI_API_KEY, EMBEDDINGS_DIR, get_setting


# ==================== Initialization ====================

# Set OpenAI API key from environment
openai.api_key = OPENAI_API_KEY

# Lazy loading for embedding model
_sbert_model = None

def get_sbert_model():
    """Get Sentence-BERT model (lazy loaded)."""
    global _sbert_model
    if _sbert_model is None:
        model_name = get_setting('sentence_bert_model', 'all-MiniLM-L6-v2')
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model


# ==================== Text Preprocessing ====================

def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split on non-alphanumeric."""
    return re.findall(r'\b\w+\b', text.lower())


def get_ngrams(tokens: List[str], n: int) -> Set[tuple]:
    """Generate n-grams from token list."""
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stopwords."""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
        'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'then'
    }
    return [t for t in tokens if t not in stopwords]


# ==================== Lexical Similarity ====================

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute n-gram overlap (Jaccard) between two texts.
    Default is trigrams (n=3).
    """
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    ngrams1 = get_ngrams(tokens1, n)
    ngrams2 = get_ngrams(tokens2, n)

    return jaccard_similarity(ngrams1, ngrams2)


def multi_ngram_overlap(text1: str, text2: str, ns: List[int] = [2, 3, 4]) -> Dict[str, float]:
    """Compute overlap for multiple n-gram sizes."""
    return {f'{n}gram': ngram_overlap(text1, text2, n) for n in ns}


def find_shared_phrases(text1: str, text2: str, min_length: int = 3, max_length: int = 8) -> List[str]:
    """
    Find exact phrase matches between two texts.
    Returns list of shared phrases (min_length to max_length words).
    """
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    shared = []
    for n in range(max_length, min_length - 1, -1):  # Start with longer phrases
        ngrams1 = get_ngrams(tokens1, n)
        ngrams2 = get_ngrams(tokens2, n)
        matches = ngrams1 & ngrams2

        for match in matches:
            phrase = ' '.join(match)
            # Don't add if it's a substring of an already-found longer phrase
            if not any(phrase in existing for existing in shared):
                shared.append(phrase)

    return shared


def word_overlap(text1: str, text2: str, remove_stops: bool = True) -> float:
    """
    Simple word overlap (Jaccard on unique words).
    Optionally removes stopwords.
    """
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if remove_stops:
        tokens1 = remove_stopwords(tokens1)
        tokens2 = remove_stopwords(tokens2)

    return jaccard_similarity(set(tokens1), set(tokens2))


# ==================== TF-IDF Similarity ====================

class TFIDFMatcher:
    """TF-IDF based similarity matcher."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=10000,
            stop_words='english'
        )
        self.corpus_vectors = None
        self.corpus_texts = []

    def fit(self, texts: List[str]):
        """Fit the vectorizer on a corpus."""
        self.corpus_texts = texts
        self.corpus_vectors = self.vectorizer.fit_transform(texts)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF cosine similarity between two texts."""
        if self.corpus_vectors is None:
            # Fit on just these two texts
            vectors = self.vectorizer.fit_transform([text1, text2])
            return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # Transform both texts using fitted vectorizer
        vec1 = self.vectorizer.transform([text1])
        vec2 = self.vectorizer.transform([text2])
        return cosine_similarity(vec1, vec2)[0][0]

    def find_similar(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find most similar texts from the corpus."""
        if self.corpus_vectors is None:
            raise ValueError("Must call fit() first")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.corpus_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]


def tfidf_similarity(text1: str, text2: str) -> float:
    """Quick TF-IDF similarity between two texts."""
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    try:
        vectors = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except ValueError:
        # Empty vocabulary (texts have no valid terms)
        return 0.0


# ==================== Embedding Functions ====================

def embed_texts_sbert(texts: List[str]) -> np.ndarray:
    """Embed texts using Sentence-BERT."""
    model = get_sbert_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100)


def embed_texts_openai(texts: List[str], model: str = "text-embedding-3-large") -> np.ndarray:
    """Embed texts using OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return np.array([item.embedding for item in response.data])


def embed_texts(texts: List[str], use_openai: bool = False) -> np.ndarray:
    """
    Embed texts using configured model.
    Default is Sentence-BERT (free, local).
    Set use_openai=True for OpenAI embeddings (paid, higher quality).
    """
    if use_openai:
        return embed_texts_openai(texts)
    return embed_texts_sbert(texts)


def embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# ==================== FAISS Index Operations ====================

def save_faiss_index(embeddings: np.ndarray, path: str = None):
    """Save embeddings to a FAISS index."""
    if path is None:
        path = str(EMBEDDINGS_DIR / "faiss_index.faiss")

    embeddings = np.ascontiguousarray(embeddings.astype('float32'))
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized vectors)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, path)
    return path


def load_faiss_index(path: str = None):
    """Load a FAISS index from disk."""
    if path is None:
        path = str(EMBEDDINGS_DIR / "faiss_index.faiss")
    return faiss.read_index(path)


def query_index(index, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query FAISS index for similar items.
    Returns (distances, indices) arrays.
    """
    query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]


# ==================== LLM Analysis ====================

MATCH_ANALYSIS_PROMPT = """You are analyzing potential textual parallels between the Book of Mormon and a 19th-century source text.

Compare these two passages and determine if there's evidence of literary borrowing, thematic similarity, or textual dependence.

## Book of Mormon Passage:
"{bom_text}"

## Source Text Passage ({source_name}):
"{source_text}"

## Analysis Instructions:
1. Look for direct phrase copying, paraphrasing, or shared concepts
2. Consider theological, narrative, or stylistic similarities
3. Note any shared unusual terminology or phrasing
4. Assess whether similarities could be coincidental or indicate dependence

## Response Format (JSON):
{{
    "match_score": <0.0 to 1.0>,
    "match_type": "<direct_copy|paraphrase|thematic|stylistic|none>",
    "confidence": "<high|medium|low>",
    "explanation": "<brief explanation of the connection or lack thereof>",
    "shared_elements": ["<list of specific shared phrases, concepts, or patterns>"]
}}

Respond with only valid JSON."""


def analyze_match_with_llm(bom_text: str, source_text: str, source_name: str,
                           model: str = None) -> Dict:
    """
    Use LLM to analyze a potential match between texts.
    Returns parsed JSON response with match details.
    """
    from openai import OpenAI
    client = OpenAI()

    if model is None:
        model = get_setting('llm_model_tier1', 'gpt-4o-mini')

    prompt = MATCH_ANALYSIS_PROMPT.format(
        bom_text=bom_text,
        source_text=source_text,
        source_name=source_name
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a literary analyst specializing in textual criticism and source analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )

    content = response.choices[0].message.content

    # Parse JSON response
    try:
        # Handle markdown code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]

        result = json.loads(content.strip())
        result['llm_model'] = model
        return result
    except json.JSONDecodeError:
        # Return raw response if JSON parsing fails
        return {
            'match_score': 0.0,
            'match_type': 'unknown',
            'confidence': 'low',
            'explanation': content,
            'shared_elements': [],
            'llm_model': model,
            'parse_error': True
        }


# Legacy function for backwards compatibility
def ask_gpt(bom_verse: str, source_paragraph: str) -> str:
    """
    Legacy function - use analyze_match_with_llm for new code.
    """
    result = analyze_match_with_llm(bom_verse, source_paragraph, "Unknown Source")
    return f"match_score: {result.get('match_score', 0.0)}\nmatch_explanation: {result.get('explanation', '')}"


# ==================== Composite Scoring ====================

def compute_composite_score(
    ngram_score: float = 0.0,
    tfidf_score: float = 0.0,
    embedding_score: float = 0.0,
    llm_score: float = 0.0,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute weighted composite score from multiple methods.
    """
    if weights is None:
        weights = {
            'ngram': 0.15,
            'tfidf': 0.15,
            'embedding': 0.30,
            'llm': 0.40
        }

    score = (
        weights.get('ngram', 0) * ngram_score +
        weights.get('tfidf', 0) * tfidf_score +
        weights.get('embedding', 0) * embedding_score +
        weights.get('llm', 0) * llm_score
    )

    return min(1.0, max(0.0, score))


def classify_match_type(
    ngram_score: float,
    embedding_score: float,
    shared_phrases: List[str]
) -> str:
    """
    Classify the type of match based on scores.
    """
    # High lexical + high semantic = direct copy
    if ngram_score > 0.3 and len(shared_phrases) >= 2:
        return 'direct_copy'

    # High semantic + low lexical = paraphrase
    if embedding_score > 0.85 and ngram_score < 0.15:
        return 'paraphrase'

    # Moderate semantic + low lexical = thematic
    if embedding_score > 0.7 and ngram_score < 0.1:
        return 'thematic'

    # Low scores overall
    if embedding_score < 0.5:
        return 'none'

    return 'unknown'


if __name__ == "__main__":
    # Test the utilities
    text1 = "And it came to pass that I, Nephi, did go forth into the wilderness"
    text2 = "And it came to pass in those days that the army did march into the wilderness"

    print("Testing lexical similarity:")
    print(f"  Word overlap: {word_overlap(text1, text2):.3f}")
    print(f"  Trigram overlap: {ngram_overlap(text1, text2, 3):.3f}")
    print(f"  TF-IDF similarity: {tfidf_similarity(text1, text2):.3f}")
    print(f"  Shared phrases: {find_shared_phrases(text1, text2)}")

    print("\nTesting embeddings:")
    embs = embed_texts_sbert([text1, text2])
    print(f"  Embedding similarity: {embedding_similarity(embs[0], embs[1]):.3f}")

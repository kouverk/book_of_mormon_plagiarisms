"""
LLM Classification of Matches

Runs GPT-4o-mini on all matches to:
1. Classify match type (direct_copy, paraphrase, thematic, stylistic, none)
2. Assign confidence (high, medium, low)
3. Generate explanations
4. Add appropriate tags

Cost estimate: ~$3 for 63,000 matches at ~100 tokens/match
"""

import json
import time
from typing import Dict, List, Optional
from tqdm import tqdm
from openai import OpenAI

from database import get_db, OFFICIAL_TAGS
from config import get_setting


client = OpenAI()
MODEL = "gpt-4o-mini"

# Source name lookup
SOURCE_NAMES = {
    'kjv_bible': 'King James Bible',
    'view_of_hebrews': 'View of the Hebrews (1825)',
    'the_late_war': 'The Late War (1816)',
    'first_book_of_napoleon': 'First Book of Napoleon (1809)',
}


CLASSIFICATION_PROMPT = """Analyze this potential textual parallel between the Book of Mormon and a 19th-century source.

## Book of Mormon ({bom_reference}):
"{bom_text}"

## Source ({source_name}, {source_location}):
"{source_text}"

## Instructions:
Determine if there's evidence of literary borrowing. Consider:
- Direct phrase copying or near-verbatim matches
- Paraphrased ideas (same concept, different words)
- Thematic parallels (shared theological/narrative concepts)
- Stylistic similarities (KJV-style language patterns)
- Whether this could be coincidental biblical language

## Response (JSON only):
{{
    "match_type": "direct_copy|paraphrase|thematic|stylistic|none",
    "confidence": "high|medium|low",
    "explanation": "<1-2 sentence explanation>",
    "is_biblical": <true if both are quoting the Bible>,
    "key_parallels": ["<specific shared phrases or concepts>"]
}}"""


def classify_match(match: Dict) -> Dict:
    """Classify a single match using GPT-4o-mini."""
    source_key = match.get('source_passage_id', '').split('.')[0]
    source_name = SOURCE_NAMES.get(source_key, source_key)

    bom_ref = f"{match.get('book', '')} {match.get('chapter', '')}:{match.get('verse', '')}"

    prompt = CLASSIFICATION_PROMPT.format(
        bom_reference=bom_ref,
        bom_text=match.get('verse_text', '')[:1000],  # Truncate long texts
        source_name=source_name,
        source_location=match.get('location', 'Unknown'),
        source_text=match.get('source_text', '')[:1000]
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a literary analyst. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )

        content = response.choices[0].message.content

        # Parse JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]

        result = json.loads(content.strip())
        result['success'] = True
        return result

    except json.JSONDecodeError as e:
        return {
            'match_type': 'unknown',
            'confidence': 'low',
            'explanation': f'JSON parse error: {str(e)}',
            'is_biblical': False,
            'key_parallels': [],
            'success': False
        }
    except Exception as e:
        return {
            'match_type': 'unknown',
            'confidence': 'low',
            'explanation': f'API error: {str(e)}',
            'is_biblical': False,
            'key_parallels': [],
            'success': False
        }


def classify_all_matches(
    min_score: float = 0.5,
    limit: Optional[int] = None,
    batch_size: int = 50,
    skip_classified: bool = True
):
    """
    Classify all matches above a score threshold.

    Args:
        min_score: Only classify matches with embedding_score >= this
        limit: Max number of matches to process (None for all)
        batch_size: How often to save progress
        skip_classified: Skip matches that already have LLM classification
    """
    db = get_db()

    # Get matches to classify
    with db.get_connection() as conn:
        if skip_classified:
            query = """
                SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                       sp.text as source_text, sp.source, sp.source_name, sp.location
                FROM matches m
                JOIN bom_verses bv ON m.bom_verse_id = bv.id
                JOIN source_passages sp ON m.source_passage_id = sp.id
                WHERE m.embedding_score >= ?
                  AND (m.llm_explanation IS NULL OR m.llm_explanation = '')
                ORDER BY m.embedding_score DESC
            """
        else:
            query = """
                SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                       sp.text as source_text, sp.source, sp.source_name, sp.location
                FROM matches m
                JOIN bom_verses bv ON m.bom_verse_id = bv.id
                JOIN source_passages sp ON m.source_passage_id = sp.id
                WHERE m.embedding_score >= ?
                ORDER BY m.embedding_score DESC
            """

        if limit:
            query += f" LIMIT {limit}"

        rows = conn.execute(query, (min_score,)).fetchall()
        matches = [dict(row) for row in rows]

    print(f"Found {len(matches)} matches to classify (min_score={min_score})")

    if not matches:
        print("No matches to classify!")
        return

    # Process matches
    classified = 0
    errors = 0
    tags_to_add = []

    print(f"Classifying with {MODEL}...")

    for i, match in enumerate(tqdm(matches)):
        result = classify_match(match)

        if not result.get('success', False):
            errors += 1
            continue

        # Update the match in database
        with db.get_connection() as conn:
            conn.execute("""
                UPDATE matches SET
                    match_type = ?,
                    confidence = ?,
                    llm_explanation = ?,
                    llm_model = ?,
                    llm_score = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                result.get('match_type', 'unknown'),
                result.get('confidence', 'low'),
                result.get('explanation', ''),
                MODEL,
                1.0 if result.get('match_type') in ('direct_copy', 'paraphrase') else
                0.7 if result.get('match_type') == 'thematic' else
                0.3 if result.get('match_type') == 'stylistic' else 0.0,
                match['id']
            ))

        # Collect tags
        match_id = match['id']

        # Add match type tag
        match_type = result.get('match_type', 'unknown')
        if match_type in OFFICIAL_TAGS:
            tags_to_add.append({'match_id': match_id, 'tag': match_type, 'source': 'llm'})

        # Add biblical tag if applicable
        if result.get('is_biblical'):
            tags_to_add.append({'match_id': match_id, 'tag': 'biblical_parallel', 'source': 'llm'})

        # Add confidence tag
        if result.get('confidence') == 'high':
            tags_to_add.append({'match_id': match_id, 'tag': 'high_confidence', 'source': 'llm'})

        classified += 1

        # Batch save tags
        if len(tags_to_add) >= batch_size * 3:
            db.add_tags_batch(tags_to_add)
            tags_to_add = []

        # Rate limiting - be nice to the API
        if (i + 1) % 100 == 0:
            time.sleep(1)

    # Save remaining tags
    if tags_to_add:
        db.add_tags_batch(tags_to_add)

    print(f"\nClassified {classified} matches ({errors} errors)")

    # Show summary
    print("\nTag counts:")
    tag_counts = db.get_tag_counts()
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {tag}: {count}")


def classify_top_matches(n: int = 100):
    """Quick classification of just the top N matches by score."""
    classify_all_matches(min_score=0.0, limit=n)


def show_classified_examples(n: int = 10):
    """Show examples of classified matches."""
    db = get_db()

    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                   sp.text as source_text, sp.source_name, sp.location
            FROM matches m
            JOIN bom_verses bv ON m.bom_verse_id = bv.id
            JOIN source_passages sp ON m.source_passage_id = sp.id
            WHERE m.llm_explanation IS NOT NULL AND m.llm_explanation != ''
            ORDER BY m.embedding_score DESC
            LIMIT ?
        """, (n,)).fetchall()

    print(f"\nTop {n} classified matches:\n")
    print("=" * 80)

    for row in rows:
        match = dict(row)
        print(f"\n[{match['match_type'].upper()}] Score: {match['embedding_score']:.3f} | Confidence: {match['confidence']}")
        print(f"BOM: {match['book']} {match['chapter']}:{match['verse']}")
        print(f"  \"{match['verse_text'][:150]}...\"")
        print(f"Source: {match['source_name']} - {match['location']}")
        print(f"  \"{match['source_text'][:150]}...\"")
        print(f"Explanation: {match['llm_explanation']}")
        print("-" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "all":
            # Classify all matches above 0.5 score
            min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
            classify_all_matches(min_score=min_score)

        elif cmd == "top":
            # Just classify top N
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            classify_top_matches(n)

        elif cmd == "show":
            # Show classified examples
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            show_classified_examples(n)

        else:
            print(f"Unknown command: {cmd}")
            print("Commands: all [min_score], top [n], show [n]")
    else:
        print("LLM Match Classifier")
        print("Usage:")
        print("  python classify_matches.py all [min_score]  - Classify all matches (default 0.5)")
        print("  python classify_matches.py top [n]          - Classify top N matches")
        print("  python classify_matches.py show [n]         - Show classified examples")

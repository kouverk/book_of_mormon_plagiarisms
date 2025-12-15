"""
SQLite database module for storing and querying textual parallel matches.

Schema supports:
- Incremental match additions
- Multiple detection methods per match
- Manual curation (confirmed/rejected/needs_review)
- Analytics queries
- Future React frontend via API
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from enum import Enum

from config import DATABASE_PATH, DATA_DIR


class MatchType(Enum):
    DIRECT_COPY = "direct_copy"
    PARAPHRASE = "paraphrase"
    THEMATIC = "thematic"
    NAME_BORROW = "name_borrow"
    STRUCTURAL = "structural"
    STYLISTIC = "stylistic"
    UNKNOWN = "unknown"


class ReviewStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


SCHEMA = """
-- Book of Mormon verses (the query side)
CREATE TABLE IF NOT EXISTS bom_verses (
    id TEXT PRIMARY KEY,              -- "1nephi.1.1"
    book TEXT NOT NULL,               -- "1 Nephi"
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    text TEXT NOT NULL,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bom_book ON bom_verses(book);
CREATE INDEX IF NOT EXISTS idx_bom_chapter ON bom_verses(book, chapter);

-- Source text passages (what we're comparing against)
CREATE TABLE IF NOT EXISTS source_passages (
    id TEXT PRIMARY KEY,              -- "voh.ch3.p5"
    source TEXT NOT NULL,             -- "view_of_hebrews"
    source_name TEXT,                 -- "View of the Hebrews"
    location TEXT,                    -- "Chapter 3, Paragraph 5"
    page INTEGER,                     -- Page number if available
    text TEXT NOT NULL,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_source ON source_passages(source);

-- Detected matches (many-to-many between verses and passages)
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bom_verse_id TEXT NOT NULL,
    source_passage_id TEXT NOT NULL,

    -- Scores from different detection methods (0.0 to 1.0)
    ngram_score REAL,
    tfidf_score REAL,
    embedding_score REAL,
    llm_score REAL,
    final_score REAL,

    -- Classification
    match_type TEXT DEFAULT 'unknown',     -- direct_copy, paraphrase, thematic, etc.
    confidence TEXT DEFAULT 'low',         -- high, medium, low

    -- LLM analysis
    llm_explanation TEXT,
    llm_model TEXT,                        -- Which model produced the analysis

    -- Detection metadata
    detection_methods TEXT,                -- JSON array of methods that found this
    matched_phrases TEXT,                  -- JSON array of specific phrase matches

    -- Manual curation
    review_status TEXT DEFAULT 'pending',  -- pending, confirmed, rejected, needs_review
    reviewed_by TEXT,
    reviewed_at TIMESTAMP,
    review_notes TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (bom_verse_id) REFERENCES bom_verses(id),
    FOREIGN KEY (source_passage_id) REFERENCES source_passages(id),
    UNIQUE(bom_verse_id, source_passage_id)
);

CREATE INDEX IF NOT EXISTS idx_match_verse ON matches(bom_verse_id);
CREATE INDEX IF NOT EXISTS idx_match_source ON matches(source_passage_id);
CREATE INDEX IF NOT EXISTS idx_match_type ON matches(match_type);
CREATE INDEX IF NOT EXISTS idx_match_status ON matches(review_status);
CREATE INDEX IF NOT EXISTS idx_match_score ON matches(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_match_confidence ON matches(confidence);

-- Named entities extracted from texts
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_type TEXT NOT NULL,              -- 'bom' or source name
    text_id TEXT NOT NULL,                -- verse or passage id
    entity_text TEXT NOT NULL,            -- "Nephi", "Zarahemla"
    entity_type TEXT NOT NULL,            -- PERSON, PLACE, ORG, etc.
    start_pos INTEGER,
    end_pos INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entity_text ON entities(entity_text);
CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);

-- Track processing runs for reproducibility
CREATE TABLE IF NOT EXISTS processing_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,               -- 'embedding', 'ngram', 'llm', etc.
    source TEXT,                          -- Which source text was processed
    config_snapshot TEXT,                 -- JSON of config at time of run
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    records_processed INTEGER,
    status TEXT DEFAULT 'running',        -- running, completed, failed
    error_message TEXT
);

-- Calibration data: known parallels for validation
CREATE TABLE IF NOT EXISTS known_parallels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bom_reference TEXT NOT NULL,          -- "1 Nephi 1:1"
    source TEXT NOT NULL,                 -- "view_of_hebrews"
    source_reference TEXT,                -- "Chapter 3"
    parallel_type TEXT,                   -- Type of parallel
    description TEXT,                     -- Human description of the parallel
    citation TEXT,                        -- Where this parallel was documented (e.g., "CES Letter")
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    """SQLite database interface for the textual parallel detection project."""

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database with schema."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ==================== BOM Verses ====================

    def insert_bom_verse(self, id: str, book: str, chapter: int, verse: int, text: str) -> None:
        """Insert a Book of Mormon verse."""
        word_count = len(text.split())
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO bom_verses (id, book, chapter, verse, text, word_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (id, book, chapter, verse, text, word_count)
            )

    def insert_bom_verses_batch(self, verses: List[Dict]) -> int:
        """Batch insert BOM verses. Returns count inserted."""
        with self.get_connection() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO bom_verses (id, book, chapter, verse, text, word_count)
                   VALUES (:id, :book, :chapter, :verse, :text, :word_count)""",
                verses
            )
            return len(verses)

    def get_bom_verse(self, id: str) -> Optional[Dict]:
        """Get a single BOM verse by ID."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM bom_verses WHERE id = ?", (id,)).fetchone()
            return dict(row) if row else None

    def get_all_bom_verses(self) -> List[Dict]:
        """Get all BOM verses."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM bom_verses ORDER BY book, chapter, verse").fetchall()
            return [dict(row) for row in rows]

    def get_bom_verses_by_book(self, book: str) -> List[Dict]:
        """Get all verses from a specific book."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM bom_verses WHERE book = ? ORDER BY chapter, verse",
                (book,)
            ).fetchall()
            return [dict(row) for row in rows]

    # ==================== Source Passages ====================

    def insert_source_passage(self, id: str, source: str, source_name: str,
                              location: str, text: str, page: int = None) -> None:
        """Insert a source text passage."""
        word_count = len(text.split())
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO source_passages
                   (id, source, source_name, location, page, text, word_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (id, source, source_name, location, page, text, word_count)
            )

    def insert_source_passages_batch(self, passages: List[Dict]) -> int:
        """Batch insert source passages. Returns count inserted."""
        with self.get_connection() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO source_passages
                   (id, source, source_name, location, page, text, word_count)
                   VALUES (:id, :source, :source_name, :location, :page, :text, :word_count)""",
                passages
            )
            return len(passages)

    def get_source_passages(self, source: str = None) -> List[Dict]:
        """Get source passages, optionally filtered by source."""
        with self.get_connection() as conn:
            if source:
                rows = conn.execute(
                    "SELECT * FROM source_passages WHERE source = ?", (source,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM source_passages").fetchall()
            return [dict(row) for row in rows]

    # ==================== Matches ====================

    def insert_match(self, bom_verse_id: str, source_passage_id: str,
                     ngram_score: float = None, tfidf_score: float = None,
                     embedding_score: float = None, llm_score: float = None,
                     final_score: float = None, match_type: str = "unknown",
                     confidence: str = "low", llm_explanation: str = None,
                     llm_model: str = None, detection_methods: str = None,
                     matched_phrases: str = None) -> int:
        """Insert or update a match. Returns the match ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO matches
                   (bom_verse_id, source_passage_id, ngram_score, tfidf_score,
                    embedding_score, llm_score, final_score, match_type, confidence,
                    llm_explanation, llm_model, detection_methods, matched_phrases)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bom_verse_id, source_passage_id) DO UPDATE SET
                    ngram_score = COALESCE(excluded.ngram_score, ngram_score),
                    tfidf_score = COALESCE(excluded.tfidf_score, tfidf_score),
                    embedding_score = COALESCE(excluded.embedding_score, embedding_score),
                    llm_score = COALESCE(excluded.llm_score, llm_score),
                    final_score = COALESCE(excluded.final_score, final_score),
                    match_type = COALESCE(excluded.match_type, match_type),
                    confidence = COALESCE(excluded.confidence, confidence),
                    llm_explanation = COALESCE(excluded.llm_explanation, llm_explanation),
                    llm_model = COALESCE(excluded.llm_model, llm_model),
                    detection_methods = COALESCE(excluded.detection_methods, detection_methods),
                    matched_phrases = COALESCE(excluded.matched_phrases, matched_phrases),
                    updated_at = CURRENT_TIMESTAMP""",
                (bom_verse_id, source_passage_id, ngram_score, tfidf_score,
                 embedding_score, llm_score, final_score, match_type, confidence,
                 llm_explanation, llm_model, detection_methods, matched_phrases)
            )
            return cursor.lastrowid

    def get_matches_for_verse(self, bom_verse_id: str) -> List[Dict]:
        """Get all matches for a specific BOM verse."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT m.*, sp.text as source_text, sp.source_name, sp.location
                   FROM matches m
                   JOIN source_passages sp ON m.source_passage_id = sp.id
                   WHERE m.bom_verse_id = ?
                   ORDER BY m.final_score DESC""",
                (bom_verse_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_matches_by_source(self, source: str) -> List[Dict]:
        """Get all matches from a specific source text."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                          sp.text as source_text, sp.location
                   FROM matches m
                   JOIN bom_verses bv ON m.bom_verse_id = bv.id
                   JOIN source_passages sp ON m.source_passage_id = sp.id
                   WHERE sp.source = ?
                   ORDER BY m.final_score DESC""",
                (source,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_matches_by_type(self, match_type: str) -> List[Dict]:
        """Get all matches of a specific type."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT m.*, bv.text as verse_text, sp.text as source_text
                   FROM matches m
                   JOIN bom_verses bv ON m.bom_verse_id = bv.id
                   JOIN source_passages sp ON m.source_passage_id = sp.id
                   WHERE m.match_type = ?
                   ORDER BY m.final_score DESC""",
                (match_type,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_top_matches(self, limit: int = 100, min_score: float = 0.0) -> List[Dict]:
        """Get top matches by score."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                          sp.text as source_text, sp.source_name, sp.location
                   FROM matches m
                   JOIN bom_verses bv ON m.bom_verse_id = bv.id
                   JOIN source_passages sp ON m.source_passage_id = sp.id
                   WHERE m.final_score >= ?
                   ORDER BY m.final_score DESC
                   LIMIT ?""",
                (min_score, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def update_review_status(self, match_id: int, status: str,
                             reviewed_by: str = None, notes: str = None) -> None:
        """Update the review status of a match."""
        with self.get_connection() as conn:
            conn.execute(
                """UPDATE matches SET
                   review_status = ?, reviewed_by = ?, reviewed_at = ?, review_notes = ?
                   WHERE id = ?""",
                (status, reviewed_by, datetime.now(), notes, match_id)
            )

    # ==================== Analytics ====================

    def get_match_stats(self) -> Dict:
        """Get summary statistics about matches."""
        with self.get_connection() as conn:
            stats = {}

            # Total counts
            stats['total_verses'] = conn.execute("SELECT COUNT(*) FROM bom_verses").fetchone()[0]
            stats['total_passages'] = conn.execute("SELECT COUNT(*) FROM source_passages").fetchone()[0]
            stats['total_matches'] = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

            # Matches by type
            rows = conn.execute(
                "SELECT match_type, COUNT(*) as count FROM matches GROUP BY match_type"
            ).fetchall()
            stats['by_type'] = {row['match_type']: row['count'] for row in rows}

            # Matches by source
            rows = conn.execute(
                """SELECT sp.source, COUNT(*) as count
                   FROM matches m
                   JOIN source_passages sp ON m.source_passage_id = sp.id
                   GROUP BY sp.source"""
            ).fetchall()
            stats['by_source'] = {row['source']: row['count'] for row in rows}

            # Matches by review status
            rows = conn.execute(
                "SELECT review_status, COUNT(*) as count FROM matches GROUP BY review_status"
            ).fetchall()
            stats['by_status'] = {row['review_status']: row['count'] for row in rows}

            # Average scores
            row = conn.execute(
                """SELECT AVG(final_score) as avg_score,
                          AVG(embedding_score) as avg_embedding,
                          AVG(llm_score) as avg_llm
                   FROM matches WHERE final_score IS NOT NULL"""
            ).fetchone()
            stats['avg_scores'] = dict(row) if row else {}

            return stats

    def get_verses_with_most_matches(self, limit: int = 20) -> List[Dict]:
        """Get verses that have the most matches."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT bv.*, COUNT(m.id) as match_count,
                          AVG(m.final_score) as avg_score
                   FROM bom_verses bv
                   JOIN matches m ON bv.id = m.bom_verse_id
                   GROUP BY bv.id
                   ORDER BY match_count DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    # ==================== Export ====================

    def export_to_parquet(self, output_path: Path = None) -> Path:
        """Export matches to Parquet for analytics."""
        import pandas as pd

        if output_path is None:
            output_path = DATA_DIR / "matches_export.parquet"

        with self.get_connection() as conn:
            df = pd.read_sql_query(
                """SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                          sp.text as source_text, sp.source, sp.source_name, sp.location
                   FROM matches m
                   JOIN bom_verses bv ON m.bom_verse_id = bv.id
                   JOIN source_passages sp ON m.source_passage_id = sp.id""",
                conn
            )
            df.to_parquet(output_path, index=False)
            return output_path


# Singleton instance
_db = None

def get_db() -> Database:
    """Get the database singleton instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


if __name__ == "__main__":
    # Initialize the database
    db = get_db()
    print(f"Database initialized at: {DATABASE_PATH}")

    # Show stats
    stats = db.get_match_stats()
    print(f"Total verses: {stats['total_verses']}")
    print(f"Total passages: {stats['total_passages']}")
    print(f"Total matches: {stats['total_matches']}")

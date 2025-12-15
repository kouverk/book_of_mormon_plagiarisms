# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Detect textual parallels between the Book of Mormon and 19th-century source texts (View of the Hebrews, The Late War, First Book of Napoleon, KJV Bible). Uses multi-method detection: lexical (n-gram, TF-IDF), semantic (embeddings), and LLM analysis.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Initialize database
python database.py

# Load texts into database
python text_parser.py bom texts/raw/bom-jspapers-1830.txt
python text_parser.py source view_of_hebrews texts/raw/view_of_hebrews.txt "View of the Hebrews"

# Generate embeddings
python embed_sources.py

# Run detection pipeline
python search_and_match.py

# Test utilities
python utils.py
```

## Architecture

### Data Flow
```
Raw Texts → text_parser.py → SQLite DB (verses, passages)
                                   ↓
                            embed_sources.py → FAISS indices
                                   ↓
BoM Verse → search_and_match.py → Candidates → match_engine.py → Scored Matches
              (lexical + embedding)              (LLM analysis)
                                   ↓
                            database.py → matches table (with review status)
```

### Key Files

| File | Purpose |
|------|---------|
| `config.py` | Central config, loads YAML files from `config/` |
| `database.py` | SQLite schema and data access layer |
| `text_parser.py` | Parse raw texts into structured format |
| `utils.py` | Similarity functions (n-gram, TF-IDF, embeddings, LLM) |
| `embed_sources.py` | Generate FAISS indices from source texts |
| `search_and_match.py` | Multi-method retrieval pipeline |
| `match_engine.py` | LLM-powered semantic analysis |

### Database Schema (SQLite)

- `bom_verses` - Book of Mormon verses with book/chapter/verse structure
- `source_passages` - Passages from source texts with location metadata
- `matches` - Many-to-many matches with scores, types, and review status
- `entities` - Named entities extracted from texts
- `known_parallels` - Calibration data from CES Letter

### Configuration (YAML in `config/`)

- `sources.yaml` - Source text metadata and Archive.org URLs
- `models.yaml` - Embedding and LLM model settings
- `thresholds.yaml` - Score thresholds and weights

## Data Locations

- `texts/raw/` - Original PDF and TXT files
- `texts/structured/` - Parsed JSON files
- `data/matches.db` - SQLite database
- `embeddings/` - FAISS indices
- `calibration/` - Known parallels for validation

## Match Types

- `direct_copy` - Near-verbatim phrase borrowing
- `paraphrase` - Same idea, different words
- `thematic` - Shared theological/narrative concept
- `stylistic` - KJV-style language patterns
- `name_borrow` - Borrowed proper nouns

## Review Statuses

Matches can be manually curated: `pending`, `confirmed`, `rejected`, `needs_review`

## Environment

Requires `OPENAI_API_KEY` for embeddings and LLM analysis.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Detect textual parallels between the Book of Mormon and 19th-century source texts (View of the Hebrews, The Late War, First Book of Napoleon, KJV Bible). Uses multi-method detection: lexical (n-gram, TF-IDF), semantic (embeddings), and LLM analysis.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run commands via run.py (from project root)
python run.py database          # Initialize database
python run.py embed_sources     # Generate source embeddings
python run.py embed_bom         # Generate BOM embeddings
python run.py search [min_score]    # Find matches, save to CSV
python run.py classify [min_score]  # LLM classify matches
python run.py report            # Generate HTML report
```

## Project Structure

```
book_of_mormon_plagiarisms/
├── run.py                    # Main entry point for all commands
├── src/                      # Core Python modules
│   ├── config.py             # Central config, loads YAML files
│   ├── database.py           # SQLite schema and data access
│   ├── text_parser.py        # Parse raw texts into DB
│   ├── utils.py              # Similarity functions (n-gram, TF-IDF, embeddings)
│   ├── embed_sources.py      # Generate source FAISS indices
│   ├── embed_bom.py          # Generate BOM FAISS indices
│   ├── search_and_match.py   # Multi-method retrieval pipeline
│   ├── match_engine.py       # LLM-powered semantic analysis
│   ├── classify_matches.py   # Batch LLM classification
│   └── report_generator.py   # HTML report generation
├── config/                   # YAML configuration
│   ├── models.yaml           # Embedding and LLM model settings
│   ├── sources.yaml          # Source text metadata
│   └── thresholds.yaml       # Score thresholds and weights
├── scripts/                  # One-off conversion scripts
│   ├── bom_convert.py
│   └── dnc_convert.py
├── data/                     # Database and embeddings
│   ├── matches.db            # SQLite database
│   └── embeddings/           # FAISS indices and metadata
├── texts/                    # Source texts
│   └── raw/                  # Original PDF and TXT files
├── results/                  # Output reports
│   ├── report.html           # Interactive HTML report
│   └── matches.json          # Exported match data
└── logs/                     # Session logs (gitignored)
```

## Data Flow

```
Raw Texts → text_parser.py → SQLite DB (verses, passages)
                                  ↓
                           embed_*.py → FAISS indices
                                  ↓
BoM Verse → search_and_match.py → Candidates → classify_matches.py → Scored Matches
             (embedding search)                  (LLM analysis)
                                  ↓
                           database.py → matches table
                                  ↓
                           report_generator.py → HTML report
```

## Database Schema (SQLite)

- `bom_verses` - Book of Mormon verses with book/chapter/verse structure
- `source_passages` - Passages from source texts with location metadata
- `matches` - Many-to-many matches with scores, types, and review status

## Match Types

- `direct_copy` - Near-verbatim phrase borrowing
- `paraphrase` - Same idea, different words
- `thematic` - Shared theological/narrative concept
- `stylistic` - KJV-style language patterns
- `none` - No meaningful parallel

## Environment

Requires `OPENAI_API_KEY` for embeddings and LLM analysis.

## Current State

**Last updated: 2025-12-17**

- LLM classification: **100% complete** (19,479/19,479 matches with embedding_score >= 0.5)
- Classification breakdown: thematic (18,005), direct_copy (809), paraphrase (497), none (160), stylistic (8)
- HTML report available at `results/report.html`

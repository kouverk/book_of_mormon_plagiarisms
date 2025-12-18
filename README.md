# Book of Mormon Textual Parallel Detection

Detect potential literary borrowings, thematic connections, and textual parallels between the Book of Mormon and 19th-century source texts using NLP and embedding-based similarity methods.

## Current Status

**Classification complete!** All 19,479 high-confidence matches have been analyzed.

| Match Type | Count | Description |
|------------|-------|-------------|
| Thematic | 18,005 | Shared theological/narrative concepts |
| Direct Copy | 809 | Near-verbatim phrase borrowing |
| Paraphrase | 497 | Same idea, different words |
| None | 160 | No meaningful parallel |
| Stylistic | 8 | KJV-style language patterns |

**View the results:** Open `results/report.html` in a browser for an interactive report.

## Project Goals

1. **Primary**: Detect direct textual borrowings (phrase overlap, name borrowing)
2. **Secondary**: Identify thematic/conceptual parallels
3. **Scale**: Exhaustive analysis of all 6,604 Book of Mormon verses against 26,149 source passages

## Source Texts

### Target Text
- **Book of Mormon** (1830 edition) - 6,604 verses

### Source Texts Analyzed
| Text | Date | Passages | Key Parallels |
|------|------|----------|---------------|
| KJV Bible | 1769 | 31,102 | Biblical borrowing, theological concepts |
| View of the Hebrews | 1825 | ~2,000 | Hebrew migration to Americas, buried records |
| The Late War | 1816 | ~1,500 | KJV-style language, "stripling soldiers" |
| First Book of Napoleon | 1809 | ~1,000 | "It came to pass" style patterns |

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"

# Generate report (data already processed)
python run.py report
open results/report.html
```

## Project Structure

```
book_of_mormon_plagiarisms/
├── run.py                    # Main entry point
├── src/                      # Core Python modules
│   ├── config.py             # Central configuration
│   ├── database.py           # SQLite data access
│   ├── text_parser.py        # Parse raw texts
│   ├── utils.py              # Similarity functions
│   ├── embed_sources.py      # Generate FAISS indices
│   ├── embed_bom.py          # Embed BOM verses
│   ├── search_and_match.py   # Vector similarity search
│   ├── classify_matches.py   # LLM classification
│   └── report_generator.py   # HTML report
├── config/                   # YAML configuration
├── scripts/                  # One-off converters
├── data/
│   ├── matches.db            # SQLite database (63,252 matches)
│   └── embeddings/           # FAISS indices
├── texts/raw/                # Source text files
└── results/
    └── report.html           # Interactive results
```

## Commands

```bash
python run.py database          # Initialize database
python run.py embed_sources     # Generate source embeddings
python run.py embed_bom         # Generate BOM embeddings
python run.py search [score]    # Find matches (default: 0.4)
python run.py classify [score]  # LLM classify (default: 0.5)
python run.py report            # Generate HTML report
```

## Detection Pipeline

```
Raw Texts → Parser → SQLite DB → Embeddings → FAISS Index
                                      ↓
              BOM Verse → Vector Search → Top-K Candidates
                                      ↓
                              LLM Classification
                                      ↓
                              HTML Report
```

### Methods Used
| Method | What It Catches |
|--------|-----------------|
| OpenAI embeddings | Deep semantic similarity |
| FAISS vector search | Fast nearest-neighbor lookup |
| GPT-4o-mini | Match type classification |

## Match Categories

- **direct_copy**: Near-verbatim phrase borrowing
- **paraphrase**: Same idea, different words
- **thematic**: Shared theological/narrative concept
- **stylistic**: KJV-style language patterns
- **none**: No meaningful parallel detected

## How This Compares to WordTree (2013)

The [WordTree Foundation](http://wordtree.org/thelatewar/) compared the BOM against 100,000 pre-1830 books using 4-gram analysis. The Late War ranked in the **top 0.001%** for rare phrase connections.

| Aspect | WordTree | This Project |
|--------|----------|--------------|
| Method | 4-gram lexical only | Embeddings + LLM |
| Matching | Binary phrase exists | Similarity spectrum |
| Paraphrases | Cannot detect | Embeddings catch these |
| Output | Corpus statistics | Verse-level explanations |

WordTree was the metal detector saying "dig here." This project is the excavation - embeddings catch semantic borrowing when Joseph took an *idea* but rewrote it, and the LLM explains *why* passages connect.

## Technologies

- Python 3.12
- SQLite - data storage
- FAISS - vector similarity search
- OpenAI text-embedding-3-large
- GPT-4o-mini - classification
- Bootstrap 5 - report UI

## References

- [CES Letter - Book of Mormon](https://read.cesletter.org/bom/)
- [WordTree Foundation](http://wordtree.org/thelatewar/)
- [WordTree 4-gram Study (GitHub)](https://github.com/wordtreefoundation/4-gram-study)

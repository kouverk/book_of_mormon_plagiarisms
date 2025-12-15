# Book of Mormon Textual Parallel Detection

Detect potential literary borrowings, thematic connections, and textual parallels between the Book of Mormon and 19th-century source texts using NLP and embedding-based similarity methods.

## Project Goals

1. **Primary**: Detect direct textual borrowings (phrase overlap, name borrowing)
2. **Secondary**: Identify thematic/conceptual parallels
3. **Scale**: Exhaustive analysis of all ~6,000+ Book of Mormon verses

## Source Texts

### Target Text (What We're Analyzing)
- **Book of Mormon** (1830) - Joseph Smith

### Primary Source Texts (Pre-1830)
| Text | Date | Author | Key Parallels |
|------|------|--------|---------------|
| View of the Hebrews | 1825 (2nd ed) | Ethan Smith | Hebrew migration to Americas, civilized vs barbarous peoples, buried records |
| The Late War | 1816 | Gilbert Hunt | KJV-style language, "stripling soldiers", fortifications, earthquakes |
| First Book of Napoleon | 1809 | Unknown | "It came to pass" style, KJV language patterns |
| KJV Bible | 1769 | Various | Biblical borrowing, theological concepts |

### Secondary Source Texts (Future Phases)
- Wonders of Nature & Providence (1825) - Josiah Priest
- Pilgrim's Progress (1678) - John Bunyan
- The Apocrypha (2 Maccabees)
- Spalding Manuscript (~1812)

## Detection Methods

| Method | What It Catches | Cost |
|--------|-----------------|------|
| N-gram overlap | Direct phrase copying | Free |
| TF-IDF + cosine | Unusual word patterns | Free |
| Sentence-BERT | Semantic similarity | Free |
| OpenAI embeddings | Deep semantic | ~$1 |
| LLM judgment | Nuanced parallels | $5-50 |

## Project Structure

```
/anti_mormon_project
├── config.py                 # Central configuration
├── text_parser.py            # Structured text parsing
├── entity_extractor.py       # Named entity detection
├── report_generator.py       # HTML/JSON output
│
├── embed_sources.py          # Multi-level embeddings
├── search_and_match.py       # Multi-method pipeline
├── match_engine.py           # Tiered LLM analysis
├── utils.py                  # Lexical similarity methods
│
├── /file-convert/            # PDF to text conversion
│   ├── bom-convert.py
│   └── dnc-convert.py
│
├── /texts/
│   ├── /raw/                 # Original PDFs/TXTs
│   └── /structured/          # Parsed JSON files
│
├── /embeddings/
│   ├── verse_level/
│   ├── paragraph_level/
│   └── chapter_level/
│
├── /calibration/
│   └── known_parallels.json  # Ground truth from CES Letter
│
└── /results/
    ├── matches.json
    ├── report.html
    └── matches.csv
```

## Setup

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
# Add to ~/.zshrc or ~/.bashrc
export OPENAI_API_KEY="sk-your-key-here"
source ~/.zshrc
```

### 3. Download Source Texts

Download from Archive.org and place in `/texts/raw/`:
- [View of the Hebrews (1825)](https://archive.org/details/viewofhebrewsort00smit)
- [The Late War (1816)](https://archive.org/details/latewarbetween_00hunt)
- [First Book of Napoleon (1809)](https://archive.org/details/firstbooknapole00gruagoog)

### 4. Run the Pipeline

```bash
# Parse and structure texts
python text_parser.py

# Generate embeddings
python embed_sources.py

# Run detection pipeline
python search_and_match.py

# Generate report
python report_generator.py
```

## Match Categories

The system classifies matches into:
- **DIRECT_COPY**: Near-verbatim phrase borrowing
- **PARAPHRASE**: Same idea, different words
- **THEMATIC**: Shared theological/narrative concept
- **NAME_BORROW**: Borrowed proper nouns
- **STRUCTURAL**: Similar plot/story beats
- **STYLISTIC**: KJV-style language patterns

## Technologies

- Python 3.13
- FAISS - vector similarity search
- OpenAI Embeddings (text-embedding-3-large)
- Sentence-BERT (all-MiniLM-L6-v2)
- SpaCy - named entity recognition
- GPT-4o / Claude - semantic analysis

## Cost Estimates

| Component | Estimated Cost |
|-----------|----------------|
| OpenAI embeddings | < $1 |
| Tier 1 LLM filtering | $2-5 |
| Tier 2 LLM deep analysis | $10-20 |
| **Total** | **$15-30** |

## References

- [CES Letter - Book of Mormon](https://read.cesletter.org/bom/)
- [View of the Hebrews parallels](https://read.cesletter.org/bom/#view-of-the-hebrews)
- [The Late War parallels](https://read.cesletter.org/bom/#the-late-war)
- [First Book of Napoleon parallels](https://read.cesletter.org/bom/#the-first-book-of-napoleon)

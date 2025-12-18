"""
Text Parser for Book of Mormon and Source Texts

Parses raw text files into structured format and loads into database.
Handles:
- Book of Mormon verse parsing (book, chapter, verse structure)
- Source text chunking (by paragraph, page, or chapter)
- Text cleaning and normalization
"""

import re
from pathlib import Path
from typing import List, Dict, Generator, Optional
from tqdm import tqdm

from config import (
    TEXTS_RAW_DIR, TEXTS_STRUCTURED_DIR, get_sources_config
)
from database import get_db


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page markers like "--- Page 1 ---"
    text = re.sub(r'---\s*Page\s*\d+\s*---', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def normalize_book_name(name: str) -> str:
    """Normalize book names to consistent format."""
    # Map variations to canonical names
    book_map = {
        'first nephi': '1 Nephi',
        '1 nephi': '1 Nephi',
        'i nephi': '1 Nephi',
        'second nephi': '2 Nephi',
        '2 nephi': '2 Nephi',
        'ii nephi': '2 Nephi',
        'jacob': 'Jacob',
        'enos': 'Enos',
        'jarom': 'Jarom',
        'omni': 'Omni',
        'words of mormon': 'Words of Mormon',
        'mosiah': 'Mosiah',
        'alma': 'Alma',
        'helaman': 'Helaman',
        'third nephi': '3 Nephi',
        '3 nephi': '3 Nephi',
        'iii nephi': '3 Nephi',
        'fourth nephi': '4 Nephi',
        '4 nephi': '4 Nephi',
        'iv nephi': '4 Nephi',
        'mormon': 'Mormon',
        'ether': 'Ether',
        'moroni': 'Moroni',
    }
    return book_map.get(name.lower().strip(), name)


def make_verse_id(book: str, chapter: int, verse: int) -> str:
    """Create a standardized verse ID."""
    # Convert "1 Nephi" to "1nephi"
    book_key = book.lower().replace(' ', '')
    return f"{book_key}.{chapter}.{verse}"


class BOMParser:
    """Parser for Book of Mormon text from Joseph Smith Papers format."""

    # Regex patterns for verse detection
    VERSE_PATTERN = re.compile(
        r'(?:^|\n)\s*(\d+)\s+(.+?)(?=\n\s*\d+\s+|\n\s*(?:CHAPTER|Chapter)\s+\d+|$)',
        re.DOTALL
    )

    CHAPTER_PATTERN = re.compile(
        r'(?:CHAPTER|Chapter)\s+(\d+)',
        re.IGNORECASE
    )

    BOOK_PATTERN = re.compile(
        r'(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|1ST|2ND|3RD|4TH)?\s*(?:BOOK\s+OF\s+)?'
        r'(NEPHI|JACOB|ENOS|JAROM|OMNI|MOSIAH|ALMA|HELAMAN|MORMON|ETHER|MORONI|'
        r'WORDS\s+OF\s+MORMON)',
        re.IGNORECASE
    )

    def __init__(self, file_path: Path):
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()

    def parse(self) -> Generator[Dict, None, None]:
        """
        Parse the BOM text and yield verse dictionaries.

        Note: The 1830 BOM format is tricky - it wasn't originally versified
        the same way as modern editions. This parser attempts to handle
        the Joseph Smith Papers format.
        """
        current_book = None
        current_chapter = 0

        # Split into rough sections
        lines = self.raw_text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for book header
            book_match = self.BOOK_PATTERN.search(line)
            if book_match:
                current_book = normalize_book_name(book_match.group(1))
                current_chapter = 0
                continue

            # Check for chapter header
            chapter_match = self.CHAPTER_PATTERN.search(line)
            if chapter_match:
                current_chapter = int(chapter_match.group(1))
                continue

            # If we have a book and chapter, look for verse numbers
            if current_book and current_chapter > 0:
                # Check if line starts with a verse number
                verse_match = re.match(r'^(\d+)\s+(.+)', line)
                if verse_match:
                    verse_num = int(verse_match.group(1))
                    verse_text = clean_text(verse_match.group(2))

                    if verse_text and len(verse_text) > 5:  # Skip tiny fragments
                        yield {
                            'id': make_verse_id(current_book, current_chapter, verse_num),
                            'book': current_book,
                            'chapter': current_chapter,
                            'verse': verse_num,
                            'text': verse_text,
                            'word_count': len(verse_text.split())
                        }

    def parse_simple(self) -> Generator[Dict, None, None]:
        """
        Simpler parsing approach: split by paragraphs and assign IDs.
        Use this if the structured parsing doesn't work well.
        """
        # Remove page markers
        text = re.sub(r'---\s*Page\s*\d+\s*---', '\n', self.raw_text)

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        for i, para in enumerate(paragraphs):
            para = clean_text(para)
            if para and len(para) > 20:  # Skip short fragments
                yield {
                    'id': f'bom.para.{i}',
                    'book': 'Unknown',
                    'chapter': 0,
                    'verse': i,
                    'text': para,
                    'word_count': len(para.split())
                }


class SourceTextParser:
    """Parser for source texts (View of Hebrews, Late War, etc.)."""

    def __init__(self, file_path: Path, source_key: str, source_name: str):
        self.file_path = file_path
        self.source_key = source_key
        self.source_name = source_name

        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()

    def parse_by_paragraph(self, min_length: int = 50) -> Generator[Dict, None, None]:
        """Parse text into paragraphs."""
        # Remove page markers
        text = re.sub(r'---\s*Page\s*\d+\s*---', '\n', self.raw_text)

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        para_num = 0
        for para in paragraphs:
            para = clean_text(para)
            if para and len(para) >= min_length:
                para_num += 1
                yield {
                    'id': f'{self.source_key}.p{para_num}',
                    'source': self.source_key,
                    'source_name': self.source_name,
                    'location': f'Paragraph {para_num}',
                    'page': None,
                    'text': para,
                    'word_count': len(para.split())
                }

    def parse_by_page(self) -> Generator[Dict, None, None]:
        """Parse text by page markers."""
        # Find page markers
        pages = re.split(r'---\s*Page\s*(\d+)\s*---', self.raw_text)

        # pages[0] is before first marker, then alternates: page_num, content, page_num, content...
        for i in range(1, len(pages), 2):
            if i + 1 < len(pages):
                page_num = int(pages[i])
                page_text = clean_text(pages[i + 1])

                if page_text and len(page_text) > 50:
                    yield {
                        'id': f'{self.source_key}.page{page_num}',
                        'source': self.source_key,
                        'source_name': self.source_name,
                        'location': f'Page {page_num}',
                        'page': page_num,
                        'text': page_text,
                        'word_count': len(page_text.split())
                    }

    def parse_by_sentence(self, min_words: int = 10) -> Generator[Dict, None, None]:
        """Parse text into sentences (finest granularity)."""
        # Clean the text
        text = re.sub(r'---\s*Page\s*\d+\s*---', ' ', self.raw_text)
        text = clean_text(text)

        # Split into sentences (rough approximation)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        sent_num = 0
        for sent in sentences:
            sent = sent.strip()
            word_count = len(sent.split())
            if word_count >= min_words:
                sent_num += 1
                yield {
                    'id': f'{self.source_key}.s{sent_num}',
                    'source': self.source_key,
                    'source_name': self.source_name,
                    'location': f'Sentence {sent_num}',
                    'page': None,
                    'text': sent,
                    'word_count': word_count
                }


def parse_verse_reference(reference: str) -> tuple:
    """
    Parse a verse reference like '1 Nephi 1:1' into (book, chapter, verse).
    Returns (book_name, chapter_int, verse_int).
    """
    import re
    # Match patterns like "1 Nephi 1:1", "Alma 5:23", "Words of Mormon 1:5"
    match = re.match(r'^(.+?)\s+(\d+):(\d+)$', reference)
    if match:
        book = match.group(1)
        chapter = int(match.group(2))
        verse = int(match.group(3))
        return book, chapter, verse
    return reference, 0, 0


def load_bom_from_json(file_path: Path = None) -> int:
    """Load Book of Mormon verses from JSON file (modern versified edition)."""
    import json

    if file_path is None:
        file_path = TEXTS_RAW_DIR / 'book-of-mormon-modern.json'

    if not file_path.exists():
        raise FileNotFoundError(f"BOM JSON file not found: {file_path}")

    print(f"Loading Book of Mormon from JSON: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    verses = []
    for v in data['verses']:
        book, chapter, verse_num = parse_verse_reference(v['reference'])
        text = clean_text(v['text'])

        # Create verse ID like "1nephi.1.1"
        book_key = book.lower().replace(' ', '')
        verse_id = f"{book_key}.{chapter}.{verse_num}"

        verses.append({
            'id': verse_id,
            'book': book,
            'chapter': chapter,
            'verse': verse_num,
            'text': text,
            'word_count': len(text.split())
        })

    print(f"Found {len(verses)} verses")

    db = get_db()
    count = db.insert_bom_verses_batch(verses)
    print(f"Inserted {count} verses into database")

    return count


def load_bom_to_database(file_path: Path = None) -> int:
    """Load Book of Mormon verses into the database."""
    # Check for JSON file first (preferred)
    json_path = TEXTS_RAW_DIR / 'book-of-mormon-modern.json'
    if json_path.exists() and file_path is None:
        return load_bom_from_json(json_path)

    if file_path is None:
        # Find the BOM file
        bom_files = list(TEXTS_RAW_DIR.glob('bom*.txt')) + list(Path('texts').glob('bom*.txt'))
        if not bom_files:
            raise FileNotFoundError("No BOM text file found in texts/ or texts/raw/")
        file_path = bom_files[0]

    print(f"Parsing Book of Mormon from: {file_path}")
    parser = BOMParser(file_path)

    db = get_db()
    verses = list(parser.parse())

    if len(verses) < 100:
        # If structured parsing didn't find much, fall back to simple
        print("Structured parsing found few verses, using paragraph-based parsing...")
        verses = list(parser.parse_simple())

    print(f"Found {len(verses)} verses/passages")

    # Insert in batches
    count = db.insert_bom_verses_batch(verses)
    print(f"Inserted {count} verses into database")

    return count


def load_source_to_database(source_key: str, file_path: Path,
                            source_name: str = None,
                            parse_method: str = 'paragraph') -> int:
    """Load a source text into the database."""
    if source_name is None:
        source_name = source_key.replace('_', ' ').title()

    print(f"Parsing {source_name} from: {file_path}")
    parser = SourceTextParser(file_path, source_key, source_name)

    if parse_method == 'paragraph':
        passages = list(parser.parse_by_paragraph())
    elif parse_method == 'page':
        passages = list(parser.parse_by_page())
    elif parse_method == 'sentence':
        passages = list(parser.parse_by_sentence())
    else:
        raise ValueError(f"Unknown parse method: {parse_method}")

    print(f"Found {len(passages)} passages")

    db = get_db()
    count = db.insert_source_passages_batch(passages)
    print(f"Inserted {count} passages into database")

    return count


def load_all_sources() -> Dict[str, int]:
    """Load all configured source texts into the database."""
    sources_config = get_sources_config()
    results = {}

    # Load primary sources
    for key, config in sources_config.get('primary_sources', {}).items():
        file_name = config.get('file')
        if file_name:
            file_path = TEXTS_RAW_DIR / file_name
            if not file_path.exists():
                file_path = Path('texts') / file_name

            if file_path.exists():
                count = load_source_to_database(
                    key, file_path, config.get('name')
                )
                results[key] = count
            else:
                print(f"Warning: File not found for {key}: {file_name}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'bom':
            # Load Book of Mormon
            file_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
            load_bom_to_database(file_path)

        elif command == 'source':
            # Load a source text
            if len(sys.argv) < 4:
                print("Usage: python text_parser.py source <key> <file_path> [name]")
                sys.exit(1)
            key = sys.argv[2]
            file_path = Path(sys.argv[3])
            name = sys.argv[4] if len(sys.argv) > 4 else None
            load_source_to_database(key, file_path, name)

        elif command == 'all':
            # Load all configured sources
            load_all_sources()

        else:
            print(f"Unknown command: {command}")
            print("Commands: bom, source, all")
    else:
        print("Book of Mormon Text Parser")
        print("Usage:")
        print("  python text_parser.py bom [file_path]     - Load Book of Mormon")
        print("  python text_parser.py source <key> <path> - Load a source text")
        print("  python text_parser.py all                 - Load all configured sources")

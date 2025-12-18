#!/usr/bin/env python3
"""
Main entry point for running project commands from the project root.

Usage:
    python run.py database          # Initialize database
    python run.py parse_bom         # Parse Book of Mormon
    python run.py parse_source <id> <file> <name>  # Parse source text
    python run.py embed_sources     # Generate source embeddings
    python run.py embed_bom         # Generate BOM embeddings
    python run.py search            # Run search and match
    python run.py classify [source] [threshold]  # Classify matches
    python run.py report            # Generate HTML report
"""

import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "database":
        from database import init_db
        init_db()
        print("Database initialized.")

    elif command == "parse_bom":
        from text_parser import parse_bom_file
        from config import TEXTS_RAW_DIR
        bom_file = TEXTS_RAW_DIR / "bom-jspapers-1830-2025-06-07 0737pm.txt"
        parse_bom_file(bom_file)

    elif command == "parse_source":
        if len(sys.argv) < 5:
            print("Usage: python run.py parse_source <source_id> <filename> <display_name>")
            sys.exit(1)
        from text_parser import parse_source_file
        from config import TEXTS_RAW_DIR
        source_id = sys.argv[2]
        filename = sys.argv[3]
        display_name = sys.argv[4]
        parse_source_file(source_id, TEXTS_RAW_DIR / filename, display_name)

    elif command == "embed_sources":
        from embed_sources import main as embed_sources_main
        embed_sources_main()

    elif command == "embed_bom":
        from embed_bom import main as embed_bom_main
        embed_bom_main()

    elif command == "search":
        from search_and_match import find_all_matches
        min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
        df = find_all_matches(min_score=min_score)
        from config import RESULTS_DIR
        df.to_csv(RESULTS_DIR / "matches.csv", index=False)
        print(f"Saved {len(df)} matches to results/matches.csv")

    elif command == "classify":
        from classify_matches import classify_all_matches
        min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        classify_all_matches(min_score=min_score)

    elif command == "report":
        from report_generator import main as report_main
        report_main()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
HTML Report Generator for Book of Mormon Textual Parallels

Generates an interactive HTML report showing:
- Summary statistics
- Top matches by score
- Matches grouped by source text
- Matches grouped by BOM book
- Filter by match type, confidence, tags
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from database import get_db, OFFICIAL_TAGS
from config import RESULTS_DIR

RESULTS_DIR.mkdir(exist_ok=True)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def truncate(text: str, length: int = 200) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= length:
        return text
    return text[:length] + "..."


def get_match_data() -> Dict:
    """Gather all data needed for the report."""
    db = get_db()
    data = {}

    # Get stats
    data['stats'] = db.get_match_stats()
    data['tag_counts'] = db.get_tag_counts()

    # Get top matches
    data['top_matches'] = db.get_top_matches(limit=500, min_score=0.5)

    # Get matches by source
    data['by_source'] = {}
    for source in ['kjv_bible', 'view_of_hebrews', 'the_late_war', 'first_book_of_napoleon']:
        matches = db.get_matches_by_source(source)[:100]  # Top 100 per source
        data['by_source'][source] = matches

    # Get classified matches
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                   sp.text as source_text, sp.source_name, sp.location, sp.source
            FROM matches m
            JOIN bom_verses bv ON m.bom_verse_id = bv.id
            JOIN source_passages sp ON m.source_passage_id = sp.id
            WHERE m.llm_explanation IS NOT NULL AND m.llm_explanation != ''
            ORDER BY m.embedding_score DESC
            LIMIT 200
        """).fetchall()
        data['classified_matches'] = [dict(row) for row in rows]

    # Non-biblical high-confidence matches (most interesting!)
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                   sp.text as source_text, sp.source_name, sp.location, sp.source
            FROM matches m
            JOIN bom_verses bv ON m.bom_verse_id = bv.id
            JOIN source_passages sp ON m.source_passage_id = sp.id
            WHERE sp.source != 'kjv_bible'
              AND m.embedding_score >= 0.6
            ORDER BY m.embedding_score DESC
            LIMIT 200
        """).fetchall()
        data['non_biblical_matches'] = [dict(row) for row in rows]

    return data


def generate_match_card(match: Dict, show_explanation: bool = True) -> str:
    """Generate HTML for a single match card."""
    match_type = match.get('match_type', 'unknown')
    confidence = match.get('confidence', 'low')
    score = match.get('embedding_score') or match.get('final_score') or 0

    type_colors = {
        'direct_copy': '#dc3545',
        'paraphrase': '#fd7e14',
        'thematic': '#ffc107',
        'stylistic': '#17a2b8',
        'none': '#6c757d',
        'unknown': '#6c757d'
    }

    conf_badges = {
        'high': '<span class="badge bg-success">High Confidence</span>',
        'medium': '<span class="badge bg-warning text-dark">Medium</span>',
        'low': '<span class="badge bg-secondary">Low</span>'
    }

    source_key = match.get('source', '')
    source_display = {
        'kjv_bible': 'KJV Bible',
        'view_of_hebrews': 'View of the Hebrews',
        'the_late_war': 'The Late War',
        'first_book_of_napoleon': 'First Book of Napoleon'
    }.get(source_key, source_key)

    bom_ref = f"{match.get('book', '')} {match.get('chapter', '')}:{match.get('verse', '')}"

    explanation_html = ""
    if show_explanation and match.get('llm_explanation'):
        explanation_html = f"""
        <div class="explanation">
            <strong>Analysis:</strong> {escape_html(match['llm_explanation'])}
        </div>
        """

    return f"""
    <div class="match-card" data-type="{match_type}" data-source="{source_key}" data-score="{score:.3f}">
        <div class="match-header">
            <span class="match-type" style="background-color: {type_colors.get(match_type, '#6c757d')}">
                {match_type.replace('_', ' ').title()}
            </span>
            <span class="score">Score: {score:.3f}</span>
            {conf_badges.get(confidence, '')}
        </div>
        <div class="match-content">
            <div class="text-block bom">
                <div class="text-label">Book of Mormon - {bom_ref}</div>
                <div class="text-content">"{escape_html(truncate(match.get('verse_text', ''), 300))}"</div>
            </div>
            <div class="text-block source">
                <div class="text-label">{source_display} - {escape_html(match.get('location', ''))}</div>
                <div class="text-content">"{escape_html(truncate(match.get('source_text', ''), 300))}"</div>
            </div>
        </div>
        {explanation_html}
    </div>
    """


def generate_html_report(data: Dict) -> str:
    """Generate the full HTML report."""
    stats = data['stats']

    # Generate match cards for each section
    top_matches_html = "\n".join(
        generate_match_card(m) for m in data['top_matches'][:100]
    )

    non_biblical_html = "\n".join(
        generate_match_card(m) for m in data['non_biblical_matches'][:100]
    )

    classified_html = "\n".join(
        generate_match_card(m, show_explanation=True) for m in data['classified_matches'][:100]
    )

    # Source-specific sections
    source_sections = ""
    for source_key, source_name in [
        ('view_of_hebrews', 'View of the Hebrews'),
        ('the_late_war', 'The Late War'),
        ('first_book_of_napoleon', 'First Book of Napoleon'),
        ('kjv_bible', 'KJV Bible')
    ]:
        matches = data['by_source'].get(source_key, [])[:50]
        if matches:
            cards = "\n".join(generate_match_card(m) for m in matches)
            source_sections += f"""
            <section id="source-{source_key}" class="source-section">
                <h2>{source_name}</h2>
                <p class="section-info">Top {len(matches)} matches from {source_name}</p>
                <div class="matches-container">{cards}</div>
            </section>
            """

    # Stats summary
    by_source = stats.get('by_source', {})
    by_type = stats.get('by_type', {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Book of Mormon Textual Parallels Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {{
            --bom-color: #2c5282;
            --source-color: #744210;
        }}
        body {{
            font-family: 'Georgia', serif;
            background-color: #f8f9fa;
            line-height: 1.6;
        }}
        .navbar {{
            background-color: #1a365d;
        }}
        .container {{
            max-width: 1200px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #2c5282;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9rem;
        }}
        .match-card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #6c757d;
        }}
        .match-card[data-type="direct_copy"] {{ border-left-color: #dc3545; }}
        .match-card[data-type="paraphrase"] {{ border-left-color: #fd7e14; }}
        .match-card[data-type="thematic"] {{ border-left-color: #ffc107; }}
        .match-card[data-type="stylistic"] {{ border-left-color: #17a2b8; }}
        .match-header {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}
        .match-type {{
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .score {{
            font-weight: bold;
            color: #333;
        }}
        .text-block {{
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 0.75rem;
        }}
        .text-block.bom {{
            background-color: #ebf4ff;
            border-left: 3px solid var(--bom-color);
        }}
        .text-block.source {{
            background-color: #fffbeb;
            border-left: 3px solid var(--source-color);
        }}
        .text-label {{
            font-weight: bold;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            color: #333;
        }}
        .text-content {{
            font-style: italic;
            color: #444;
        }}
        .explanation {{
            background: #f8f9fa;
            padding: 0.75rem;
            border-radius: 4px;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        .nav-pills .nav-link {{
            color: #495057;
        }}
        .nav-pills .nav-link.active {{
            background-color: #2c5282;
        }}
        .section-info {{
            color: #666;
            margin-bottom: 1rem;
        }}
        .filter-bar {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #1a365d;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }}
        .legend {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.85rem;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark mb-4">
        <div class="container">
            <span class="navbar-brand">Book of Mormon Textual Parallels</span>
            <span class="navbar-text text-white">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    </nav>

    <div class="container">
        <!-- Summary Stats -->
        <section id="summary">
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{stats.get('total_verses', 0):,}</div>
                    <div class="stat-label">BOM Verses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('total_passages', 0):,}</div>
                    <div class="stat-label">Source Passages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('total_matches', 0):,}</div>
                    <div class="stat-label">Total Matches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{by_source.get('view_of_hebrews', 0):,}</div>
                    <div class="stat-label">View of the Hebrews</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{by_source.get('the_late_war', 0):,}</div>
                    <div class="stat-label">The Late War</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{by_source.get('first_book_of_napoleon', 0):,}</div>
                    <div class="stat-label">First Book of Napoleon</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{by_source.get('kjv_bible', 0):,}</div>
                    <div class="stat-label">KJV Bible</div>
                </div>
            </div>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background:#dc3545"></div>
                    <span>Direct Copy</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#fd7e14"></div>
                    <span>Paraphrase</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#ffc107"></div>
                    <span>Thematic</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#17a2b8"></div>
                    <span>Stylistic</span>
                </div>
            </div>
        </section>

        <!-- Navigation -->
        <ul class="nav nav-pills mb-4" id="sectionNav">
            <li class="nav-item">
                <a class="nav-link active" href="#non-biblical">Non-Biblical Matches</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#top-matches">Top Matches</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#classified">LLM Classified</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#by-source">By Source</a>
            </li>
        </ul>

        <!-- Non-Biblical Matches (Most Interesting!) -->
        <section id="non-biblical">
            <h2>Non-Biblical Matches (Most Interesting)</h2>
            <p class="section-info">
                High-scoring matches excluding KJV Bible. These represent potential borrowings from
                View of the Hebrews, The Late War, and First Book of Napoleon.
            </p>
            <div class="matches-container">
                {non_biblical_html if non_biblical_html else '<p>No non-biblical matches found above threshold.</p>'}
            </div>
        </section>

        <!-- Top Matches -->
        <section id="top-matches" style="display:none">
            <h2>Top Matches by Score</h2>
            <p class="section-info">Highest scoring matches across all sources (includes biblical parallels)</p>
            <div class="matches-container">
                {top_matches_html}
            </div>
        </section>

        <!-- LLM Classified -->
        <section id="classified" style="display:none">
            <h2>LLM Classified Matches</h2>
            <p class="section-info">Matches that have been analyzed by GPT-4o-mini with explanations</p>
            <div class="matches-container">
                {classified_html if classified_html else '<p>No classified matches yet. Run classify_matches.py</p>'}
            </div>
        </section>

        <!-- By Source -->
        <section id="by-source" style="display:none">
            <h2>Matches by Source Text</h2>
            {source_sections}
        </section>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Simple tab navigation
        document.querySelectorAll('#sectionNav .nav-link').forEach(link => {{
            link.addEventListener('click', function(e) {{
                e.preventDefault();

                // Update active nav
                document.querySelectorAll('#sectionNav .nav-link').forEach(l => l.classList.remove('active'));
                this.classList.add('active');

                // Show/hide sections
                const target = this.getAttribute('href').substring(1);
                document.querySelectorAll('section[id]').forEach(section => {{
                    if (section.id === 'summary') return; // Always show summary
                    section.style.display = section.id === target ? 'block' : 'none';
                }});
            }});
        }});
    </script>
</body>
</html>
"""
    return html


def generate_report():
    """Generate and save the HTML report."""
    print("Gathering data...")
    data = get_match_data()

    print("Generating HTML...")
    html = generate_html_report(data)

    output_path = RESULTS_DIR / "report.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved to: {output_path}")
    return output_path


def export_json():
    """Export match data as JSON for custom analysis."""
    db = get_db()

    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT m.*, bv.text as verse_text, bv.book, bv.chapter, bv.verse,
                   sp.text as source_text, sp.source_name, sp.location, sp.source
            FROM matches m
            JOIN bom_verses bv ON m.bom_verse_id = bv.id
            JOIN source_passages sp ON m.source_passage_id = sp.id
            WHERE m.embedding_score >= 0.5
            ORDER BY m.embedding_score DESC
        """).fetchall()

        matches = [dict(row) for row in rows]

    output_path = RESULTS_DIR / "matches.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, default=str)

    print(f"Exported {len(matches)} matches to: {output_path}")
    return output_path


def main():
    """Main entry point."""
    generate_report()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "html":
            generate_report()
        elif cmd == "json":
            export_json()
        else:
            print(f"Unknown command: {cmd}")
    else:
        generate_report()

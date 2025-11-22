"""
Check section name formats across all papers
"""

import json
from collections import defaultdict

chunks_file = 'vectorstore/layer2/chunks.jsonl'

paper_sections = defaultdict(set)
section_format_by_paper = {}

with open(chunks_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        chunk = json.loads(line)
        paper_id = chunk.get('paper_id')
        section = chunk.get('metadata', {}).get('section_name')
        if section:
            paper_sections[paper_id].add(section)

print("=" * 80)
print("Section Name Formats by Paper")
print("=" * 80)

# Categorize papers by section format
semantic_sections = []  # Has Introduction, Methodology, etc.
numbered_sections = []   # Has Section 1, Section 2, etc.
other_formats = []

for paper_id, sections in sorted(paper_sections.items()):
    sections_list = sorted(sections)
    
    # Check format
    has_semantic = any(s in ['Introduction', 'Methodology', 'Results', 'Dataset', 'Conclusion', 'Related_Work'] for s in sections)
    has_numbered = any(s.startswith('Section ') for s in sections)
    
    if has_semantic:
        semantic_sections.append(paper_id)
        format_type = "✓ Semantic"
    elif has_numbered:
        numbered_sections.append(paper_id)
        format_type = "⚠ Numbered"
    else:
        other_formats.append(paper_id)
        format_type = "❓ Other"
    
    print(f"\n{format_type}: {paper_id}")
    print(f"  Sections: {sections_list}")

print("\n" + "=" * 80)
print("Summary:")
print(f"  Semantic sections: {len(semantic_sections)} papers")
print(f"  Numbered sections: {len(numbered_sections)} papers")
print(f"  Other formats: {len(other_formats)} papers")
print("=" * 80)

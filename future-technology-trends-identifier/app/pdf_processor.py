# -*- coding: utf-8 -*-
"""
PDF analysis pipeline for Future Tech Trends.

Steps:
1) Extract + clean text from PDF
2) Detect sections and chunk (or fallback to naive chunking)
3) Analyze chunks concurrently via LLM
4) Merge per-chunk extractions into a single result

Return value is the merged dict that `merge_results()` produces
(typically: {"technologies": [...]}).

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

import re
from typing import Tuple, Dict, List
import PyPDF2
from .config import settings

# -----------------------------------------------------------------------------
# Section heading patterns
# -----------------------------------------------------------------------------
GENERIC_PATTERNS = [
    r'^\s*(\d+\.\d+(?:\.\d+)*)\s+([A-Z][^\n]{10,80})(?=\n|$)',
    r'^\s*[A-Z]{3,}-\d+[-A-Z]*\s+([^\n]{10,80})(?=\n|$)',
    r'^\s*(?:WP|Work Package)\s*\d+:\s*([^\n]+)',
    r'^\s*[IVX]+\.\s+[A-Z][^\n]{10,80}',
    r'^\s*[A-Z][A-Z\s]{10,}[A-Z]\s*(?=\n|$)'
]

EU_SPECIAL_PATTERNS = [
    r'^\s*Introduction\s*(?:\.{3,}\s*\d+)?\s*(?=\n|$)',
    r'^\s*Horizon Europe\s*(?:-\s*)?Work Programme\s*\d{4}-\d{4}',
    r'^\s*Part\s+\d+\s+-\s+Page\s+\d+\s+of\s+\d+\s*$',
    r'^\s*Destination\s*\d+\s*:\s*([^\n]{10,120}?)(?=\.{3,}\s*\d+|\n|$)',
    r'^\s*Destination\s*[IVX]+\s*:\s*([^\n]{10,120}?)(?=\.{3,}\s*\d+|\n|$)',
    r'^\s*Cluster\s+\d+\s*:\s*([^\n]{5,120}?)(?=\.{3,}\s*\d+|\n|$)',
    r'^\s*Call\s*-\s*([A-Z][^\n]{5,120}?)\s+\d{4}(?:\s+TWO\s+STAGE)?(?=\n|$)',
    r'^\s*Conditions\s+for\s+the\s+Call\s*(?=\n|$)',
    r'^\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,6})\s*(?=\.{3,}\s*\d+\s*$)',
    r'^\s*HORIZON-[A-Z0-9-]+:\s*([^\n]+)',
    r'^\s*[A-Z]{2}-\d+-\d{4}\b',
    r'^\s*Type\s+of\s+Action\s*:\s*(?:RIA|IA|CSA|FPA|ERA|COFUND|PDA|PPP)\b'
]

SECTION_PATTERNS = EU_SPECIAL_PATTERNS + GENERIC_PATTERNS

# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
def extract_text(pdf_path: str) -> Tuple[str, Dict[int, str]]:
    """
    Extract text with newlines preserved. Each page ends with '\n'.
    Respects MAX_PAGES using 1-based page_num (no off-by-one).
    """
    text = ""
    page_map = {}
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            page_num = i + 1
            if _should_skip_page(page_num, total_pages):
                continue
            page_text = page.extract_text() or ""
            page_text += "\n"
            if page_text:
                text += page_text
                page_map[page_num] = page_text
            if settings.max_pages > 0 and i >= settings.max_pages:
                break
    return text, page_map

def _should_skip_page(page_num: int, total_pages: int) -> bool:
    raw = [p.strip() for p in settings.skip_pages.split(',') if p.strip()]
    adjusted = []
    for p in raw:
        try:
            n = int(p)
        except ValueError:
            continue
        adjusted.append(n if n >= 0 else total_pages + n + 1)
    return page_num in adjusted

# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalize while preserving newlines so line-by-line detection works.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    return text.strip()

# -----------------------------------------------------------------------------
# Section detection
# -----------------------------------------------------------------------------
def detect_sections(text: str) -> List[tuple]:
    """
    Scan lines sequentially; when a pattern matches, capture the title and
    compute the absolute start index in the full text.
    """
    sections = []
    for line in text.split('\n'):
        for pattern in SECTION_PATTERNS:
            m = re.search(pattern, line)
            if m:
                title = m.group(1) if m.lastindex else line.strip()
                sections.append((m.start(), -1, title))
                break
    # end positions
    fixed = []
    for i in range(len(sections)):
        start, _, title = sections[i]
        end = sections[i+1][0] if i+1 < len(sections) else len(text)
        fixed.append((start, end, title))
    return [
        (s, e, t) for s, e, t in fixed
        if len(t) > 5 and not any(w in t.lower() for w in ['footer','header','page'])
    ]

# -----------------------------------------------------------------------------
# Chunking 
# -----------------------------------------------------------------------------
def chunk_text(text: str, sections: List[tuple]) -> List[dict]:
    """
    Create chunks from section spans
    """
    chunks = []
    for start, end, title in sections:
        if any(ex in title.lower() for ex in settings.exclude_sections.split(',')):
            continue
        section_text = text[start:end]
        chunks.extend(_subchunk_section(section_text, title))
    # add simple priority score
    kws = [k.strip().lower() for k in settings.priority_keywords.split(',') if k.strip()]
    for ch in chunks:
        ch['priority'] = sum(kw in ch['text'].lower() for kw in kws)
    return sorted(chunks, key=lambda x: -x['priority'])

def _subchunk_section(text: str, section_title: str) -> List[dict]:
    """
    Chunk a single section by window with overlap, preferring '\n\n' cut points.
    """
    chunks = []
    start = 0
    end = settings.chunk_size
    while start < len(text):
        if end < len(text):
            cut = text.rfind('\n\n', start, end)
            if cut != -1:
                end = cut + 2
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "section": section_title,
                "context": f"Document Section: {section_title}",
                "byte_size": len(chunk_text.encode('utf-8'))
            })
        start = max(end - settings.overlap, end)
        end = start + settings.chunk_size
    return chunks

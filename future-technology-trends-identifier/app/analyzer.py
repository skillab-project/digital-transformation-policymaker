
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from tqdm import tqdm
from pathlib import Path

from .config import settings
from .pdf_processor import extract_text, clean_text, detect_sections, chunk_text
from .llm_client import analyze_chunk
from .cluster_merge import merge_results

DEFAULT_QUERY = """You are an expert technology analyst. Your task is to read the following document chunk and identify explicitly 
mentioned, job-relevant, general-purpose emerging technologies.

The technologies you extract must be:
 - Explicitly named or clearly described in the document
 - Broad in scope
 - Relevant to future workforce needs and skills demand (i.e., tied to roles, processes, or applications in industry)

For each valid technology explicitly mentioned, extract and provide the following information:
1. Name: Clearly name the technology (be specific, e.g., “Post-quantum cryptography” not just “AI”)
2. Description: A concise description (50–100 words) explaining what the technology is and how it is used or proposed in the document.
3. Domain: Identify the domain or sector where the technology applies (e.g., ICT, Health, Energy, Manufacturing).
4. Future Occupations Needed: List the specific roles or job titles that will be needed to support or develop this technology in the future.
5. Confidence Score (1–5): Based on the context of the document, rate how explicitly the technology was mentioned (5 = clearly described and emphasized, 1 = vague or only implied).

Return the output as a valid JSON object with the following structure, always including all five fields:
{
  "technologies": [
    {
      "name": "Technology Name",
      "description": "Description here",
      "domain": "Domain here",
      "occupations": ["Occupation 1", "Occupation 2"],
      "confidence": 5
    }
  ]
}

❌ Do not invent technologies based on your own knowledge.
❌ Do not include anything that is inferred or only implied."""


def process_pdf(pdf_path: str, query: Optional[str] = None, max_chunks: int = 100) -> Dict:
    raw_text, page_map = extract_text(pdf_path)
    clean = clean_text(raw_text)
    sections = detect_sections(clean)

    if sections:
        chunks = chunk_text(clean, sections)
    else:
        # fallback to naive chunking
        chunks = []
        start = 0
        while start < len(clean):
            end = start + settings.chunk_size
            chunk = clean[start:end]
            chunks.append({
                "text": chunk,
                "context": "Full document",
                "section": "No section",
                "byte_size": len(chunk.encode("utf-8"))
            })
            start = end - settings.overlap

    # parallel analysis
    results: List[Dict] = []
    q = query or DEFAULT_QUERY
    use = chunks[:max_chunks] if max_chunks else chunks

    with ThreadPoolExecutor(max_workers=settings.parallel_chunks) as ex:
        futures = {ex.submit(analyze_chunk, ch["text"], q, ch["context"], settings.timeout): ch for ch in use}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Analyzing chunks"):
            ch = futures[fut]
            try:
                out = fut.result()
                if out:
                    out["section"] = ch["section"]
                    results.append(out)
            except Exception as e:
                # swallow and continue
                pass

    merged = merge_results(results)
    return merged

def save_json(data, out_path: str):
    Path(out_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

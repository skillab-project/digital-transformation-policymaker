
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

def merge_results(results: List[Dict]) -> Dict:
    if not results:
        return {"technologies": []}
    all_techs = []
    for r in results:
        all_techs.extend(r.get("technologies", []))
    if not all_techs:
        return {"technologies": []}

    tech_texts = [f"{t['name']}. {t.get('description','')}" for t in all_techs]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(tech_texts, convert_to_tensor=False)

    dbscan = DBSCAN(eps=0.3, min_samples=1, metric="cosine")
    labels = dbscan.fit_predict(embeddings)

    merged = []
    for cid in set(labels):
        cluster_techs = [t for t, lbl in zip(all_techs, labels) if lbl == cid]
        rep = max(cluster_techs, key=lambda x: len(x.get("description", "")))
        combined_occ = sorted(set(o for t in cluster_techs for o in t.get("occupations", [])))
        related = sorted(set(t["name"] for t in cluster_techs if t["name"] != rep["name"]))
        merged.append({
            **rep,
            "occupations": combined_occ,
            "cluster_size": len(cluster_techs),
            "related_names": related
        })
    merged = [m for m in merged if m["cluster_size"] >= 2]
    return {"technologies": merged}

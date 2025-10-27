# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:20:29 2025

@author: tsoukj
"""

# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import app.main as main_mod


@pytest.fixture()
def client(monkeypatch):
    """
    Function-scoped TestClient with heavy startup work stubbed out.
    """
    # Stub startup side effects
    monkeypatch.setattr(main_mod, "warm_esco_caches", lambda: None)
    monkeypatch.setattr(main_mod, "_load_jobs", lambda: None)
    monkeypatch.setattr(main_mod, "rehydrate_from_storage", lambda *_args, **_kwargs: None)

    return TestClient(main_mod.app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_policy_recommendations_inline_content(client, monkeypatch):
    # Stub generator & save_json
    fake_out = {
        "emerging": [
            {"name": "Quantum Networking", "description": "...", "domain": "ICT", "occupations": [], "confidence": 0.9}
        ],
        "recommendations": [
            {
                "technology": "Quantum Networking",
                "actions": [
                    {
                        "area": "Training/Reskilling",
                        "action": "Create pilot MSc modules",
                        "rationale": "Skill gap",
                        "timeframe": "short",
                        "priority": "High",
                    }
                ],
            }
        ],
        "mapping_evidence": {"occupations": [], "skills": []},
    }
    monkeypatch.setattr(main_mod, "generate_policy_recommendations", lambda **_k: fake_out)
    monkeypatch.setattr(main_mod, "save_json", lambda data, path: None)

    body = {
        "technologies": [
            {"name": "Quantum Networking", "description": "...", "domain": "ICT", "occupations": [], "confidence": 0.9}
        ],
        "target": "both",
        "similarity_threshold": 0.5,
        "max_actions_per_tech": 4,
    }

    r = client.post("/policy/recommendations?include_content=true", json=body)
    assert r.status_code == 200, r.text
    data = r.json()

    assert {"job_id", "result_path", "emerging_count", "has_recommendations", "content"} <= set(data.keys())
    content = data["content"]
    assert "emerging" in content and "recommendations" in content and "mapping_evidence" in content
    assert content["recommendations"][0]["technology"] == "Quantum Networking"


def test_map_to_esco_occupations_inline(client, monkeypatch):
    def fake_map_occ(techs, top_n, threshold):
        # Accept both dicts and pydantic models
        out = []
        for t in techs:
            name = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
            out.append({
                "technology": name,
                "matches": [{"label": "Example Occupation", "score": 0.77}],
            })
        return out

    monkeypatch.setattr(main_mod, "map_technologies_to_esco_occupations", fake_map_occ)

    body = {
        "target": "occupations",
        "top_n": 5,
        "threshold": 0.5,
        "technologies": [
            {"name": "AI in Education", "description": "...", "domain": "ICT", "occupations": [], "confidence": 0.8}
        ],
    }

    r = client.post("/map-to-esco", json=body)
    assert r.status_code == 200, r.text
    data = r.json()

    assert isinstance(data, list)
    assert data[0]["technology"] == "AI in Education"
    assert data[0]["matches"][0]["label"] == "Example Occupation"

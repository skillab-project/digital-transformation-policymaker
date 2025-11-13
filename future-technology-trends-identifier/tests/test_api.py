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


def test_policy_recommendations_inline_content(client):
    """
    /policy/recommendations no longer supports inline content.
    It should ignore include_content and return only the base response fields.
    """
    body = {
        "technologies": [{
            "name": "AI in Education",
            "description": "...",
            "domain": "ICT",
            "occupations": [],
            "confidence": 0.8
        }],
        "target": "both"
    }

    r = client.post("/policy/recommendations?include_content=true", json=body)
    assert r.status_code == 200
    data = r.json()

    # Expected keys (NO 'content')
    assert set(data.keys()) == {
        "job_id",
        "result_path",
        "emerging_count",
        "has_recommendations"
    }


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

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:20:29 2025

@author: tsoukj
"""

# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import app.main as main_mod
import app.jobs as jobs_mod


@pytest.fixture()
def client(monkeypatch):
    """
    Function-scoped TestClient with heavy startup work stubbed out.
    """
    # Stub startup side effects
    monkeypatch.setattr(main_mod, "warm_esco_caches", lambda: None)
    monkeypatch.setattr(main_mod, "_load_jobs", lambda: None)
    monkeypatch.setattr(main_mod, "rehydrate_from_storage", lambda *_args, **_kwargs: None)
    jobs_mod._jobs.clear()

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


def test_analyze_pdf_persists_user_id(client, monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "process_pdf",
        lambda *_args, **_kwargs: {"technologies": [{"name": "AI in Education"}]},
    )

    files = {"file": ("sample.pdf", b"%PDF-1.4 mock pdf", "application/pdf")}
    data = {"user_id": "user-123"}

    r = client.post("/analyze/pdf", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["status"] == "queued"
    assert body["user_id"] == "user-123"

    status = client.get(f"/jobs/{body['job_id']}")
    assert status.status_code == 200, status.text
    status_body = status.json()
    assert status_body["status"] == "done"
    assert status_body["user_id"] == "user-123"


def test_policy_job_inherits_user_id_and_source_job(client, monkeypatch):
    source_job_id = jobs_mod.new_job()
    jobs_mod.set_status(
        source_job_id,
        "done",
        user_id="user-456",
        result_path="storage/source.analysis.json",
    )

    monkeypatch.setattr(
        main_mod,
        "load_json",
        lambda *_args, **_kwargs: {
            "technologies": [{"name": "Quantum Networking", "description": "", "domain": "ICT"}]
        },
    )
    monkeypatch.setattr(
        main_mod,
        "generate_policy_recommendations",
        lambda **_kwargs: {"emerging": [], "recommendations": [], "mapping_evidence": {}},
    )

    r = client.post("/policy/recommendations", json={"job_id": source_job_id, "target": "both"})
    assert r.status_code == 200, r.text
    body = r.json()

    status = client.get(f"/jobs/{body['job_id']}")
    assert status.status_code == 200, status.text
    status_body = status.json()
    assert status_body["user_id"] == "user-456"
    assert status_body["source_job_id"] == source_job_id


def test_duplicate_pdf_reuse_is_scoped_to_same_user(client, monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "process_pdf",
        lambda *_args, **_kwargs: {"technologies": [{"name": "Edge AI"}]},
    )

    files = {"file": ("same.pdf", b"%PDF-1.4 same content", "application/pdf")}

    first = client.post("/analyze/pdf", files=files, data={"user_id": "user-a"})
    assert first.status_code == 200, first.text
    first_body = first.json()

    second_same_user = client.post("/analyze/pdf", files=files, data={"user_id": "user-a"})
    assert second_same_user.status_code == 200, second_same_user.text
    same_user_body = second_same_user.json()
    assert same_user_body["job_id"] == first_body["job_id"]
    assert same_user_body["message"] == "Duplicate PDF detected. Reusing previous result."

    third_other_user = client.post("/analyze/pdf", files=files, data={"user_id": "user-b"})
    assert third_other_user.status_code == 200, third_other_user.text
    other_user_body = third_other_user.json()
    assert other_user_body["job_id"] != first_body["job_id"]
    assert other_user_body["user_id"] == "user-b"


def test_list_user_analyses_returns_only_matching_user_results(client, monkeypatch):
    jobs_mod.set_status(
        "analysis-user-1",
        "done",
        user_id="user-1",
        result_path="storage/analysis-user-1.analysis.json",
    )
    jobs_mod.set_status(
        "analysis-user-2",
        "done",
        user_id="user-2",
        result_path="storage/analysis-user-2.analysis.json",
    )
    jobs_mod.set_status(
        "policy-user-1",
        "done",
        user_id="user-1",
        type="policy",
        result_path="storage/policy-user-1.policy.json",
        source_job_id="analysis-user-1",
    )

    monkeypatch.setattr(
        main_mod.Path,
        "exists",
        lambda self: str(self).endswith(".analysis.json") or str(self).endswith(".policy.json"),
    )

    r = client.get("/users/user-1/analyses")
    assert r.status_code == 200, r.text
    data = r.json()

    assert len(data) == 1
    assert data[0]["job_id"] == "analysis-user-1"
    assert data[0]["user_id"] == "user-1"
    assert data[0]["type"] == "analysis"
    assert data[0]["content"] is None


def test_list_user_policies_can_include_content(client, monkeypatch):
    jobs_mod.set_status(
        "policy-1",
        "done",
        user_id="user-9",
        type="policy",
        result_path="storage/policy-1.policy.json",
        source_job_id="analysis-9",
    )

    monkeypatch.setattr(main_mod.Path, "exists", lambda self: True)
    monkeypatch.setattr(
        main_mod,
        "load_json",
        lambda path: {"emerging": [{"name": "Quantum Networking"}], "recommendations": []},
    )

    r = client.get("/users/user-9/policies?include_content=true")
    assert r.status_code == 200, r.text
    data = r.json()

    assert len(data) == 1
    assert data[0]["job_id"] == "policy-1"
    assert data[0]["type"] == "policy"
    assert data[0]["source_job_id"] == "analysis-9"
    assert data[0]["content"]["emerging"][0]["name"] == "Quantum Networking"

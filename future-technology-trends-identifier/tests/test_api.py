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


def _seed_catalog(monkeypatch):
    """Seed three completed analyses across two titles/sectors."""
    jobs_mod.set_status(
        "a1", "done", user_id="user-1", title="AI Report", sector="ICT",
        description="desc A", filename="a1.pdf",
        created_at="2026-01-01T00:00:00+00:00",
        result_path="storage/a1.analysis.json",
    )
    jobs_mod.set_status(
        "a2", "done", user_id="user-1", title="AI Report", sector="ICT",
        description="desc A", filename="a2.pdf",
        created_at="2026-01-02T00:00:00+00:00",
        result_path="storage/a2.analysis.json",
    )
    jobs_mod.set_status(
        "a3", "done", user_id="user-2", title="Energy Grid", sector="Energy",
        description="desc E", filename="a3.pdf",
        created_at="2026-01-03T00:00:00+00:00",
        result_path="storage/a3.analysis.json",
    )
    monkeypatch.setattr(main_mod.Path, "exists", lambda self: True)


def test_analyses_titles_lists_distinct_titles_newest_first(client, monkeypatch):
    _seed_catalog(monkeypatch)

    r = client.get("/analyses/titles")
    assert r.status_code == 200, r.text
    data = r.json()

    by_title = {t["title"]: t for t in data}
    assert set(by_title) == {"AI Report", "Energy Grid"}
    assert by_title["AI Report"]["count"] == 2
    assert by_title["AI Report"]["sector"] == "ICT"
    assert by_title["AI Report"]["description"] == "desc A"
    assert by_title["AI Report"]["created_at"] == "2026-01-02T00:00:00+00:00"
    # Newest-first ordering (Energy Grid is the most recent analysis)
    assert data[0]["title"] == "Energy Grid"


def test_analyses_by_title_returns_matching_records(client, monkeypatch):
    _seed_catalog(monkeypatch)

    r = client.get("/analyses/by-title/AI Report")
    assert r.status_code == 200, r.text
    data = r.json()

    assert len(data) == 2
    assert {d["job_id"] for d in data} == {"a1", "a2"}
    assert all(d["title"] == "AI Report" for d in data)
    assert {d["filename"] for d in data} == {"a1.pdf", "a2.pdf"}
    assert data[0]["content"] is None


def test_analyses_by_title_can_include_content(client, monkeypatch):
    _seed_catalog(monkeypatch)
    monkeypatch.setattr(
        main_mod,
        "load_json",
        lambda path: {"technologies": [{"name": "Edge AI"}], "title": "AI Report"},
    )

    r = client.get("/analyses/by-title/AI Report?include_content=true")
    assert r.status_code == 200, r.text
    data = r.json()

    assert data[0]["content"]["technologies"][0]["name"] == "Edge AI"


def test_analysis_title_exists(client, monkeypatch):
    _seed_catalog(monkeypatch)  # seeds titles "AI Report" and "Energy Grid"

    r = client.get("/analyses/title-exists", params={"title": "AI Report"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exists"] is True
    assert body["message"]

    # Case-insensitive + whitespace-insensitive.
    assert client.get("/analyses/title-exists", params={"title": "  ai report "}).json()["exists"] is True

    # A free title.
    free = client.get("/analyses/title-exists", params={"title": "Brand New"}).json()
    assert free["exists"] is False
    assert free["message"] is None


def test_analyze_pdf_full_runs_pipeline(client, monkeypatch):
    import json as _json

    monkeypatch.setattr(main_mod, "process_pdf",
                        lambda *a, **k: {"technologies": [{"name": "Edge AI"}]})
    monkeypatch.setattr(
        main_mod, "generate_policy_recommendations",
        lambda **k: {"emerging": [],
                     "recommendations": [{"technology": "Edge AI", "actions": []}],
                     "mapping_evidence": {"occupations": [], "skills": []}},
    )

    files = [
        ("files", ("a.pdf", b"%PDF-1.4 a", "application/pdf")),
        ("files", ("b.pdf", b"%PDF-1.4 b", "application/pdf")),
    ]
    r = client.post("/analyze/pdf/full", files=files,
                    data={"user_id": "u1", "title": "Batch", "sector": "ICT", "description": "d"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2 and len(body["job_ids"]) == 2

    # Background task ran to completion under the TestClient.
    for jid in body["job_ids"]:
        info = jobs_mod.get_job(jid)
        assert info["status"] == "done" and info["stage"] == "done"
        assert info["type"] == "analysis"
        saved = _json.load(open(info["result_path"], encoding="utf-8"))
        assert saved["title"] == "Batch" and saved["technologies"][0]["name"] == "Edge AI"

    # Aggregate progress via titles?include_running.
    titles = {t["title"]: t for t in client.get("/analyses/titles?include_running=true").json()}
    assert titles["Batch"]["total"] == 2
    assert titles["Batch"]["done_count"] == 2
    assert titles["Batch"]["status"] == "done"

    # Each PDF has a stored policy (recommendations + mapping_evidence).
    pol = client.get("/policies/by-title/Batch?include_content=true").json()
    assert len(pol) == 2
    assert all(p["content"]["recommendations"][0]["technology"] == "Edge AI" for p in pol)
    assert all("mapping_evidence" in p["content"] for p in pol)

    # Cleanup files written to ./storage by the pipeline.
    import shutil, os
    shutil.rmtree("storage", ignore_errors=True)
    os.makedirs("storage", exist_ok=True)


def test_titles_include_running_reports_progress(client):
    jobs_mod.set_status("a1", "done", type="analysis", title="Batch", sector="ICT",
                        result_path="storage/a1.analysis.json")
    jobs_mod.set_status("a2", "running", type="analysis", stage="analyzing",
                        title="Batch", sector="ICT")

    # Default listing: only the completed PDF is counted.
    default = {t["title"]: t for t in client.get("/analyses/titles").json()}
    assert default["Batch"]["count"] == 1

    running = {t["title"]: t for t in client.get("/analyses/titles?include_running=true").json()}
    assert running["Batch"]["total"] == 2
    assert running["Batch"]["done_count"] == 1
    assert running["Batch"]["status"] == "running"

    # by-title?include_running exposes both jobs with their live status.
    recs = client.get("/analyses/by-title/Batch?include_running=true").json()
    assert {r["job_id"] for r in recs} == {"a1", "a2"}
    statuses = {r["job_id"]: r["status"] for r in recs}
    assert statuses["a2"] == "running"
    assert next(r for r in recs if r["job_id"] == "a2")["stage"] == "analyzing"


def test_analyses_sectors_lists_distinct_sectors(client, monkeypatch):
    _seed_catalog(monkeypatch)

    r = client.get("/analyses/sectors")
    assert r.status_code == 200, r.text
    assert r.json() == ["Energy", "ICT"]


def test_analyses_by_sector_lists_titles(client, monkeypatch):
    _seed_catalog(monkeypatch)

    r = client.get("/analyses/by-sector/ICT")
    assert r.status_code == 200, r.text
    data = r.json()

    assert [t["title"] for t in data] == ["AI Report"]
    assert data[0]["sector"] == "ICT"
    assert data[0]["count"] == 2


def test_old_users_analyses_endpoint_is_removed(client):
    r = client.get("/users/user-1/analyses")
    assert r.status_code == 404


def test_delete_analyses_by_title_removes_analyses_and_policies(client):
    jobs_mod.set_status("a1", "done", user_id="user-1", title="AI Report", sector="ICT",
                        result_path="storage/a1.analysis.json")
    jobs_mod.set_status("a2", "done", user_id="user-1", title="AI Report", sector="ICT",
                        result_path="storage/a2.analysis.json")
    jobs_mod.set_status("a3", "done", user_id="user-2", title="Energy Grid", sector="Energy",
                        result_path="storage/a3.analysis.json")
    jobs_mod.set_status("p1", "done", user_id="user-1", type="policy",
                        result_path="storage/p1.policy.json", source_job_id="a1")

    r = client.delete("/analyses/by-title/AI Report")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["deleted_analyses"] == 2
    assert data["deleted_policies"] == 1

    # Removed from the registry (and its policy), other title untouched.
    assert jobs_mod.get_job("a1") is None
    assert jobs_mod.get_job("a2") is None
    assert jobs_mod.get_job("p1") is None
    assert jobs_mod.get_job("a3") is not None

    titles = {t["title"] for t in client.get("/analyses/titles").json()}
    assert titles == {"Energy Grid"}


def test_delete_analyses_by_title_404_when_missing(client):
    r = client.delete("/analyses/by-title/Nonexistent")
    assert r.status_code == 404


def test_policies_by_title_returns_only_that_titles_policies(client, monkeypatch):
    # Two analyses under "AI Report" (different users), one under "Energy Grid".
    jobs_mod.set_status("a1", "done", user_id="user-1", title="AI Report", sector="ICT",
                        result_path="storage/a1.analysis.json")
    jobs_mod.set_status("a2", "done", user_id="user-2", title="AI Report", sector="ICT",
                        result_path="storage/a2.analysis.json")
    jobs_mod.set_status("a3", "done", user_id="user-1", title="Energy Grid", sector="Energy",
                        result_path="storage/a3.analysis.json")
    # Policies: p1 from a1 (user-1), p2 from a2 (user-2), p3 from a3 (other title).
    jobs_mod.set_status("p1", "done", user_id="user-1", type="policy",
                        result_path="storage/p1.policy.json", source_job_id="a1")
    jobs_mod.set_status("p2", "done", user_id="user-2", type="policy",
                        result_path="storage/p2.policy.json", source_job_id="a2")
    jobs_mod.set_status("p3", "done", user_id="user-1", type="policy",
                        result_path="storage/p3.policy.json", source_job_id="a3")

    monkeypatch.setattr(main_mod.Path, "exists", lambda self: True)

    data = client.get("/policies/by-title/AI Report").json()
    # Both policies for the title, across users; not the Energy Grid one.
    assert {d["job_id"] for d in data} == {"p1", "p2"}
    assert all(d["type"] == "policy" for d in data)
    assert all(d["content"] is None for d in data)


def test_policies_by_title_include_content(client, monkeypatch):
    jobs_mod.set_status("a1", "done", user_id="user-1", title="AI Report", sector="ICT",
                        result_path="storage/a1.analysis.json")
    jobs_mod.set_status("p1", "done", user_id="user-1", type="policy",
                        result_path="storage/p1.policy.json", source_job_id="a1")

    monkeypatch.setattr(main_mod.Path, "exists", lambda self: True)
    monkeypatch.setattr(
        main_mod, "load_json",
        lambda path: {"recommendations": [{"technology": "Edge AI", "actions": []}],
                      "mapping_evidence": {"occupations": [], "skills": []}},
    )

    data = client.get("/policies/by-title/AI Report?include_content=true").json()
    assert len(data) == 1
    assert data[0]["job_id"] == "p1"
    assert data[0]["source_job_id"] == "a1"
    assert data[0]["content"]["recommendations"][0]["technology"] == "Edge AI"


def test_rehydrate_recovers_metadata_from_files(client, tmp_path):
    """
    With an empty registry, rehydrate_from_storage must recover title/sector/
    description from the embedded metadata in the .analysis.json files so the
    catalog endpoints still work after a registry loss.
    """
    import json as _json

    jobs_mod._jobs.clear()
    jid = "11111111-2222-3333-4444-555555555555"
    payload = {
        "technologies": [{"name": "Edge AI"}],
        "title": "Recovered Report",
        "sector": "ICT",
        "description": "from file",
        "created_at": "2026-02-02T00:00:00+00:00",
        "filename": "doc.pdf",
        "user_id": "user-1",
    }
    (tmp_path / f"{jid}.analysis.json").write_text(_json.dumps(payload), encoding="utf-8")

    # Real rehydrate (the client fixture stubs the app's startup call, but we
    # invoke the function directly here against a temp storage dir).
    jobs_mod.rehydrate_from_storage(str(tmp_path))

    info = jobs_mod.get_job(jid)
    assert info is not None
    assert info["title"] == "Recovered Report"
    assert info["sector"] == "ICT"
    assert info["description"] == "from file"
    assert info["created_at"] == "2026-02-02T00:00:00+00:00"

    # And the catalog endpoint now lists the recovered analysis.
    titles = {t["title"] for t in client.get("/analyses/titles").json()}
    assert "Recovered Report" in titles


def test_rehydrate_recovers_policy_linkage_from_files(client, tmp_path):
    """
    Policy results must survive a registry loss too: rehydrate recovers the
    type/source_job_id/user_id embedded in the .policy.json so the per-user
    policy listing (used by the UI to restore recommendations) still works.
    """
    import json as _json

    jobs_mod._jobs.clear()
    pid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    payload = {
        "emerging": [],
        "recommendations": [{"technology": "Edge AI", "actions": []}],
        "mapping_evidence": {"occupations": [], "skills": []},
        "type": "policy",
        "user_id": "user-1",
        "source_job_id": "analysis-1",
        "created_at": "2026-02-03T00:00:00+00:00",
    }
    (tmp_path / f"{pid}.policy.json").write_text(_json.dumps(payload), encoding="utf-8")

    jobs_mod.rehydrate_from_storage(str(tmp_path))

    info = jobs_mod.get_job(pid)
    assert info is not None
    assert info["type"] == "policy"
    assert info["source_job_id"] == "analysis-1"
    assert info["user_id"] == "user-1"

    # The per-user policy listing includes it, with content loaded from disk.
    data = client.get("/users/user-1/policies?include_content=true").json()
    assert len(data) == 1
    assert data[0]["job_id"] == pid
    assert data[0]["source_job_id"] == "analysis-1"
    assert data[0]["content"]["recommendations"][0]["technology"] == "Edge AI"


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

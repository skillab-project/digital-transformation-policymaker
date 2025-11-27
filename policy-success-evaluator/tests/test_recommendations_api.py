import app.main as main
from fastapi.testclient import TestClient

client = TestClient(main.app)

def _fake_llm_output(kpi_id: str):
    # Minimal valid payload matching your schema
    return {
        "kpi_id": kpi_id,
        "recommendations": [
            {
                "lever_type": "Monitoring",          # safe default (works for on-track/off-track after filters)
                "title": "Lightweight Monitoring & Early-Warning",
                "mechanism": "Quarterly progress checks with thresholds and alerting.",
                "rational": "Ensures sustained performance and quick reaction to regressions.",
                "expected_impact": "Medium",
                "time_to_effect": "Short",
                "risks_tradeoffs": "Low admin overhead; risk of complacency.",
                "prerequisites": ["Define thresholds", "Assign monitoring owner"],
            }
        ],
    }

def test_kpi_recommendations_multiple(monkeypatch):
    # Monkeypatch the LLM call so tests are offline & deterministic
    def fake_call(kpi, scope, trend_summary, on_track=None):
        return _fake_llm_output(kpi.id)

    monkeypatch.setattr(main, "call_llm_for_recommendations", fake_call)

    payload = {
        "kpis": [
            {
                "id": "kpi_digital_adoption",
                "name": "SME Digital Adoption Rate",
                "unit": "percentage",
                "direction": "higher_is_better",
                "current_value": 42,
                "target_value": 60,
                "target_deadline": "2026-Q4",
                "time_series": [
                    {"period": "2024-Q1", "value": 38.0},
                    {"period": "2024-Q2", "value": 40.0},
                    {"period": "2024-Q3", "value": 41.0},
                ],
            },
            {
                "id": "kpi_energy_efficiency",
                "name": "Industry Energy Efficiency Index",
                "unit": "index",
                "direction": "higher_is_better",
                "current_value": 70,
                "target_value": 85,
                "target_deadline": "2026-Q4",
                # No time series is also supported
            },
        ],
        "scope": {
            "sector": "Manufacturing",
            "region": "EL52",
            "policy": "Digital & Green Transition",
            "description": "Industrial decarbonisation and digital transformation initiative"
        },
    }

    r = client.post("/kpi/recommendations", json=payload)
    assert r.status_code == 200
    data = r.json()
    # We expect one response per KPI
    assert isinstance(data, list) and len(data) == 2

    for item in data:
        assert "kpi_id" in item
        assert "trend_analysis" in item  # may be None if no time_series
        assert "recommendations" in item and isinstance(item["recommendations"], list)
        assert len(item["recommendations"]) >= 1

        rec = item["recommendations"][0]
        # Check required fields from your Pydantic model
        for key in [
            "lever_type", "title", "mechanism", "rational",
            "expected_impact", "time_to_effect",
            "risks_tradeoffs", "prerequisites"
        ]:
            assert key in rec

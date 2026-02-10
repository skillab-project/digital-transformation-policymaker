import requests
import app.main as main

def test_policy_endpoint_is_online_and_reachable():
    url = f"{main.settings.policy_base}/policy"
    response = requests.get(url, timeout=20)
    assert response.status_code < 500, (
        f"Endpoint appears unavailable: {url} returned {response.status_code}. "
        f"Response body: {response.text[:500]}"
    )


def test_kpi_report_endpoint_is_online_and_reachable():
    url = f"{main.settings.policy_base}/report/kpi"
    response = requests.get(url, timeout=20)
    assert response.status_code < 500, (
        f"Endpoint appears unavailable: {url} returned {response.status_code}. "
        f"Response body: {response.text[:500]}"
    )

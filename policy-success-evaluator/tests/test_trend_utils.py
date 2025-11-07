from app.main import KPI, TimePoint, describe_trend

def test_describe_trend_on_track_true():
    # Higher is better, good upward slope, tiny remaining gap, deadline close → ON_TRACK=True
    kpi = KPI(
        id="kpi1",
        name="Digital Adoption",
        unit="%",
        direction="higher_is_better",
        current_value=54.0,
        target_value=55.0,
        target_deadline="2025-Q4",
        time_series=[
            TimePoint(period="2025-Q1", value=50.0),
            TimePoint(period="2025-Q2", value=52.0),
            TimePoint(period="2025-Q3", value=54.0),
        ],
    )
    trend = describe_trend(kpi)
    assert isinstance(trend, dict)
    assert "trend_summary" in trend
    assert trend["on_track"] is True

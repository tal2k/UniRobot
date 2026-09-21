"""Dashboard metric lists stay consistent (single-source guard)."""

from lab.dashboard import CHART_GROUPS, METRICS


def test_chart_groups_reference_known_metrics():
    known = set(METRICS) | {"Train/mean_reward"}
    for group in CHART_GROUPS:
        for tag in group:
            assert tag in known, f"chart tag without friendly name: {tag}"


def test_page_embeds_generated_groups():
    import json

    from lab.dashboard import PAGE

    assert "__GROUPS_JSON__" not in PAGE
    assert json.dumps(CHART_GROUPS) in PAGE

from app.services.acr_engine import compute_legacy_acr_metrics


def test_legacy_engine_empty(tmp_path):
    assert compute_legacy_acr_metrics(str(tmp_path), "missing") == []

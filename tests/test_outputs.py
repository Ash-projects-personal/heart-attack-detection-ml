import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_all_models_scored():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    assert m["dataset"]["rows_raw"] == 303
    for name, r in m["results"].items():
        assert 0.5 < r["test_accuracy"] <= 1.0, name
        assert 0.5 < r["test_roc_auc"] <= 1.0, name

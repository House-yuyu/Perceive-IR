import json
from pathlib import Path

from scripts.check_perceiveIR_assets import check_assets


def test_inference_does_not_require_training_proxies(tmp_path):
    manifest = json.loads((Path(__file__).parents[1] / "weight/manifest.json").read_text())
    rows = check_assets(tmp_path, manifest, "inference3")
    assert {row["path"] for row in rows} == {
        "weight/perceiveIR_3task.pth", "weight/dinov2-base/config.json",
        "weight/dinov2-base/model.safetensors",
    }
    assert all(row["status"] == "missing" for row in rows)
    for row in rows:
        path = tmp_path / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert all(row["status"] == "missing" for row in check_assets(tmp_path, manifest, "inference3"))
    for row in rows:
        (tmp_path / row["path"]).write_bytes(b"fixture")
    assert all(row["status"] == "present" for row in check_assets(tmp_path, manifest, "inference3"))
    assert any(row["status"] == "missing" for row in check_assets(tmp_path, manifest, "train"))

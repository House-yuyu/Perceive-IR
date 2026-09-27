import json
from collections import Counter
from pathlib import Path

from perceiveIR import evaluate_stage2_adair as evaluation


def test_five_task_enumeration_and_validation_overlap(tmp_path, monkeypatch):
    counts = {"bsd68": 68, "dehaze": 500, "derain": 100,
              "deblur": 1111, "lowlight": 15}

    def fake_list_images(root):
        parts = root.parts
        if "bsd68" in parts:
            name, count = "bsd68", counts["bsd68"]
        elif "dehaze" in parts:
            name, count = "dehaze", counts["dehaze"]
        elif "derain" in parts:
            name, count = "derain", counts["derain"]
        elif "deblur" in parts:
            name, count = "deblur", counts["deblur"]
        else:
            name, count = "lowlight", counts["lowlight"]
        return [root / (f"{index:04d}_haze.png" if name == "dehaze" and root.name == "input"
                        else f"{index:04d}.png") for index in range(count)]

    def fake_index(root):
        return {path.stem: path for path in fake_list_images(root)}

    monkeypatch.setattr(evaluation, "list_images", fake_list_images)
    monkeypatch.setattr(evaluation, "index_by_stem", fake_index)
    root = tmp_path / "data"
    full = evaluation.build_samples(root, None, "adair5")
    assert len(full) == 1930
    assert Counter(row["dataset"] for row in full) == {
        "bsd68_sigma15": 68, "bsd68_sigma25": 68, "bsd68_sigma50": 68,
        "SOTS-Outdoor": 500, "Rain100L": 100, "GoPro": 1111, "LOLv1": 15,
    }
    assert not any(row["used_for_validation"] for row in full)
    marked = [next(row for row in full if row["task"] == task)
              for task in ("deblur", "lowlight")]
    manifest = tmp_path / "validation.json"
    manifest.write_text(json.dumps({"records": marked}))
    selected = evaluation.build_samples(root, manifest, "adair5")
    assert sum(row["used_for_validation"] for row in selected) == 2
    relocated = [{**row, "target": row["target"].replace(str(root), "/other/machine/AiOIR")}
                 for row in marked]
    manifest.write_text(json.dumps({"records": relocated}))
    moved_root = evaluation.build_samples(root, manifest, "adair5")
    assert sum(row["used_for_validation"] for row in moved_root) == 2
    assert len(evaluation.build_samples(root, None, "adair3")) == 804
    assert len(evaluation.build_samples(root, None, "adair5-common")) == 804


def test_summary_separates_image_weighted_and_column_means():
    groups = ("bsd68_sigma15", "bsd68_sigma25", "bsd68_sigma50",
              "SOTS-Outdoor", "Rain100L", "GoPro", "LOLv1")
    rows = [{"dataset": group, "psnr": float(index), "ssim": index / 10,
             "input_psnr": 0.0, "used_for_validation": False}
            for index, group in enumerate(groups, start=1)]
    rows.append({**rows[-1], "psnr": 70.0})
    result = evaluation.summarize(rows, 8, 20_000, "adair5", "checkpoint.pth",
                                  validation_known=False)
    assert result["complete"]
    assert result["column_mean"]["psnr"] == (1 + 2 + 3 + 4 + 5 + 6 + 38.5) / 7
    assert result["summary"]["Average"]["psnr"] == (sum(range(1, 8)) + 70) / 8
    assert result["excluding_validation_scenes"] is None
    assert not result["validation_exclusion_available"]

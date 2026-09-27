from PIL import Image
import pytest

from perceiveIR.data import build_paper_samples
from perceiveIR.stage1_data import build_source_samples


@pytest.mark.parametrize("folder", ["Denoise", "Denosie"])
def test_training_stages_accept_both_denoise_layouts(tmp_path, folder):
    target = tmp_path / "train" / folder / "gt/clean.png"
    target.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(target)
    for task, low, high in (("Dehaze", "scene_haze", "scene"),
                            ("Derain", "rain-1", "norain-1"),
                            ("Deblur", "scene", "scene"),
                            ("Enhance", "scene", "scene")):
        for directory, name in (("input", low), ("gt", high)):
            path = tmp_path / "train" / task / directory / f"{name}.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16)).save(path)
    main_samples = build_paper_samples(tmp_path, tasks=["denoise"])
    stage1_samples = [s for s in build_source_samples(tmp_path) if s.task == "denoise"]
    assert len(main_samples) == 3
    assert {s.noise_sigma for s in main_samples} == {15, 25, 50}
    assert {s.gt_path for s in main_samples} == {target}
    assert len(stage1_samples) == 1 and stage1_samples[0].gt_path == target

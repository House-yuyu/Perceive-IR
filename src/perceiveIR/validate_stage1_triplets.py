from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from .stage1_data import build_source_samples, sample_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate rendered stage-1 quality triplets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    expected_by_fold = Counter(sample_fold(sample) for sample in build_source_samples(config["data"]["root"]))
    manifests = [Path(path) for path in config["prompt"]["manifests"]]
    counts = Counter()
    tasks = Counter()
    medium_paths: set[str] = set()
    errors: list[str] = []

    for manifest in manifests:
        if not manifest.is_file():
            errors.append(f"missing manifest: {manifest}")
            continue
        with open(manifest, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                fold = int(record["fold"])
                counts[fold] += 1
                tasks[record["task"]] += 1
                paths = {key: Path(record[key]) for key in ("low", "medium", "high")}
                medium_key = str(paths["medium"].resolve())
                if medium_key in medium_paths:
                    errors.append(f"duplicate medium path at {manifest}:{line_number}: {medium_key}")
                medium_paths.add(medium_key)
                for key, path in paths.items():
                    if not path.is_file() or path.stat().st_size <= 0:
                        errors.append(f"missing or empty {key} at {manifest}:{line_number}: {path}")
                if args.verify_images and not errors:
                    try:
                        with Image.open(paths["low"]) as low, Image.open(paths["medium"]) as medium, Image.open(
                            paths["high"]
                        ) as high:
                            if low.size != medium.size or medium.size != high.size:
                                errors.append(
                                    f"size mismatch at {manifest}:{line_number}: "
                                    f"low={low.size} medium={medium.size} high={high.size}"
                                )
                            medium.verify()
                    except Exception as error:
                        errors.append(f"invalid image at {manifest}:{line_number}: {error}")
                if len(errors) >= 20:
                    break
        if len(errors) >= 20:
            break

    expected_total = sum(expected_by_fold.values())
    actual_total = sum(counts.values())
    for fold, expected in sorted(expected_by_fold.items()):
        if counts[fold] != expected:
            errors.append(f"fold {fold}: expected {expected} records, found {counts[fold]}")
    if actual_total != expected_total:
        errors.append(f"combined: expected {expected_total} records, found {actual_total}")
    if len(medium_paths) != actual_total:
        errors.append(f"expected {actual_total} unique medium paths, found {len(medium_paths)}")

    report = {
        "valid": not errors,
        "verify_images": bool(args.verify_images),
        "expected_by_fold": dict(sorted(expected_by_fold.items())),
        "actual_by_fold": dict(sorted(counts.items())),
        "task_counts": dict(sorted(tasks.items())),
        "total": actual_total,
        "unique_medium_paths": len(medium_paths),
        "errors": errors,
    }
    output = Path(config["medium"]["triplet_root"]) / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    temporary.replace(output)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

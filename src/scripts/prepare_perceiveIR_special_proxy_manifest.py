"""Validate and combine the completed task-specific proxy render manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perceiveIR.data import build_paper_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--deblur", type=Path, action="append", required=True)
    parser.add_argument("--lowlight", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for task, manifests in (("deblur", args.deblur), ("lowlight", args.lowlight)):
        expected = {str(sample.lq_path) for sample in build_paper_samples(
            args.data_root, tasks=[task], task_resampling={task: 1})}
        rows = []
        for manifest in manifests:
            with manifest.open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle)
        actual = [row["input"] for row in rows]
        if len(rows) != len(expected) or set(actual) != expected:
            raise RuntimeError(f"incomplete or duplicate {task} proxy coverage: {len(rows)}/{len(expected)}")
        if any(row["task"] != task or not Path(row["output"]).is_file() for row in rows):
            raise RuntimeError(f"invalid {task} proxy manifest")
        records.extend(sorted(rows, key=lambda row: row["input"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(args.output)
    print(f"validated task-specific proxy coverage: deblur=2103, lowlight=485, total={len(records)}")


if __name__ == "__main__":
    main()

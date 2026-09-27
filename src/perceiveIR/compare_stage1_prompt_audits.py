from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_stage1_prompts import TASKS


def percent(value: float) -> str:
    return f"{value:.2%}"


def change(previous: float, current: float) -> str:
    return f"{(current - previous) * 100:+.2f} pp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two held-out three-prompt audits.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.baseline).open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    with Path(args.candidate).open("r", encoding="utf-8") as handle:
        candidate = json.load(handle)
    if baseline["test_source_counts"] != candidate["test_source_counts"]:
        raise RuntimeError("audit test counts differ; direct comparison is invalid")
    if baseline["configuration"]["num_crops"] != candidate["configuration"]["num_crops"]:
        raise RuntimeError("audit crop counts differ; direct comparison is invalid")

    old_overall = baseline["overall"]
    new_overall = candidate["overall"]
    lines = [
        "# Three-prompt audit comparison",
        "",
        f"Baseline checkpoint: `{baseline['configuration']['prompt_checkpoint']}`",
        "",
        f"Candidate checkpoint: `{candidate['configuration']['prompt_checkpoint']}`",
        "",
        f"Test triplets: {old_overall['triplets']}; CLIP crops per image: {baseline['configuration']['num_crops']}.",
        "",
        "| Metric | Baseline | Candidate | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, group, key in (
        ("Image accuracy", "overall", "image_accuracy"),
        ("Strict triplet", "overall", "strict_triplet_rate"),
        ("Monotonic ordering", "overall", "monotonic_ordering_rate"),
        ("Task-macro image accuracy", "task_macro", "image_accuracy"),
        ("Task-macro strict triplet", "task_macro", "strict_triplet_rate"),
    ):
        previous = baseline[group][key]
        current = candidate[group][key]
        lines.append(f"| {label} | {percent(previous)} | {percent(current)} | {change(previous, current)} |")

    lines += [
        "",
        "| Task | N | Image accuracy old → new | Medium recall old → new | Strict triplet old → new | Monotonic old → new |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        old = baseline["by_task"][task]
        new = candidate["by_task"][task]
        if old["triplets"] != new["triplets"]:
            raise RuntimeError(f"audit sample count differs for {task}")
        old_medium = old["per_quality_accuracy"]["mediocre"]
        new_medium = new["per_quality_accuracy"]["mediocre"]
        lines.append(
            f"| {task} | {old['triplets']} | "
            f"{percent(old['image_accuracy'])} → {percent(new['image_accuracy'])} | "
            f"{percent(old_medium)} → {percent(new_medium)} | "
            f"{percent(old['strict_triplet_rate'])} → {percent(new['strict_triplet_rate'])} | "
            f"{percent(old['monotonic_ordering_rate'])} → {percent(new['monotonic_ordering_rate'])} |"
        )

    lines += [
        "",
        f"Engineering gate: {'PASS' if candidate['suggested_gate']['passed'] else 'FAIL'}.",
        "",
        "The same AiO test images were used for checkpoint selection, so this comparison is diagnostic rather than a pristine final benchmark.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

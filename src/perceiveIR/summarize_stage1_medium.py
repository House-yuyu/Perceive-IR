from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASKS = ("denoise", "dehaze", "derain", "deblur", "lowlight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate medium-quality audit results.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_metrics(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(root.glob("fold_*/step_*/metrics.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    if not records:
        raise RuntimeError(f"no metrics.jsonl files found below {root}")
    return pd.DataFrame.from_records(records)


def task_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (step, task), group in data.groupby(["step", "task"], sort=True):
        progress = group["progress"]
        rows.append(
            {
                "step": int(step),
                "task": task,
                "count": len(group),
                "psnr_low_mean": group["psnr_low"].mean(),
                "psnr_medium_mean": group["psnr_medium"].mean(),
                "delta_psnr_mean": group["delta_psnr"].mean(),
                "psnr_improved_rate": (group["delta_psnr"] > 0).mean(),
                "ssim_low_mean": group["ssim_low"].mean(),
                "ssim_medium_mean": group["ssim_medium"].mean(),
                "delta_ssim_mean": group["delta_ssim"].mean(),
                "ssim_improved_rate": (group["delta_ssim"] > 0).mean(),
                "lpips_low_mean": group["lpips_low"].mean(),
                "lpips_medium_mean": group["lpips_medium"].mean(),
                "lpips_gain_mean": group["lpips_gain"].mean(),
                "lpips_improved_rate": (group["lpips_gain"] > 0).mean(),
                "progress_q10": progress.quantile(0.10),
                "progress_median": progress.median(),
                "progress_q90": progress.quantile(0.90),
                "intermediate_rate_005_095": ((progress > 0.05) & (progress < 0.95)).mean(),
            }
        )
    return pd.DataFrame(rows)


def checkpoint_summary(by_task: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step, group in by_task.groupby("step", sort=True):
        progress_center = group["progress_median"].mean()
        minimum_ordering = min(
            group["psnr_improved_rate"].min(),
            group["ssim_improved_rate"].min(),
            group["lpips_improved_rate"].min(),
        )
        intermediate = group["intermediate_rate_005_095"].mean()
        center_score = max(0.0, 1.0 - abs(progress_center - 0.55) / 0.55)
        rows.append(
            {
                "step": int(step),
                "macro_delta_psnr": group["delta_psnr_mean"].mean(),
                "macro_delta_ssim": group["delta_ssim_mean"].mean(),
                "macro_lpips_gain": group["lpips_gain_mean"].mean(),
                "minimum_task_delta_psnr": group["delta_psnr_mean"].min(),
                "minimum_task_delta_ssim": group["delta_ssim_mean"].min(),
                "minimum_task_lpips_gain": group["lpips_gain_mean"].min(),
                "minimum_psnr_improved_rate": group["psnr_improved_rate"].min(),
                "minimum_ssim_improved_rate": group["ssim_improved_rate"].min(),
                "minimum_lpips_improved_rate": group["lpips_improved_rate"].min(),
                "macro_progress_median": progress_center,
                "macro_intermediate_rate_005_095": intermediate,
                "passes_positive_psnr_gate": bool((group["delta_psnr_mean"] > 0).all()),
                "passes_positive_all_metrics_gate": bool(
                    (group["delta_psnr_mean"] > 0).all()
                    and (group["delta_ssim_mean"] > 0).all()
                    and (group["lpips_gain_mean"] > 0).all()
                ),
                "passes_initial_gate": bool(
                    (group["delta_psnr_mean"] > 0).all()
                    and (group["psnr_improved_rate"] >= 0.90).all()
                    and (group["ssim_improved_rate"] >= 0.90).all()
                    and (group["lpips_improved_rate"] >= 0.85).all()
                ),
                "selection_score": 0.45 * minimum_ordering + 0.35 * intermediate + 0.20 * center_score,
            }
        )
    return pd.DataFrame(rows).sort_values("step")


def plot_results(by_task: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for task in TASKS:
        rows = by_task[by_task["task"] == task]
        axes[0].plot(rows["step"], rows["progress_median"], marker="o", label=task)
        axes[1].plot(rows["step"], rows["psnr_improved_rate"], marker="o", label=task)
    axes[0].axhspan(0.25, 0.85, color="green", alpha=0.08)
    axes[0].set(title="Median restoration progress", xlabel="optimizer step", ylabel="progress r")
    axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="PSNR ordering pass rate", xlabel="optimizer step", ylabel="fraction improved", ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_metrics(Path(args.input_root))
    by_task = task_summary(data)
    by_checkpoint = checkpoint_summary(by_task)
    by_task.to_csv(output / "summary_by_task.csv", index=False)
    by_checkpoint.to_csv(output / "summary_by_checkpoint.csv", index=False)
    plot_results(by_task, output / "checkpoint_comparison.png")

    eligible = by_checkpoint[by_checkpoint["passes_positive_psnr_gate"]]
    candidates = eligible if not eligible.empty else by_checkpoint
    selected = candidates.sort_values("selection_score", ascending=False).iloc[0]
    recommendation = {
        "recommended_step": int(selected["step"]),
        "passed_positive_psnr_gate": bool(selected["passes_positive_psnr_gate"]),
        "passed_positive_all_metrics_gate": bool(selected["passes_positive_all_metrics_gate"]),
        "passed_initial_gate": bool(selected["passes_initial_gate"]),
        "selection_score": float(selected["selection_score"]),
        "note": "The score is an engineering aid; inspect per-task contact sheets before final selection.",
    }
    with open(output / "recommendation.json", "w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, indent=2)
    print(by_checkpoint.to_string(index=False))
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()

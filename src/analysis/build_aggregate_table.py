"""Build the aggregate results table from raw experiment CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASETS = ("imagenette", "imagewoof")
CODECS = ("jpeg", "webp")
QUALITIES_DESC = [95, 80, 60, 40, 20, 10]
OUTPUT_COLUMNS = [
    "dataset",
    "codec",
    "quality",
    "n_images",
    "original_size_bytes",
    "compressed_size_bytes",
    "compression_ratio",
    "size_reduction_pct",
    "baseline_accuracy_pct",
    "compressed_accuracy_pct",
    "accuracy_change_pp",
    "mean_psnr",
    "mean_ssim",
]


def require_columns(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """Fail clearly when a required CSV column is missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def load_baseline(results_root: Path, dataset: str) -> pd.DataFrame:
    """Load one baseline CSV."""
    path = results_root / dataset / f"{dataset}_baseline.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    require_columns(frame, path, ["path", "is_correct"])
    return frame[["path", "is_correct"]].rename(
        columns={"is_correct": "baseline_is_correct"}
    )


def load_compression_results(results_root: Path, dataset: str, codec: str) -> pd.DataFrame:
    """Load one compression CSV."""
    path = results_root / dataset / "compression" / f"{dataset}_{codec}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    columns = [
        "dataset",
        "path",
        "codec",
        "quality",
        "is_correct",
        "original_size_bytes",
        "compressed_size_bytes",
        "psnr",
        "ssim",
    ]
    frame = pd.read_csv(path)
    require_columns(frame, path, columns)
    return frame[columns].rename(columns={"is_correct": "compressed_is_correct"})


def aggregate_results(results_root: Path) -> pd.DataFrame:
    """Aggregate raw per-image baseline and compression rows."""
    rows = []

    for dataset in DATASETS:
        baseline = load_baseline(results_root, dataset)
        for codec in CODECS:
            compressed = load_compression_results(results_root, dataset, codec)
            merged = compressed.merge(baseline, on="path", how="left", validate="many_to_one")
            if merged["baseline_is_correct"].isna().any():
                raise ValueError(f"Could not match all {dataset} {codec} rows to baseline paths")

            for quality in QUALITIES_DESC:
                group = merged[merged["quality"] == quality]
                original_size = int(group["original_size_bytes"].sum())
                compressed_size = int(group["compressed_size_bytes"].sum())
                baseline_accuracy = group["baseline_is_correct"].mean() * 100
                compressed_accuracy = group["compressed_is_correct"].mean() * 100

                rows.append(
                    {
                        "dataset": dataset,
                        "codec": codec,
                        "quality": int(quality),
                        "n_images": int(len(group)),
                        "original_size_bytes": original_size,
                        "compressed_size_bytes": compressed_size,
                        "compression_ratio": original_size / compressed_size,
                        "size_reduction_pct": (1 - compressed_size / original_size) * 100,
                        "baseline_accuracy_pct": baseline_accuracy,
                        "compressed_accuracy_pct": compressed_accuracy,
                        "accuracy_change_pp": compressed_accuracy - baseline_accuracy,
                        "mean_psnr": group["psnr"].mean(),
                        "mean_ssim": group["ssim"].mean(),
                    }
                )

    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return table.sort_values(
        by=["dataset", "codec", "quality"],
        ascending=[True, True, False],
    )


def main() -> None:
    """Parse arguments and write the aggregate CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/aggregate_results.csv"),
    )
    args = parser.parse_args()

    table = aggregate_results(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.6f")

    print(f"Output path: {args.output}")
    print(f"Generated rows: {len(table)}")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()

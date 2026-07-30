"""Render visual aggregate result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ("imagenette", "imagewoof")
CODECS = ("jpeg", "webp")
QUALITIES_DESC = [95, 80, 60, 40, 20, 10]
DISPLAY_COLUMNS = [
    "Codec",
    "Quality",
    "Compression ratio",
    "Size reduction",
    "Accuracy",
    "Accuracy change",
    "PSNR",
    "SSIM",
]
REQUIRED_COLUMNS = [
    "dataset",
    "codec",
    "quality",
    "n_images",
    "original_size_bytes",
    "compression_ratio",
    "size_reduction_pct",
    "baseline_accuracy_pct",
    "compressed_accuracy_pct",
    "accuracy_change_pp",
    "mean_psnr",
    "mean_ssim",
]


def load_aggregate_results(path: Path) -> pd.DataFrame:
    """Load the aggregate CSV used by the renderer."""
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame


def format_signed_pp(value: float) -> str:
    """Format percentage-point changes with a visible sign."""
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.2f} pp"


def format_dataset_table(frame: pd.DataFrame, dataset: str) -> tuple[str, str, pd.DataFrame]:
    """Return the title, summary, and formatted rows for one dataset."""
    dataset_frame = frame[frame["dataset"] == dataset].copy()
    dataset_frame["codec_order"] = dataset_frame["codec"].map({"jpeg": 0, "webp": 1})
    dataset_frame["quality_order"] = dataset_frame["quality"].map(
        {quality: index for index, quality in enumerate(QUALITIES_DESC)}
    )
    dataset_frame = dataset_frame.sort_values(["codec_order", "quality_order"])

    first_row = dataset_frame.iloc[0]
    title = "Imagenette" if dataset == "imagenette" else "ImageWoof"
    summary = (
        f"Baseline accuracy: {first_row['baseline_accuracy_pct']:.2f}% | "
        f"Images: {int(first_row['n_images']):,} | "
        f"Original dataset size: {first_row['original_size_bytes'] / (1024 ** 2):.2f} MiB"
    )

    formatted = pd.DataFrame(
        {
            "Codec": dataset_frame["codec"].map({"jpeg": "JPEG", "webp": "WebP"}),
            "Quality": dataset_frame["quality"].astype(int).astype(str),
            "Compression ratio": dataset_frame["compression_ratio"].map(
                lambda value: f"{value:.2f}×"
            ),
            "Size reduction": dataset_frame["size_reduction_pct"].map(
                lambda value: f"{value:.1f}%"
            ),
            "Accuracy": dataset_frame["compressed_accuracy_pct"].map(
                lambda value: f"{value:.2f}%"
            ),
            "Accuracy change": dataset_frame["accuracy_change_pp"].map(format_signed_pp),
            "PSNR": dataset_frame["mean_psnr"].map(lambda value: f"{value:.2f} dB"),
            "SSIM": dataset_frame["mean_ssim"].map(lambda value: f"{value:.4f}"),
        }
    )
    return title, summary, formatted


def size_reduction_color(value: float) -> str:
    """Return restrained background color for size reduction cells."""
    if value <= 0:
        return "#f9eeee"
    return "#edf7ef"


def accuracy_change_color(value: float) -> str:
    """Return restrained background color for accuracy-change cells."""
    if value > 0:
        return "#edf7ef"
    strength = min(abs(value) / 16, 1)
    red = int(252 - 10 * strength)
    green = int(246 - 34 * strength)
    blue = int(237 - 48 * strength)
    return f"#{red:02x}{green:02x}{blue:02x}"


def render_dataset_table(frame: pd.DataFrame, dataset: str, output_dir: Path) -> Path:
    """Render one dataset table image."""
    title, summary, formatted = format_dataset_table(frame, dataset)
    raw_rows = frame[frame["dataset"] == dataset].copy()
    raw_rows["codec_order"] = raw_rows["codec"].map({"jpeg": 0, "webp": 1})
    raw_rows["quality_order"] = raw_rows["quality"].map(
        {quality: index for index, quality in enumerate(QUALITIES_DESC)}
    )
    raw_rows = raw_rows.sort_values(["codec_order", "quality_order"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12.4, 6.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    fig.text(0.045, 0.955, title, fontsize=18, fontweight="bold", ha="left", va="top")
    fig.text(0.045, 0.905, summary, fontsize=11.5, color="#333333", ha="left", va="top")

    table = ax.table(
        cellText=formatted.values,
        colLabels=DISPLAY_COLUMNS,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.02, 0.08, 0.96, 0.89],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    widths = [0.11, 0.09, 0.17, 0.15, 0.12, 0.16, 0.10, 0.10]
    for (row, column), cell in table.get_celld().items():
        cell.set_width(widths[column])
        cell.set_edgecolor("#d8dde3")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#263342")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.072)
            continue

        source_index = row - 1
        base_color = "#ffffff" if source_index % 2 == 0 else "#f7f9fb"
        cell.set_facecolor(base_color)
        cell.set_height(0.064)
        if source_index == 6:
            cell.set_linewidth(1.2)
            cell.set_edgecolor("#aeb7c2")
        if column == 3:
            cell.set_facecolor(
                size_reduction_color(float(raw_rows.loc[source_index, "size_reduction_pct"]))
            )
        elif column == 5:
            cell.set_facecolor(
                accuracy_change_color(float(raw_rows.loc[source_index, "accuracy_change_pp"]))
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_results_table.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Parse arguments and render both dataset tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/analysis/aggregate_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/tables"))
    args = parser.parse_args()

    frame = load_aggregate_results(args.input)
    paths = [render_dataset_table(frame, dataset, args.output_dir) for dataset in DATASETS]
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

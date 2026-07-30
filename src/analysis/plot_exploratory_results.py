"""Plot exploratory figures from aggregate experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = ("imagenette", "imagewoof")
CODECS = ("jpeg", "webp")
DATASET_TITLES = {"imagenette": "Imagenette", "imagewoof": "ImageWoof"}
CODEC_STYLES = {
    "jpeg": {"label": "JPEG", "color": "#1f4e79", "marker": "o", "linestyle": "-"},
    "webp": {"label": "WebP", "color": "#8a5a00", "marker": "s", "linestyle": "--"},
}
ANNOTATION_OFFSETS = {
    "jpeg": {
        95: (-14, 10),
        80: (6, -13),
        60: (6, 8),
        40: (6, -13),
        20: (6, 8),
        10: (6, -13),
    },
    "webp": {
        95: (8, -15),
        80: (8, 9),
        60: (8, -13),
        40: (8, 9),
        20: (8, 9),
        10: (8, -13),
    },
}
SSIM_ANNOTATION_OFFSETS = {
    "jpeg": {
        95: (-14, 12),
        80: (6, 8),
        60: (-8, -18),
        40: (6, -13),
        20: (6, 8),
        10: (6, -13),
    },
    "webp": {
        95: (8, -18),
        80: (12, -14),
        60: (8, 9),
        40: (8, 9),
        20: (8, 9),
        10: (8, -13),
    },
}
REQUIRED_COLUMNS = [
    "dataset",
    "codec",
    "quality",
    "compression_ratio",
    "accuracy_change_pp",
    "mean_psnr",
    "mean_ssim",
]


def load_aggregate_results(path: Path) -> pd.DataFrame:
    """Load the aggregate CSV required for plotting."""
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame


def sorted_codec_rows(frame: pd.DataFrame, dataset: str, codec: str) -> pd.DataFrame:
    """Return rows for one dataset and codec sorted by compression ratio."""
    rows = frame[(frame["dataset"] == dataset) & (frame["codec"] == codec)].copy()
    return rows.sort_values("compression_ratio")


def padded_limits(values: pd.Series, lower_reference: float | None = None) -> tuple[float, float]:
    """Return padded axis limits for a numeric series."""
    minimum = float(values.min())
    maximum = float(values.max())
    if lower_reference is not None:
        minimum = min(minimum, lower_reference)
        maximum = max(maximum, lower_reference)
    padding = (maximum - minimum) * 0.08 if maximum > minimum else 1.0
    return minimum - padding, maximum + padding


def annotate_quality(ax: plt.Axes, row: pd.Series, codec: str, metric: str | None = None) -> None:
    """Annotate a point with its codec quality."""
    quality = int(row["quality"])
    offsets = SSIM_ANNOTATION_OFFSETS if metric == "mean_ssim" else ANNOTATION_OFFSETS
    dx, dy = offsets[codec][quality]
    ax.annotate(
        f"q{quality}",
        (row["compression_ratio"], row["y_value"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        color="#222222",
        clip_on=False,
    )


def apply_common_style(ax: plt.Axes) -> None:
    """Apply shared styling to a plot panel."""
    ax.grid(True, color="#d9dee5", linewidth=0.7, alpha=0.85)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#9aa4af")
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=9)


def plot_rate_accuracy(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot accuracy change against compression ratio."""
    x_limits = padded_limits(frame["compression_ratio"], lower_reference=1.0)
    y_limits = padded_limits(frame["accuracy_change_pp"], lower_reference=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    legend_handles = []

    for ax, dataset in zip(axes, DATASETS):
        apply_common_style(ax)
        ax.axhline(0, color="#555555", linewidth=0.9, linestyle=":")
        ax.axvline(1, color="#555555", linewidth=0.9, linestyle=":")
        baseline = ax.scatter(
            [1.0],
            [0.0],
            color="#333333",
            marker="D",
            s=46,
            label="Original baseline",
            zorder=4,
        )

        for codec in CODECS:
            rows = sorted_codec_rows(frame, dataset, codec).copy()
            rows["y_value"] = rows["accuracy_change_pp"]
            style = CODEC_STYLES[codec]
            (line,) = ax.plot(
                rows["compression_ratio"],
                rows["accuracy_change_pp"],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=5.5,
            )
            if dataset == DATASETS[0]:
                legend_handles.append(line)
            for _, row in rows.iterrows():
                annotate_quality(ax, row, codec)

        if dataset == DATASETS[0]:
            legend_handles.append(baseline)
        ax.set_title(DATASET_TITLES[dataset], fontsize=12, fontweight="bold")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel("Total compression ratio", fontsize=10)

    axes[0].set_ylabel("Accuracy change from baseline (percentage points)", fontsize=10)
    fig.suptitle("Classification performance versus compression ratio", fontsize=15, fontweight="bold")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rate_accuracy_tradeoff.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rate_distortion(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot PSNR and SSIM against compression ratio."""
    x_limits = padded_limits(frame["compression_ratio"])
    psnr_limits = padded_limits(frame["mean_psnr"])
    ssim_limits = padded_limits(frame["mean_ssim"])

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), sharex=True)
    fig.patch.set_facecolor("white")
    legend_handles = []

    for column, dataset in enumerate(DATASETS):
        for row_index, metric in enumerate(("mean_psnr", "mean_ssim")):
            ax = axes[row_index, column]
            apply_common_style(ax)
            for codec in CODECS:
                rows = sorted_codec_rows(frame, dataset, codec).copy()
                rows["y_value"] = rows[metric]
                style = CODEC_STYLES[codec]
                (line,) = ax.plot(
                    rows["compression_ratio"],
                    rows[metric],
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=5.5,
                )
                if column == 0 and row_index == 0:
                    legend_handles.append(line)
                for _, point in rows.iterrows():
                    annotate_quality(ax, point, codec, metric)

            ax.set_xlim(x_limits)
            if metric == "mean_psnr":
                ax.set_ylim(psnr_limits)
                ax.set_title(f"{DATASET_TITLES[dataset]} - PSNR", fontsize=12, fontweight="bold")
                ax.set_ylabel("Mean PSNR (dB)", fontsize=10)
            else:
                ax.set_ylim(ssim_limits)
                ax.set_title(f"{DATASET_TITLES[dataset]} - SSIM", fontsize=12, fontweight="bold")
                ax.set_ylabel("Mean SSIM", fontsize=10)
                ax.set_xlabel("Total compression ratio", fontsize=10)

    fig.suptitle("Rate-distortion behaviour of JPEG and WebP", fontsize=15, fontweight="bold")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rate_distortion_tradeoff.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Parse arguments and generate the exploratory figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/analysis/aggregate_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/plots"))
    args = parser.parse_args()

    frame = load_aggregate_results(args.input)
    paths = [
        plot_rate_accuracy(frame, args.output_dir),
        plot_rate_distortion(frame, args.output_dir),
    ]
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

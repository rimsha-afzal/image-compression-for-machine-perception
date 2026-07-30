"""Analyze the targeted threshold-refinement compression conditions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_ROOT = Path("results")
OUTPUT_ROOT = RESULTS_ROOT / "analysis" / "refinement"
NEW_CONDITIONS_PATH = OUTPUT_ROOT / "new_conditions.csv"
TABLE_DIR = OUTPUT_ROOT / "tables"
PLOT_PATH = OUTPUT_ROOT / "threshold_refinement.png"

DATASETS = ("imagenette", "imagewoof")
DATASET_TITLES = {"imagenette": "Imagenette", "imagewoof": "ImageWoof"}
CODEC_TITLES = {"jpeg": "JPEG", "webp": "WebP"}
NEW_CONDITIONS = (
    ("imagenette", "jpeg", 70, RESULTS_ROOT / "imagenette" / "compression" / "refinement" / "imagenette_jpeg_q70.csv"),
    ("imagewoof", "jpeg", 70, RESULTS_ROOT / "imagewoof" / "compression" / "refinement" / "imagewoof_jpeg_q70.csv"),
    ("imagenette", "webp", 77, RESULTS_ROOT / "imagenette" / "compression" / "refinement" / "imagenette_webp_q77.csv"),
    ("imagewoof", "webp", 77, RESULTS_ROOT / "imagewoof" / "compression" / "refinement" / "imagewoof_webp_q77.csv"),
)
OUTPUT_COLUMNS = [
    "dataset",
    "codec",
    "quality",
    "n_images",
    "compression_ratio",
    "size_reduction_pct",
    "baseline_accuracy_pct",
    "compressed_accuracy_pct",
    "accuracy_change_pp",
    "accuracy_loss_pp",
    "mean_psnr",
    "mean_ssim",
    "within_0_5pp",
    "within_1pp",
    "within_2pp",
]
RAW_COLUMNS = [
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
TABLE_COLUMNS = [
    "Codec",
    "Quality",
    "Compression ratio",
    "Size reduction",
    "Accuracy",
    "Accuracy loss",
    "PSNR",
    "SSIM",
    "Primary 1 pp result",
]
CODEC_STYLES = {
    "jpeg": {"label": "JPEG", "color": "#1f4e79", "marker": "o", "linestyle": "-"},
    "webp": {"label": "WebP", "color": "#8a5a00", "marker": "s", "linestyle": "--"},
}
ANNOTATION_OFFSETS = {
    ("jpeg", 60): (7, 8),
    ("jpeg", 70): (7, -14),
    ("webp", 80): (7, -16),
    ("webp", 77): (7, 10),
}


def require_columns(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def load_baseline(dataset: str) -> pd.DataFrame:
    path = RESULTS_ROOT / dataset / f"{dataset}_baseline.csv"
    frame = pd.read_csv(path)
    require_columns(frame, path, ["path", "is_correct"])
    return frame[["path", "is_correct"]].rename(columns={"is_correct": "baseline_is_correct"})


def load_refinement_csv(dataset: str, codec: str, quality: int, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, path, RAW_COLUMNS)
    unexpected = frame[
        (frame["dataset"] != dataset)
        | (frame["codec"] != codec)
        | (frame["quality"].astype(int) != quality)
    ]
    if not unexpected.empty:
        raise ValueError(f"{path} contains rows outside {dataset} {codec} q{quality}")
    return frame[RAW_COLUMNS].rename(columns={"is_correct": "compressed_is_correct"})


def summarize_condition(dataset: str, codec: str, quality: int, path: Path) -> dict[str, object]:
    baseline = load_baseline(dataset)
    compressed = load_refinement_csv(dataset, codec, quality, path)
    merged = compressed.merge(baseline, on="path", how="left", validate="one_to_one")
    if merged["baseline_is_correct"].isna().any():
        raise ValueError(f"Could not match all {dataset} {codec} q{quality} rows to baseline paths")

    original_size = int(merged["original_size_bytes"].sum())
    compressed_size = int(merged["compressed_size_bytes"].sum())
    compression_ratio = original_size / compressed_size
    baseline_accuracy = merged["baseline_is_correct"].mean() * 100
    compressed_accuracy = merged["compressed_is_correct"].mean() * 100
    accuracy_change = compressed_accuracy - baseline_accuracy
    accuracy_loss = baseline_accuracy - compressed_accuracy

    row = {
        "dataset": dataset,
        "codec": codec,
        "quality": int(quality),
        "n_images": int(len(merged)),
        "compression_ratio": compression_ratio,
        "size_reduction_pct": (1 - compressed_size / original_size) * 100,
        "baseline_accuracy_pct": baseline_accuracy,
        "compressed_accuracy_pct": compressed_accuracy,
        "accuracy_change_pp": accuracy_change,
        "accuracy_loss_pp": accuracy_loss,
        "mean_psnr": merged["psnr"].mean(),
        "mean_ssim": merged["ssim"].mean(),
    }
    for label, tolerance in (("within_0_5pp", 0.5), ("within_1pp", 1.0), ("within_2pp", 2.0)):
        row[label] = bool(compression_ratio > 1 and accuracy_loss <= tolerance)
    return row


def build_new_conditions() -> pd.DataFrame:
    rows = [summarize_condition(*condition) for condition in NEW_CONDITIONS]
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return frame.sort_values(["dataset", "codec"]).reset_index(drop=True)


def format_table_rows(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = frame[frame["dataset"] == dataset].copy()
    rows["codec_order"] = rows["codec"].map({"jpeg": 0, "webp": 1})
    rows = rows.sort_values(["codec_order", "quality"]).reset_index(drop=True)
    return pd.DataFrame(
        {
            "Codec": rows["codec"].map(CODEC_TITLES),
            "Quality": rows["quality"].astype(int).astype(str),
            "Compression ratio": rows["compression_ratio"].map(lambda value: f"{value:.2f}×"),
            "Size reduction": rows["size_reduction_pct"].map(lambda value: f"{value:.1f}%"),
            "Accuracy": rows["compressed_accuracy_pct"].map(lambda value: f"{value:.2f}%"),
            "Accuracy loss": rows["accuracy_loss_pp"].map(lambda value: f"{value:.2f} pp"),
            "PSNR": rows["mean_psnr"].map(lambda value: f"{value:.2f} dB"),
            "SSIM": rows["mean_ssim"].map(lambda value: f"{value:.4f}"),
            "Primary 1 pp result": rows["within_1pp"].map(lambda value: "Pass" if value else "Fail"),
        }
    )


def render_table(frame: pd.DataFrame, dataset: str) -> Path:
    formatted = format_table_rows(frame, dataset)
    raw_rows = frame[frame["dataset"] == dataset].copy()
    raw_rows["codec_order"] = raw_rows["codec"].map({"jpeg": 0, "webp": 1})
    raw_rows = raw_rows.sort_values(["codec_order", "quality"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13.2, 2.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    title = DATASET_TITLES[dataset]
    baseline_accuracy = raw_rows.loc[0, "baseline_accuracy_pct"]
    fig.text(0.045, 0.94, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(
        0.045,
        0.82,
        f"Baseline accuracy: {baseline_accuracy:.2f}% | Primary tolerance: 1 pp",
        fontsize=10.5,
        color="#333333",
        ha="left",
        va="top",
    )

    table = ax.table(
        cellText=formatted.values,
        colLabels=TABLE_COLUMNS,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.02, 0.06, 0.96, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.25)

    widths = [0.10, 0.08, 0.17, 0.14, 0.11, 0.13, 0.10, 0.09, 0.20]
    for (row, column), cell in table.get_celld().items():
        cell.set_width(widths[column])
        cell.set_edgecolor("#d8dde3")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#263342")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.13)
            continue

        source_index = row - 1
        cell.set_height(0.12)
        cell.set_facecolor("#ffffff" if source_index % 2 == 0 else "#f7f9fb")
        if column == 8:
            passed = bool(raw_rows.loc[source_index, "within_1pp"])
            cell.set_facecolor("#e8f5e9" if passed else "#fbeaea")
            cell.set_text_props(weight="bold", color="#1f4d2c" if passed else "#7a2727")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / f"{dataset}_new_conditions.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_reference_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(
        frame,
        path,
        ["dataset", "codec", "quality", "compression_ratio", "accuracy_change_pp"],
    )
    reference = frame[
        ((frame["codec"] == "jpeg") & (frame["quality"].astype(int) == 60))
        | ((frame["codec"] == "webp") & (frame["quality"].astype(int) == 80))
    ].copy()
    reference["accuracy_loss_pp"] = -reference["accuracy_change_pp"]
    return reference[["dataset", "codec", "quality", "compression_ratio", "accuracy_loss_pp"]]


def build_plot_frame(new_frame: pd.DataFrame) -> pd.DataFrame:
    reference = load_reference_rows(RESULTS_ROOT / "analysis" / "aggregate_results.csv")
    new_rows = new_frame[["dataset", "codec", "quality", "compression_ratio", "accuracy_loss_pp"]]
    frame = pd.concat([reference, new_rows], ignore_index=True)
    allowed = {("jpeg", 60), ("jpeg", 70), ("webp", 80), ("webp", 77)}
    frame = frame[frame.apply(lambda row: (row["codec"], int(row["quality"])) in allowed, axis=1)]
    return frame


def apply_plot_style(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9dee5", linewidth=0.7, alpha=0.85)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#9aa4af")
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=9)


def render_refinement_plot(new_frame: pd.DataFrame) -> Path:
    plot_frame = build_plot_frame(new_frame)
    x_min = min(0.9, float(plot_frame["compression_ratio"].min()) - 0.08)
    x_max = float(plot_frame["compression_ratio"].max()) + 0.18
    y_min = min(0.0, float(plot_frame["accuracy_loss_pp"].min()) - 0.15)
    y_max = max(2.2, float(plot_frame["accuracy_loss_pp"].max()) + 0.35)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    legend_handles = []

    for ax, dataset in zip(axes, DATASETS):
        apply_plot_style(ax)
        ax.axhline(0.5, color="#8b949e", linewidth=0.9, linestyle=":", zorder=1)
        ax.axhline(1.0, color="#3a3a3a", linewidth=1.4, linestyle="-.", zorder=1)
        ax.axhline(2.0, color="#8b949e", linewidth=0.9, linestyle=":", zorder=1)
        ax.axvline(1.0, color="#555555", linewidth=0.9, linestyle=":", zorder=1)
        ax.text(x_min + 0.02, 0.52, "0.5 pp", fontsize=8, color="#555555", va="bottom")
        ax.text(x_min + 0.02, 1.02, "1 pp", fontsize=8.5, color="#222222", va="bottom", fontweight="bold")
        ax.text(x_min + 0.02, 2.02, "2 pp", fontsize=8, color="#555555", va="bottom")

        for codec in ("jpeg", "webp"):
            rows = plot_frame[(plot_frame["dataset"] == dataset) & (plot_frame["codec"] == codec)].copy()
            rows = rows.sort_values("compression_ratio")
            style = CODEC_STYLES[codec]
            (line,) = ax.plot(
                rows["compression_ratio"],
                rows["accuracy_loss_pp"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.9,
                markersize=6.0,
                label=style["label"],
                zorder=3,
            )
            if dataset == DATASETS[0]:
                legend_handles.append(line)
            for _, row in rows.iterrows():
                quality = int(row["quality"])
                dx, dy = ANNOTATION_OFFSETS[(codec, quality)]
                ax.annotate(
                    f"{style['label']} q{quality}",
                    (row["compression_ratio"], row["accuracy_loss_pp"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.2,
                    color="#222222",
                    clip_on=False,
                )

        ax.set_title(DATASET_TITLES[dataset], fontsize=12, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Total compression ratio", fontsize=10)

    axes[0].set_ylabel("Accuracy loss from baseline (percentage points)", fontsize=10)
    fig.suptitle("Targeted threshold refinement", fontsize=15, fontweight="bold")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return PLOT_PATH


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = build_new_conditions()
    frame.to_csv(NEW_CONDITIONS_PATH, index=False, float_format="%.6f")

    paths = [NEW_CONDITIONS_PATH]
    paths.extend(render_table(frame, dataset) for dataset in DATASETS)
    paths.append(render_refinement_plot(frame))

    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

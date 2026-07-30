"""Run compressed-image ResNet-50 classification experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from compression import SUPPORTED_CODECS, compress_image, image_quality_metrics
from load_datasets import DATASETS, image_files
from load_resnet50 import build_preprocess, load_resnet50, predict_image


DEFAULT_QUALITIES = [95, 80, 60, 40, 20, 10]


# Yield validation images for a dataset together with their target ImageNet indices.
def iter_validation_images(dataset: str, data_dir: Path):
    dataset_root = data_dir / DATASETS[dataset]["root"]
    val_dir = dataset_root / "val"
    class_mapping = DATASETS[dataset]["classes"]

    for image_path in image_files(val_dir):
        class_id = image_path.parent.name
        if class_id not in class_mapping:
            raise ValueError(f"No ImageNet mapping for class folder: {class_id}")
        target_index, target_label = class_mapping[class_id]
        yield image_path, class_id, target_index, target_label


def validate_qualities(qualities: list[int]) -> list[int]:
    for quality in qualities:
        if not 1 <= quality <= 100:
            raise ValueError("quality values must be between 1 and 100")
    return qualities


# Classify compressed validation images and write one CSV row per image and quality.
def run_compression_experiment(
    dataset: str,
    codec: str,
    qualities: list[int],
    data_dir: Path,
    output_csv: Path,
    model,
    class_names: list[str],
    preprocess,
    device: str | None,
    limit: int | None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    normalized_codec = codec.lower()

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "path",
                "class_id",
                "target_index",
                "target_label",
                "codec",
                "quality",
                "predicted_index",
                "predicted_class",
                "top1_confidence",
                "is_correct",
                "original_size_bytes",
                "compressed_size_bytes",
                "compression_ratio",
                "psnr",
                "ssim",
            ],
        )
        writer.writeheader()

        for count, (image_path, class_id, target_index, target_label) in enumerate(
            iter_validation_images(dataset, data_dir),
            start=1,
        ):
            if limit is not None and count > limit:
                break

            original_size = image_path.stat().st_size
            with Image.open(image_path) as original:
                original_rgb = original.convert("RGB")
                for quality in qualities:
                    compressed_image, compressed_size = compress_image(
                        original_rgb,
                        normalized_codec,
                        quality,
                    )
                    psnr, ssim = image_quality_metrics(original_rgb, compressed_image)
                    prediction = predict_image(
                        compressed_image,
                        model=model,
                        class_names=class_names,
                        preprocess=preprocess,
                        device=device,
                        return_top5=False,
                    )

                    writer.writerow(
                        {
                            "dataset": dataset,
                            "path": image_path.as_posix(),
                            "class_id": class_id,
                            "target_index": target_index,
                            "target_label": target_label,
                            "codec": normalized_codec,
                            "quality": quality,
                            "predicted_index": prediction.predicted_index,
                            "predicted_class": prediction.predicted_class,
                            "top1_confidence": prediction.top1_confidence,
                            "is_correct": prediction.predicted_index == target_index,
                            "original_size_bytes": original_size,
                            "compressed_size_bytes": compressed_size,
                            "compression_ratio": original_size / compressed_size,
                            "psnr": psnr,
                            "ssim": ssim,
                        }
                    )


# Parse CLI arguments and run the selected compression classification jobs.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        action="append",
        help="Dataset to evaluate. Defaults to both.",
    )
    parser.add_argument(
        "--codec",
        choices=sorted(SUPPORTED_CODECS),
        action="append",
        help="Codec to evaluate. Defaults to both.",
    )
    parser.add_argument("--qualities", nargs="+", type=int, default=DEFAULT_QUALITIES)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output CSV path. Only valid when one dataset and one codec are selected.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    selected_datasets = args.dataset or sorted(DATASETS)
    selected_codecs = args.codec or sorted(SUPPORTED_CODECS)
    if args.output is not None and (len(selected_datasets) != 1 or len(selected_codecs) != 1):
        parser.error("--output requires exactly one --dataset and one --codec")
    qualities = validate_qualities(args.qualities)

    model, class_names = load_resnet50(device=args.device)
    preprocess = build_preprocess()

    for dataset in selected_datasets:
        for codec in selected_codecs:
            output_csv = (
                args.output
                if args.output is not None
                else args.results_dir / dataset / "compression" / f"{dataset}_{codec}.csv"
            )
            run_compression_experiment(
                dataset=dataset,
                codec=codec,
                qualities=qualities,
                data_dir=args.data_dir,
                output_csv=output_csv,
                model=model,
                class_names=class_names,
                preprocess=preprocess,
                device=args.device,
                limit=args.limit,
            )
            print(f"Wrote compressed predictions to {output_csv}")


if __name__ == "__main__":
    main()

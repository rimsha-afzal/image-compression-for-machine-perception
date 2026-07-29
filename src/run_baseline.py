"""Run baseline ResNet-50 classification on validation images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from load_datasets import DATASETS, image_files
from load_resnet50 import build_preprocess, load_resnet50, predict_image


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


# Run frozen ResNet-50 on validation images and write one CSV row per image.
def run_baseline(
    dataset: str,
    data_dir: Path,
    output_csv: Path,
    device: str | None,
    limit: int | None,
) -> None:
    model, class_names = load_resnet50(device=device)
    preprocess = build_preprocess()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "path",
                "class_id",
                "target_index",
                "target_label",
                "predicted_index",
                "predicted_class",
                "top1_confidence",
                "is_correct",
            ],
        )
        writer.writeheader()

        for count, (image_path, class_id, target_index, target_label) in enumerate(
            iter_validation_images(dataset, data_dir),
            start=1,
        ):
            if limit is not None and count > limit:
                break

            prediction = predict_image(
                image_path,
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
                    "predicted_index": prediction.predicted_index,
                    "predicted_class": prediction.predicted_class,
                    "top1_confidence": prediction.top1_confidence,
                    "is_correct": prediction.predicted_index == target_index,
                }
            )


# Parse CLI arguments and run the selected baseline classification job.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_csv = args.output or Path("results") / args.dataset / f"{args.dataset}_baseline.csv"
    run_baseline(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_csv=output_csv,
        device=args.device,
        limit=args.limit,
    )
    print(f"Wrote baseline predictions to {output_csv}")


if __name__ == "__main__":
    main()

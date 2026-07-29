"""Download and inspect Imagenette/ImageWoof validation sets."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image


DATASETS = {
    "imagenette": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        "root": "imagenette2-320",
        "classes": {
            "n01440764": (0, "tench"),
            "n02102040": (217, "English springer"),
            "n02979186": (482, "cassette player"),
            "n03000684": (491, "chain saw"),
            "n03028079": (497, "church"),
            "n03394916": (566, "French horn"),
            "n03417042": (569, "garbage truck"),
            "n03425413": (571, "gas pump"),
            "n03445777": (574, "golf ball"),
            "n03888257": (701, "parachute"),
        },
    },
    "imagewoof": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
        "root": "imagewoof2-320",
        "classes": {
            "n02086240": (155, "Shih-Tzu"),
            "n02087394": (159, "Rhodesian ridgeback"),
            "n02088364": (162, "beagle"),
            "n02089973": (167, "English foxhound"),
            "n02093754": (182, "Border terrier"),
            "n02096294": (193, "Australian terrier"),
            "n02099601": (207, "golden retriever"),
            "n02105641": (229, "Old English sheepdog"),
            "n02111889": (258, "Samoyed"),
            "n02115641": (273, "dingo"),
        },
    },
}


JPEG_EXTENSIONS = {".jpg", ".jpeg"}


# Download and extract a dataset archive, keeping only validation data by default.
def download_if_needed(name: str, data_dir: Path, keep_train: bool) -> Path:
    info = DATASETS[name]
    archive_path = data_dir / Path(info["url"]).name
    dataset_root = data_dir / info["root"]

    data_dir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"Downloading {name}: {info['url']}")
        urllib.request.urlretrieve(info["url"], archive_path)

    if not dataset_root.exists():
        print(f"Extracting {archive_path.name}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_dir)

    # remove train split if no model is being trained
    train_dir = dataset_root / "train"
    if train_dir.exists() and not keep_train:
        print(f"Removing unused training split: {train_dir}")
        shutil.rmtree(train_dir)

    return dataset_root


# Return all JPEG image files below a folder using a case-insensitive extension check.
def image_files(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )


# Inspect validation images and verify their folder labels against ResNet-50 ImageNet indices (index mapping control).
def inspect_dataset(name: str, dataset_root: Path) -> dict:
    val_dir = dataset_root / "val"
    expected_classes = DATASETS[name]["classes"]
    class_dirs = sorted(path for path in val_dir.iterdir() if path.is_dir())
    class_ids = [path.name for path in class_dirs]
    files = image_files(val_dir)

    formats: Counter[str] = Counter()
    dimensions: Counter[tuple[int, int]] = Counter()
    per_class_counts = {}

    for class_dir in class_dirs:
        class_files = image_files(class_dir)
        per_class_counts[class_dir.name] = len(class_files)
        for path in class_files:
            with Image.open(path) as image:
                formats[image.format or path.suffix.lower().lstrip(".")] += 1
                dimensions[image.size] += 1

    missing_mappings = sorted(set(class_ids) - set(expected_classes))
    unused_mappings = sorted(set(expected_classes) - set(class_ids))

    return {
        "name": name,
        "root": str(dataset_root),
        "folder_structure": {
            "root_entries": sorted(path.name for path in dataset_root.iterdir()),
            "validation_dir": str(val_dir),
            "validation_class_dirs": class_ids,
        },
        "number_of_classes": len(class_ids),
        "number_of_validation_images": len(files),
        "image_formats": dict(sorted(formats.items())),
        "image_dimensions": {
            f"{width}x{height}": count
            for (width, height), count in dimensions.most_common()
        },
        "class_labels": {
            class_id: {
                "label": expected_classes[class_id][1],
                "resnet50_imagenet_index": expected_classes[class_id][0],
                "validation_images": per_class_counts.get(class_id, 0),
            }
            for class_id in class_ids
            if class_id in expected_classes
        },
        "label_mapping_ok": not missing_mappings and not unused_mappings,
        "missing_resnet50_mappings": missing_mappings,
        "expected_mappings_not_found_in_val": unused_mappings,
    }


# Parse CLI arguments, download selected datasets, inspect them, and print JSON summaries.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        action="append",
        help="Dataset to download and inspect. Defaults to both.",
    )
    parser.add_argument(
        "--keep-train",
        action="store_true",
        help="Keep the extracted training split. By default only validation data is retained.",
    )
    args = parser.parse_args()

    selected = args.dataset or sorted(DATASETS)
    summaries = []
    for name in selected:
        root = download_if_needed(name, args.data_dir, args.keep_train)
        summaries.append(inspect_dataset(name, root))

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

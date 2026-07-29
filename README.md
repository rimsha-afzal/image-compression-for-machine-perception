# Image Compression for Machine Perception

This repository contains a fixed experiment pipeline for evaluating how lossy compression affects machine perception. It measures frozen ResNet-50 classification on Imagenette and ImageWoof validation images before and after compression.

## Fixed Experimental Scope

- Datasets: Imagenette and ImageWoof validation sets.
- Classifier: ResNet-50 with `IMAGENET1K_V2` ImageNet weights.
- Training: none; no fine-tuning is performed.
- Compression conditions: JPEG recompression and JPEG-to-WebP transcoding.
- Tested qualities: `95`, `80`, `60`, `40`, `20`, `10`.
- Metrics stored for compressed images: compressed size, compression ratio, PSNR, SSIM, and top-1 classification output.

The distributed dataset images are already JPEG encoded. JPEG conditions therefore represent additional JPEG recompression, while WebP conditions represent transcoding from decoded JPEG source images to WebP.

## Repository Structure

```text
src/
  compression.py
  load_datasets.py
  load_resnet50.py
  run_baseline.py
  run_compression.py
results/
  imagenette/
    imagenette_baseline.csv
    compression/
      imagenette_jpeg.csv
      imagenette_webp.csv
  imagewoof/
    imagewoof_baseline.csv
    compression/
      imagewoof_jpeg.csv
      imagewoof_webp.csv
```

The source image directory, `data/`, is not stored in Git.

## Environment Setup

Run commands from the repository root.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download and inspect both validation sets:

```bash
python src/load_datasets.py
```

Select one dataset:

```bash
python src/load_datasets.py --dataset imagenette
python src/load_datasets.py --dataset imagewoof
```

## Baseline Command

Run frozen ResNet-50 on the original validation images:

```bash
python src/run_baseline.py --dataset imagenette
python src/run_baseline.py --dataset imagewoof
```

Default outputs are:

- `results/imagenette/imagenette_baseline.csv`
- `results/imagewoof/imagewoof_baseline.csv`

Use `--output` to write a baseline CSV somewhere else.

## Compression Command

Run the compression experiment for both datasets and both codecs:

```bash
python src/run_compression.py
```

Select one dataset or codec:

```bash
python src/run_compression.py --dataset imagenette
python src/run_compression.py --codec webp
```

Default outputs are stored as `results/<dataset>/compression/<dataset>_<codec>.csv`. Use `--results-dir` to change the root output directory.

## Raw Output Files and Schemas

The six completed raw CSV result files are stored under `results/`. Baseline CSVs contain one row per validation image with dataset, path, target label, prediction, confidence, and correctness fields. Compression CSVs contain one row per validation image and quality, adding codec, quality, compressed size, compression ratio, PSNR, and SSIM fields.

See `results/README.md` for the exact retained files and columns.

## Important Experimental Framing

These files are raw per-image experiment outputs. No aggregate analysis, threshold estimate, plots, class-level analysis, confidence intervals, or final research report is currently included.

## Current Project Status

The repository currently preserves the completed raw baseline and compression CSV results plus the reproducible data-loading, baseline inference, compression, metric, and compressed-inference code. The next project stage will analyse the preserved raw results.

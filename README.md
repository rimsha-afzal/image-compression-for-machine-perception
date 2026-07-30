# Image Compression for Machine Perception

This repository contains a completed experiment pipeline and report for evaluating how lossy image compression affects machine perception. It measures frozen ResNet-50 classification on Imagenette and ImageWoof validation images before and after JPEG recompression and JPEG-to-WebP transcoding.

## Experimental Scope

- Datasets: Imagenette and ImageWoof validation sets.
- Classifier: ResNet-50 with `IMAGENET1K_V2` ImageNet weights.
- Training: none; no fine-tuning is performed.
- Exploratory compression qualities: `95`, `80`, `60`, `40`, `20`, `10`.
- Targeted refinement qualities: JPEG `70` and WebP `77`.
- Metrics: compressed size, compression ratio, size reduction, PSNR, SSIM, and top-1 classification accuracy.

The distributed dataset images are already JPEG encoded. JPEG conditions therefore represent additional JPEG recompression, while WebP conditions represent transcoding from decoded JPEG source images to WebP. PSNR and SSIM use the decoded distributed JPEG image as the reference.

## Repository Structure

```text
src/
  compression.py
  load_datasets.py
  load_resnet50.py
  run_baseline.py
  run_compression.py
  analysis/
    build_aggregate_table.py
    plot_exploratory_results.py
    render_aggregate_tables.py
    refinement/
      analyze_refinement.py
results/
  imagenette/
    imagenette_baseline.csv
    compression/
      imagenette_jpeg.csv
      imagenette_webp.csv
      refinement/
        imagenette_jpeg_q70.csv
        imagenette_webp_q77.csv
  imagewoof/
    imagewoof_baseline.csv
    compression/
      imagewoof_jpeg.csv
      imagewoof_webp.csv
      refinement/
        imagewoof_jpeg_q70.csv
        imagewoof_webp_q77.csv
  analysis/
    aggregate_results.csv
    plots/
    tables/
    refinement/
report/
  main.tex
  references.bib
  sections/
  main.pdf
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

The report documents the verified environment used for the final analysis. Package versions are not pinned in `requirements.txt`.

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

## Compression Commands

Run the exploratory compression experiment for both datasets and both codecs:

```bash
python src/run_compression.py
```

Select one dataset or codec:

```bash
python src/run_compression.py --dataset imagenette
python src/run_compression.py --codec webp
```

Run a targeted condition by selecting one dataset, one codec, an explicit quality, and an output path. For example:

```bash
python src/run_compression.py --dataset imagenette --codec jpeg --qualities 70 --output results/imagenette/compression/refinement/imagenette_jpeg_q70.csv
python src/run_compression.py --dataset imagewoof --codec webp --qualities 77 --output results/imagewoof/compression/refinement/imagewoof_webp_q77.csv
```

## Analysis Outputs

The completed analysis outputs are stored under `results/analysis/`:

- `aggregate_results.csv`: aggregate exploratory results by dataset, codec, and quality.
- `tables/`: rendered aggregate result tables for Imagenette and ImageWoof.
- `plots/rate_accuracy_tradeoff.png`: rate-accuracy exploratory figure.
- `plots/rate_distortion_tradeoff.png`: rate-distortion exploratory figure.
- `refinement/new_conditions.csv`: aggregate targeted refinement results.
- `refinement/threshold_refinement.png`: targeted threshold-refinement figure.
- `refinement/tables/`: rendered targeted refinement tables.

## Report

The final LaTeX report is in `report/`.

- Source: `report/main.tex` and `report/sections/`
- Bibliography: `report/references.bib`
- Compiled PDF: `report/main.pdf`

The report identifies practical tested operating regions rather than exact continuous optima. Using the primary 1 percentage-point accuracy-loss tolerance, JPEG q70 is the strongest tested JPEG condition passing on both datasets, while WebP q80 is the strongest tested WebP condition passing on both datasets. WebP q77 is treated as an informative near-boundary point and remains formally classified according to the strict rule.

## Current Project Status

The repository contains the completed raw experiment outputs, aggregate analysis, targeted refinement results, figures, tables, and final LaTeX report. No training, model fine-tuning, confidence-interval analysis, bootstrap analysis, or McNemar test is included.

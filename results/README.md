# Result Files

This directory contains the completed raw experiment outputs, aggregate analysis outputs, targeted refinement outputs, and rendered result figures/tables.

## Raw Baseline Files

- `imagenette/imagenette_baseline.csv`
- `imagewoof/imagewoof_baseline.csv`

Baseline files are produced with:

```bash
python src/run_baseline.py --dataset imagenette
python src/run_baseline.py --dataset imagewoof
```

Baseline CSV columns:

```text
dataset,path,class_id,target_index,target_label,predicted_index,predicted_class,top1_confidence,is_correct
```

## Raw Exploratory Compression Files

- `imagenette/compression/imagenette_jpeg.csv`
- `imagenette/compression/imagenette_webp.csv`
- `imagewoof/compression/imagewoof_jpeg.csv`
- `imagewoof/compression/imagewoof_webp.csv`

These files contain JPEG and WebP results for qualities `95`, `80`, `60`, `40`, `20`, and `10`. They are produced with:

```bash
python src/run_compression.py
```

Compression CSV columns:

```text
dataset,path,class_id,target_index,target_label,codec,quality,predicted_index,predicted_class,top1_confidence,is_correct,original_size_bytes,compressed_size_bytes,compression_ratio,psnr,ssim
```

## Raw Targeted Refinement Files

- `imagenette/compression/refinement/imagenette_jpeg_q70.csv`
- `imagenette/compression/refinement/imagenette_webp_q77.csv`
- `imagewoof/compression/refinement/imagewoof_jpeg_q70.csv`
- `imagewoof/compression/refinement/imagewoof_webp_q77.csv`

These files use the same schema as the exploratory compression CSVs. They evaluate JPEG q70 and WebP q77 around the predefined 1 percentage-point accuracy-loss tolerance.

## Aggregate and Visual Analysis Files

- `analysis/aggregate_results.csv`
- `analysis/tables/imagenette_results_table.png`
- `analysis/tables/imagewoof_results_table.png`
- `analysis/plots/rate_accuracy_tradeoff.png`
- `analysis/plots/rate_distortion_tradeoff.png`
- `analysis/refinement/new_conditions.csv`
- `analysis/refinement/threshold_refinement.png`
- `analysis/refinement/tables/imagenette_new_conditions.png`
- `analysis/refinement/tables/imagewoof_new_conditions.png`

`aggregate_results.csv` summarizes the exploratory raw compression files by dataset, codec, and quality. `analysis/refinement/new_conditions.csv` summarizes the targeted JPEG q70 and WebP q77 refinement files.

## Notes

Paths inside the CSVs are relative to the repository root. Compression ratios and size reductions are computed from total original and compressed byte counts. Mean PSNR and SSIM are arithmetic means over evaluated images, using the decoded distributed JPEG source as the reference.

The final interpretation of these results is provided in `report/main.pdf`.

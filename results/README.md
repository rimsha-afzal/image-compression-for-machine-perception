# Raw Result Files

This directory retains the six completed raw CSV files from the baseline and compression experiments:

- `imagenette/imagenette_baseline.csv`
- `imagenette/compression/imagenette_jpeg.csv`
- `imagenette/compression/imagenette_webp.csv`
- `imagewoof/imagewoof_baseline.csv`
- `imagewoof/compression/imagewoof_jpeg.csv`
- `imagewoof/compression/imagewoof_webp.csv`

Baseline files are produced with:

```bash
python src/run_baseline.py --dataset imagenette
python src/run_baseline.py --dataset imagewoof
```

Compression files are produced with:

```bash
python src/run_compression.py
```

Baseline CSV columns:

```text
dataset,path,class_id,target_index,target_label,predicted_index,predicted_class,top1_confidence,is_correct
```

Compression CSV columns:

```text
dataset,path,class_id,target_index,target_label,codec,quality,predicted_index,predicted_class,top1_confidence,is_correct,original_size_bytes,compressed_size_bytes,compression_ratio,psnr,ssim
```

Paths inside the CSVs are relative to the repository root. These CSVs are raw per-image experiment outputs, not aggregate conclusions. Future analysis must derive its results from these preserved files.

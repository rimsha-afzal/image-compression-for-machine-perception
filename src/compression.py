"""In-memory lossy image compression helpers."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


SUPPORTED_CODECS = {"jpeg": "JPEG", "webp": "WEBP"}


def compress_image(
    image: Image.Image,
    codec: str,
    quality: int,
) -> tuple[Image.Image, int]:
    """Compress and decode a PIL image in memory.

    Returns an independent RGB image reconstructed from the compressed bytes
    together with the encoded byte size.
    """
    normalized_codec = codec.lower()
    if normalized_codec not in SUPPORTED_CODECS:
        supported = ", ".join(sorted(SUPPORTED_CODECS))
        raise ValueError(f"Unsupported codec {codec!r}. Expected one of: {supported}")

    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")

    rgb_image = image.convert("RGB")
    buffer = BytesIO()
    rgb_image.save(
        buffer,
        format=SUPPORTED_CODECS[normalized_codec],
        quality=quality,
    )

    compressed_bytes = buffer.getvalue()
    compressed_size = len(compressed_bytes)

    with Image.open(BytesIO(compressed_bytes)) as decoded:
        decoded_image = decoded.convert("RGB").copy()

    return decoded_image, compressed_size


def image_quality_metrics(
    original_image: Image.Image,
    compressed_image: Image.Image,
) -> tuple[float, float]:
    """Return PSNR and SSIM between decoded RGB images."""
    original_array = np.asarray(original_image.convert("RGB"))
    compressed_array = np.asarray(compressed_image.convert("RGB"))

    psnr = peak_signal_noise_ratio(
        original_array,
        compressed_array,
        data_range=255,
    )
    ssim = structural_similarity(
        original_array,
        compressed_array,
        data_range=255,
        channel_axis=-1,
    )

    return float(psnr), float(ssim)

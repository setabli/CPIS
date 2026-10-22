import os

from tgrs_config import TGRSConfig

os.environ["CUDA_VISIBLE_DEVICES"] = TGRSConfig.GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import random
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from tgrs_model import TGRS


def setup():
    random.seed(TGRSConfig.SEED)
    np.random.seed(TGRSConfig.SEED)
    torch.manual_seed(TGRSConfig.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TGRSConfig.SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    if TGRSConfig.SAM_BATCH_SIZE < 1:
        raise ValueError("SAM_BATCH_SIZE must be positive.")

    for name in (
        "BOX_THRESHOLD",
        "TEXT_THRESHOLD",
        "NMS_THRESHOLD",
        "MASK_THRESHOLD",
    ):
        value = getattr(TGRSConfig, name)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "{} must be between 0 and 1.".format(name)
            )

    if min(TGRSConfig.LATENT_SIZE) < 1:
        raise ValueError("LATENT_SIZE must be positive.")

    use_cuda = (
        torch.cuda.is_available()
        and not TGRSConfig.FORCE_CPU
    )

    return torch.device(
        "cuda:0" if use_cuda else "cpu"
    )


def save_outputs(result, runtime_seconds):
    output_dir = TGRSConfig.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    image = result["image"]
    mask_hr = result["mask_hr"]
    mask_latent = result["mask_latent"]
    confidence = result["confidence"]

    binary_mask = mask_hr.astype(np.uint8)

    Image.fromarray(image).save(
        output_dir / "input.png"
    )

    Image.fromarray(binary_mask * 255).save(
        output_dir / "mask_hr.png"
    )

    Image.fromarray(
        (mask_latent[0, 0] * 255).astype(np.uint8)
    ).save(
        output_dir / "mask_latent.png"
    )

    Image.fromarray(
        np.rint(confidence * 255).astype(np.uint8)
    ).save(
        output_dir / "confidence.png"
    )

    np.save(
        output_dir / "mask_hr.npy",
        mask_hr,
    )

    np.save(
        output_dir / "mask_latent.npy",
        mask_latent,
    )

    np.save(
        output_dir / "confidence.npy",
        confidence,
    )

    inside = image * binary_mask[:, :, None]
    outside = image * (1 - binary_mask[:, :, None])

    Image.fromarray(inside).save(
        output_dir / "inside_mask.png"
    )

    Image.fromarray(outside).save(
        output_dir / "outside_mask.png"
    )

    overlay = image.astype(np.float32)
    editable = binary_mask.astype(bool)

    overlay[editable] = (
        0.6 * overlay[editable]
        + 0.4 * np.array(
            [255, 80, 50],
            dtype=np.float32,
        )
    )

    Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    ).save(
        output_dir / "mask_overlay.png"
    )

    detection_image = Image.fromarray(image)
    draw = ImageDraw.Draw(detection_image)

    detections = []

    for box, score, phrase, quality in zip(
        result["boxes"],
        result["scores"],
        result["phrases"],
        result["sam_quality_estimates"],
    ):
        coordinates = [
            float(value) for value in box
        ]

        draw.rectangle(
            coordinates,
            outline=(255, 0, 0),
            width=3,
        )

        draw.text(
            (
                coordinates[0],
                max(0, coordinates[1] - 14),
            ),
            "{} {:.3f}".format(phrase, float(score)),
            fill=(255, 0, 0),
        )

        detections.append(
            {
                "phrase": phrase,
                "detection_score": float(score),
                "box_xyxy": coordinates,
                "sam_quality_estimate": float(quality),
            }
        )

    detection_image.save(
        output_dir / "detections.png"
    )

    metadata = {
        "image_path": str(TGRSConfig.IMAGE_PATH),
        "target_text": TGRSConfig.TARGET_TEXT,
        "region_mode": TGRSConfig.REGION_MODE,
        "image_size_wh": [
            int(image.shape[1]),
            int(image.shape[0]),
        ],
        "latent_mask_shape": list(mask_latent.shape),
        "mask_threshold": TGRSConfig.MASK_THRESHOLD,
        "editable_fraction": float(mask_hr.mean()),
        "runtime_seconds": runtime_seconds,
        "detections": detections,
    }

    with (output_dir / "result.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
            ensure_ascii=False,
        )


def main():
    device = setup()

    if not TGRSConfig.IMAGE_PATH.is_file():
        raise FileNotFoundError(
            "Input image not found: {}".format(
                TGRSConfig.IMAGE_PATH
            )
        )

    with Image.open(TGRSConfig.IMAGE_PATH) as source:
        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

    if TGRSConfig.IMAGE_SIZE is not None:
        image = image.resize(
            TGRSConfig.IMAGE_SIZE,
            resample=Image.Resampling.LANCZOS,
        )

    print("Loading TGRS models...")
    selector = TGRS(device)
    print("Models loaded.")
    print("Device:", device)
    print("Target text:", TGRSConfig.TARGET_TEXT)
    print("Region mode:", TGRSConfig.REGION_MODE)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    started = time.perf_counter()

    result = selector.select(
        image=image,
        target_text=TGRSConfig.TARGET_TEXT,
        region_mode=TGRSConfig.REGION_MODE,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    runtime_seconds = time.perf_counter() - started

    save_outputs(result, runtime_seconds)

    print("Detected instances:", len(result["boxes"]))
    print(
        "Editable area: {:.2f}%".format(
            result["mask_hr"].mean() * 100
        )
    )
    print(
        "High-resolution mask shape:",
        result["mask_hr"].shape,
    )
    print(
        "Latent mask shape:",
        result["mask_latent"].shape,
    )
    print(
        "Inference runtime: {:.3f}s".format(
            runtime_seconds
        )
    )
    print("Output directory:", TGRSConfig.OUTPUT_DIR)


if __name__ == "__main__":
    main()
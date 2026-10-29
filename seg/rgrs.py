import os
import json
import math
import random
import time
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    QWEN_MODEL_PATH = BASE_DIR / "qwen2.5-vl"
    SAM_CHECKPOINT = BASE_DIR / "seg" / "sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"
    INSTRUCTION_PATH = BASE_DIR / "rgrs_instruction.md"

    IMAGE_PATH = BASE_DIR / "images" / "secret.png"
    RECEIVED_IMAGE_PATH = ""
    OUTPUT_DIR = BASE_DIR / "output" / "rgrs"

    GPU = "0"
    FORCE_CPU = False
    DTYPE = "auto"
    SEED = 0
    LOCAL_FILES_ONLY = True
    DETERMINISTIC = True

    IMAGE_SIZE = (512, 512)
    LATENT_SIZE = (64, 64)
    REFERENCE_POINT = (0.5, 0.5)
    MAX_CANDIDATES = 8
    MAX_NEW_TOKENS = 1024
    QWEN_MIN_PIXELS = 4 * 28 * 28
    QWEN_MAX_PIXELS = 512 * 512


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("Non-finite JSON value: {}".format(value))


def decode_json(text):
    decoder = json.JSONDecoder(
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
    clean = text.strip()
    try:
        return decoder.decode(clean), "strict"
    except (ValueError, json.JSONDecodeError):
        pass
    for offset, char in enumerate(clean):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(clean, offset)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "candidates" in value:
            return value, "extracted"
    raise ValueError("Qwen did not return a valid candidates JSON object. Inspect qwen_raw.txt.")


def validate_pair(value, name, integers=False):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("{} must contain two values.".format(name))
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("{} contains a non-numeric value.".format(name))
        if not math.isfinite(float(item)):
            raise ValueError("{} contains a non-finite value.".format(name))
        if integers and (not isinstance(item, int) or item <= 0):
            raise ValueError("{} must contain positive integers.".format(name))


def resize_for_qwen(image, min_pixels, max_pixels, factor=28):
    width, height = image.size
    if factor < 1 or min_pixels < factor * factor or max_pixels < min_pixels:
        raise ValueError("Invalid Qwen resizing parameters.")
    if max(width, height) / min(width, height) > 200:
        raise ValueError("The image aspect ratio is too large for Qwen2.5-VL.")
    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_width * resized_height > max_pixels:
        ratio = math.sqrt(width * height / max_pixels)
        resized_height = max(factor, math.floor(height / ratio / factor) * factor)
        resized_width = max(factor, math.floor(width / ratio / factor) * factor)
    elif resized_width * resized_height < min_pixels:
        ratio = math.sqrt(min_pixels / (width * height))
        resized_height = max(factor, math.ceil(height * ratio / factor) * factor)
        resized_width = max(factor, math.ceil(width * ratio / factor) * factor)
    if not min_pixels <= resized_width * resized_height <= max_pixels:
        raise ValueError("The configured pixel budget cannot accommodate this aspect ratio.")
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def validate_candidates(payload, qwen_size, image_size, max_candidates):
    if not isinstance(payload, dict) or set(payload) != {"candidates"}:
        raise ValueError("The JSON object must contain only the candidates key.")
    values = payload["candidates"]
    if not isinstance(values, list) or not 1 <= len(values) <= max_candidates:
        raise ValueError("The candidates list must contain between 1 and {} entries.".format(max_candidates))
    qwen_width, qwen_height = qwen_size
    image_width, image_height = image_size
    candidates = []
    seen = set()
    for index, item in enumerate(values):
        if not isinstance(item, dict) or set(item) != {"label", "bbox_2d"}:
            raise ValueError("Candidate {} must contain label and bbox_2d only.".format(index))
        label = item["label"]
        box = item["bbox_2d"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Candidate {} has an invalid label.".format(index))
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Candidate {} requires four XYXY coordinates.".format(index))
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError("Candidate {} contains invalid coordinates.".format(index))
        x1, y1, x2, y2 = [float(v) for v in box]
        if not (0 <= x1 < x2 <= qwen_width and 0 <= y1 < y2 <= qwen_height):
            raise ValueError("Candidate {} is outside the Qwen image or has reversed XYXY coordinates.".format(index))
        mapped = (
            x1 * image_width / qwen_width,
            y1 * image_height / qwen_height,
            x2 * image_width / qwen_width,
            y2 * image_height / qwen_height,
        )
        if mapped in seen:
            continue
        seen.add(mapped)
        candidates.append({
            "label": label.strip(),
            "bbox_qwen_xyxy": [x1, y1, x2, y2],
            "bbox_image_xyxy": list(mapped),
            "normalized_center": [
                (mapped[0] + mapped[2]) / (2 * image_width),
                (mapped[1] + mapped[3]) / (2 * image_height),
            ],
        })
    return candidates


def choose_candidate(candidates, reference_point):
    if not candidates:
        raise ValueError("No candidate is available.")
    validate_pair(reference_point, "REFERENCE_POINT")
    if any(v < 0 or v > 1 for v in reference_point):
        raise ValueError("REFERENCE_POINT must be normalized to [0, 1].")
    ranked = []
    for item in candidates:
        value = dict(item)
        cx, cy = value["normalized_center"]
        value["distance_squared"] = (cx - reference_point[0]) ** 2 + (cy - reference_point[1]) ** 2
        ranked.append(value)
    ranked.sort(key=lambda item: (
        item["distance_squared"],
        *item["bbox_image_xyxy"],
        item["label"],
    ))
    return ranked[0], ranked


def create_editable_masks(preserved_mask, latent_size):
    validate_pair(latent_size, "LATENT_SIZE", integers=True)
    preserved = np.asarray(preserved_mask, dtype=bool)
    if preserved.ndim != 2:
        raise ValueError("The SAM mask must be two-dimensional.")
    if not preserved.any():
        raise ValueError("SAM returned an empty preserved region.")
    if preserved.all():
        raise ValueError("SAM returned a full-image preserved region.")
    editable = np.logical_not(preserved).astype(np.uint8)
    latent = np.asarray(
        Image.fromarray(editable * 255).resize(tuple(latent_size), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return editable, (latent > 127).astype(np.uint8)


def mask_iou(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Mask shapes must match for IoU evaluation.")
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    return float(intersection / union) if union else 1.0


@dataclass
class RGRSResult:
    image: Image.Image
    preserved_mask: np.ndarray
    editable_mask: np.ndarray
    latent_mask: np.ndarray
    metadata: Dict[str, Any]


class RGRS:
    def __init__(self, config=Config):
        self.config = config
        self._validate_config()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU) if not config.FORCE_CPU else ""
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if config.LOCAL_FILES_ONLY:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as error:
            raise RuntimeError("Missing or incompatible dependencies. Install requirements.txt in Python 3.10 or 3.11.") from error

        self.torch = torch
        self.transformers_version = transformers.__version__
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(config.DETERMINISTIC)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(bool(config.DETERMINISTIC))
        if config.DETERMINISTIC:
            for name, value in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False), ("enable_math_sdp", True)):
                if hasattr(torch.backends.cuda, name):
                    getattr(torch.backends.cuda, name)(value)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not config.FORCE_CPU else "cpu")
        if self.device.type == "cpu":
            self.dtype = torch.float32
        elif config.DTYPE == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = getattr(torch, config.DTYPE)

        print("Loading local Qwen2.5-VL model...")
        self.processor = AutoProcessor.from_pretrained(
            str(config.QWEN_MODEL_PATH),
            local_files_only=config.LOCAL_FILES_ONLY,
            use_fast=False,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(config.QWEN_MODEL_PATH),
            local_files_only=config.LOCAL_FILES_ONLY,
            torch_dtype=self.dtype,
            device_map={"": str(self.device)},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model.eval().requires_grad_(False)
        vision = self.model.config.vision_config
        self.patch_size = int(vision.patch_size)
        self.resize_factor = self.patch_size * int(vision.spatial_merge_size)
        self.processor.image_processor.do_resize = False
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.do_sample = False
        self.generation_config.num_beams = 1
        self.generation_config.temperature = None
        self.generation_config.top_p = None
        self.generation_config.top_k = None
        self.generation_config.max_new_tokens = config.MAX_NEW_TOKENS
        self.generation_config.repetition_penalty = 1.0

        print("Loading local SAM checkpoint...")
        sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=str(config.SAM_CHECKPOINT))
        sam.to(device=self.device)
        sam.eval().requires_grad_(False)
        self.predictor = SamPredictor(sam)
        self.template = Path(config.INSTRUCTION_PATH).read_text(encoding="utf-8")
        required = ("{{image_width}}", "{{image_height}}", "{{max_candidates}}")
        if any(token not in self.template for token in required):
            raise ValueError("The instruction template is missing required placeholders.")
        print("Device: {}".format(self.device))
        print("Qwen dtype: {}".format(self.dtype))
        print("Reference point: {}".format(config.REFERENCE_POINT))
        print("Model loading completed.")

    def _validate_config(self):
        config = self.config
        if not (Path(config.QWEN_MODEL_PATH) / "config.json").is_file():
            raise FileNotFoundError("Set QWEN_MODEL_PATH to a complete local Qwen2.5-VL-Instruct model directory.")
        for path in (config.SAM_CHECKPOINT, config.INSTRUCTION_PATH):
            if not Path(path).is_file():
                raise FileNotFoundError("Required file not found: {}".format(path))
        if config.SAM_MODEL_TYPE not in ("vit_b", "vit_l", "vit_h"):
            raise ValueError("SAM_MODEL_TYPE must be vit_b, vit_l, or vit_h.")
        if config.DTYPE not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("DTYPE must be auto, float32, float16, or bfloat16.")
        if config.IMAGE_SIZE is not None:
            validate_pair(config.IMAGE_SIZE, "IMAGE_SIZE", integers=True)
        validate_pair(config.LATENT_SIZE, "LATENT_SIZE", integers=True)
        validate_pair(config.REFERENCE_POINT, "REFERENCE_POINT")
        if any(v < 0 or v > 1 for v in config.REFERENCE_POINT):
            raise ValueError("REFERENCE_POINT must be normalized to [0, 1].")
        if config.MAX_CANDIDATES < 1 or config.MAX_NEW_TOKENS < 1:
            raise ValueError("Candidate and generation limits must be positive.")

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _open_image(self, source):
        if isinstance(source, Image.Image):
            image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            with Image.open(source) as value:
                image = ImageOps.exif_transpose(value).convert("RGB")
        original_size = image.size
        if self.config.IMAGE_SIZE is not None and image.size != tuple(self.config.IMAGE_SIZE):
            image = image.resize(tuple(self.config.IMAGE_SIZE), Image.Resampling.LANCZOS)
        return image, original_size

    def select(self, image_source, output_dir=None):
        image, original_size = self._open_image(image_source)
        destination = Path(output_dir) if output_dir is not None else None
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
        qwen_image = resize_for_qwen(
            image,
            self.config.QWEN_MIN_PIXELS,
            self.config.QWEN_MAX_PIXELS,
            self.resize_factor,
        )
        instruction = self.template.replace("{{image_width}}", str(qwen_image.width))
        instruction = instruction.replace("{{image_height}}", str(qwen_image.height))
        instruction = instruction.replace("{{max_candidates}}", str(self.config.MAX_CANDIDATES))
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self._sync()
        started = time.perf_counter()
        inputs = self.processor(text=[text], images=[qwen_image], padding=True, return_tensors="pt")
        grid = inputs["image_grid_thw"][0]
        actual_size = (int(grid[2]) * self.patch_size, int(grid[1]) * self.patch_size)
        if actual_size != qwen_image.size:
            raise RuntimeError("Unexpected image resizing: Qwen coordinates would not align with the instruction.")
        inputs = inputs.to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=self.generation_config)
        continuation = generated[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        self._sync()
        qwen_seconds = time.perf_counter() - started
        if destination is not None:
            (destination / "qwen_raw.txt").write_text(raw + "\n", encoding="utf-8")
            (destination / "instruction.txt").write_text(instruction, encoding="utf-8")
            qwen_image.save(destination / "qwen_input.png")
        payload, parse_mode = decode_json(raw)
        candidates = validate_candidates(payload, qwen_image.size, image.size, self.config.MAX_CANDIDATES)
        selected, ranked = choose_candidate(candidates, self.config.REFERENCE_POINT)
        image_array = np.asarray(image, dtype=np.uint8)
        self._sync()
        started = time.perf_counter()
        with self.torch.inference_mode():
            self.predictor.set_image(image_array, image_format="RGB")
            masks, scores, _ = self.predictor.predict(
                box=np.asarray(selected["bbox_image_xyxy"], dtype=np.float32),
                multimask_output=False,
                return_logits=False,
            )
        self._sync()
        sam_seconds = time.perf_counter() - started
        preserved = np.asarray(masks[0], dtype=bool)
        editable, latent = create_editable_masks(preserved, self.config.LATENT_SIZE)
        metadata = {
            "image_source": str(image_source) if not isinstance(image_source, Image.Image) else "PIL.Image",
            "original_image_size_wh": list(original_size),
            "working_image_size_wh": list(image.size),
            "qwen_image_size_wh": list(qwen_image.size),
            "reference_point_xy": list(self.config.REFERENCE_POINT),
            "selection_rule": "minimum normalized bounding-box-center Euclidean distance",
            "tie_breaking": "lexicographic image-space XYXY coordinates, then label",
            "selected_candidate": selected,
            "ranked_candidates": ranked,
            "json_parse_mode": parse_mode,
            "sam_predicted_mask_quality": float(scores[0]),
            "preserved_fraction": float(preserved.mean()),
            "editable_fraction": float(editable.mean()),
            "mask_convention": "1=editable, 0=preserved",
            "latent_size_wh": list(self.config.LATENT_SIZE),
            "qwen_runtime_seconds": qwen_seconds,
            "sam_runtime_seconds": sam_seconds,
            "inference_runtime_seconds": qwen_seconds + sam_seconds,
            "qwen_model_path": str(self.config.QWEN_MODEL_PATH),
            "sam_checkpoint": str(self.config.SAM_CHECKPOINT),
            "sam_model_type": self.config.SAM_MODEL_TYPE,
            "transformers_version": self.transformers_version,
            "torch_version": str(self.torch.__version__),
            "device": str(self.device),
            "qwen_dtype": str(self.dtype),
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": self.config.MAX_NEW_TOKENS,
        }
        result = RGRSResult(image, preserved, editable, latent, metadata)
        if destination is not None:
            self.save(result, destination)
        return result

    def save(self, result, output_dir):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        result.image.save(destination / "image.png")
        Image.fromarray(result.preserved_mask.astype(np.uint8) * 255).save(destination / "preserved_mask.png")
        Image.fromarray(result.editable_mask * 255).save(destination / "mask.png")
        Image.fromarray(result.latent_mask * 255).save(destination / "mask_latent.png")
        np.save(destination / "mask_latent.npy", result.latent_mask)
        save_json(destination / "result.json", result.metadata)
        preview = result.image.copy()
        draw = ImageDraw.Draw(preview)
        selected = result.metadata["selected_candidate"]
        for item in result.metadata["ranked_candidates"]:
            color = (30, 210, 70) if item == selected else (255, 160, 0)
            draw.rectangle(tuple(item["bbox_image_xyxy"]), outline=color, width=3)
        px, py = self.config.REFERENCE_POINT
        x, y = px * (preview.width - 1), py * (preview.height - 1)
        radius = max(3, min(preview.size) // 100)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 30, 30))
        preview.save(destination / "boxes.png")
        overlay = np.asarray(result.image, dtype=np.float32).copy()
        region = result.preserved_mask
        overlay[region] = 0.6 * overlay[region] + 0.4 * np.asarray([30, 210, 70], dtype=np.float32)
        Image.fromarray(np.clip(np.rint(overlay), 0, 255).astype(np.uint8)).save(destination / "overlay.png")


def print_result(name, result):
    data = result.metadata
    print("\n{}".format(name))
    print("Selected label: {}".format(data["selected_candidate"]["label"]))
    print("Selected XYXY box: {}".format(data["selected_candidate"]["bbox_image_xyxy"]))
    print("Preserved fraction: {:.2%}".format(data["preserved_fraction"]))
    print("Editable fraction: {:.2%}".format(data["editable_fraction"]))
    print("Qwen runtime: {:.3f} s".format(data["qwen_runtime_seconds"]))
    print("SAM runtime: {:.3f} s".format(data["sam_runtime_seconds"]))
    print("JSON parse mode: {}".format(data["json_parse_mode"]))


def main():
    if not Path(Config.IMAGE_PATH).is_file():
        raise FileNotFoundError("Set Config.IMAGE_PATH to an existing input image.")
    if Config.RECEIVED_IMAGE_PATH and not Path(Config.RECEIVED_IMAGE_PATH).is_file():
        raise FileNotFoundError("Set RECEIVED_IMAGE_PATH to an existing image, or leave it empty.")
    selector = RGRS(Config)
    sender = selector.select(Config.IMAGE_PATH, Path(Config.OUTPUT_DIR) / "sender")
    print_result("Sender", sender)
    if Config.RECEIVED_IMAGE_PATH:
        receiver = selector.select(Config.RECEIVED_IMAGE_PATH, Path(Config.OUTPUT_DIR) / "receiver")
        print_result("Receiver", receiver)
        editable_iou = 100.0 * mask_iou(sender.editable_mask, receiver.editable_mask)
        preserved_iou = 100.0 * mask_iou(sender.preserved_mask, receiver.preserved_mask)
        comparison = {
            "editable_mask_iou_percent": editable_iou,
            "preserved_mask_iou_percent": preserved_iou,
        }
        save_json(Path(Config.OUTPUT_DIR) / "comparison.json", comparison)
        print("\nSender-receiver editable-mask IoU: {:.4f}%".format(editable_iou))
        print("Sender-receiver preserved-mask IoU: {:.4f}%".format(preserved_iou))
    print("\nOutput directory: {}".format(Path(Config.OUTPUT_DIR).resolve()))


if __name__ == "__main__":
    main()

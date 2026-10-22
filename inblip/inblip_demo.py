import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    MODEL_PATH = BASE_DIR
    TEMPLATE_PATH = BASE_DIR / "inblip_prompts.json"
    IMAGE_PATH = PROJECT_DIR / "images" / "input.png"
    MASK_PATH = PROJECT_DIR / "images" / "mask.png"
    OUTPUT_DIR = PROJECT_DIR / "inblip_output"
    GPU = "3"
    FORCE_CPU = False
    DTYPE = "float16"
    SEED = 0
    MASK_THRESHOLD = 127
    OBJECT_REGION = "black"
    USE_OBJECT_CROP = True
    CROP_MARGIN_RATIO = 0.10
    CROP_MIN_PADDING = 4
    INCLUDE_FULL_IMAGE = True


os.environ["CUDA_VISIBLE_DEVICES"] = Config.GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import re

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


GENERATION_ARGS = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "repetition_penalty": 1.0,
    "use_cache": True,
}


def setup():
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for name, value in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
    ):
        if hasattr(torch.backends.cuda, name):
            getattr(torch.backends.cuda, name)(value)
    torch.set_num_threads(1)
    if Config.OBJECT_REGION not in ("black", "white"):
        raise ValueError("OBJECT_REGION must be 'black' or 'white'.")
    if not 0 <= Config.MASK_THRESHOLD <= 255:
        raise ValueError("MASK_THRESHOLD must be between 0 and 255.")
    if Config.CROP_MARGIN_RATIO < 0 or Config.CROP_MIN_PADDING < 0:
        raise ValueError("Crop margins must be non-negative.")
    if Config.DTYPE not in ("float16", "float32", "bfloat16"):
        raise ValueError("Unsupported DTYPE.")
    for path in (Config.IMAGE_PATH, Config.MASK_PATH, Config.TEMPLATE_PATH):
        if not path.is_file():
            raise FileNotFoundError("Input file not found: {}".format(path))
    if not Config.MODEL_PATH.is_dir():
        raise FileNotFoundError("Local model directory not found: {}".format(Config.MODEL_PATH))
    use_cuda = torch.cuda.is_available() and not Config.FORCE_CPU
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = getattr(torch, Config.DTYPE) if use_cuda else torch.float32
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return device, dtype


def load_templates():
    with Config.TEMPLATE_PATH.open("r", encoding="utf-8") as stream:
        templates = json.load(stream)
    for stream_name in ("object", "context"):
        questions = templates.get(stream_name)
        if not isinstance(questions, list) or not questions:
            raise ValueError("Each view must have a non-empty question list.")
        names = [question["field"] for question in questions]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate question fields in {}.".format(stream_name))
    entries = templates["object"] + templates["context"]
    if Config.INCLUDE_FULL_IMAGE:
        entries = entries + [templates["full_image"]]
    for question in entries:
        if not isinstance(question["question"], str) or not question["question"].strip():
            raise ValueError("Question text must be non-empty.")
        if not isinstance(question["max_new_tokens"], int) or question["max_new_tokens"] < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
    return templates


def prepare_views():
    with Image.open(Config.IMAGE_PATH) as source:
        image = source.convert("RGB")
    with Image.open(Config.MASK_PATH) as source:
        mask = source.convert("L")
    if mask.size != image.size:
        raise ValueError("Image and mask sizes differ. Align them before inference.")
    image_array = np.asarray(image, dtype=np.uint8)
    black_region = np.asarray(mask, dtype=np.uint8) <= Config.MASK_THRESHOLD
    object_region = black_region if Config.OBJECT_REGION == "black" else ~black_region
    context_region = ~object_region
    if not object_region.any() or not context_region.any():
        raise ValueError("The mask must contain both object and context regions.")
    object_array = np.where(object_region[:, :, None], image_array, 0).astype(np.uint8)
    context_array = np.where(context_region[:, :, None], image_array, 0).astype(np.uint8)
    ys, xs = np.where(object_region)
    height, width = object_region.shape
    padding = max(
        Config.CROP_MIN_PADDING,
        int(round(Config.CROP_MARGIN_RATIO * max(
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        ))),
    )
    box = (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + padding + 1),
        min(height, int(ys.max()) + padding + 1),
    )
    object_full = Image.fromarray(object_array)
    object_crop = object_full.crop(box)
    context_full = Image.fromarray(context_array)
    object_full.save(Config.OUTPUT_DIR / "object_full.png")
    object_crop.save(Config.OUTPUT_DIR / "object_crop.png")
    context_full.save(Config.OUTPUT_DIR / "context_full.png")
    Image.fromarray(object_region.astype(np.uint8) * 255).save(Config.OUTPUT_DIR / "object_mask.png")
    metadata = {
        "image_size_wh": list(image.size),
        "object_region": Config.OBJECT_REGION,
        "object_fraction": float(object_region.mean()),
        "context_fraction": float(context_region.mean()),
        "object_crop_box_xyxy": list(box),
        "mask_threshold": Config.MASK_THRESHOLD,
        "object_input": "object_crop" if Config.USE_OBJECT_CROP else "object_full",
        "context_input": "context_full",
    }
    return {
        "full_image": image,
        "object": object_crop if Config.USE_OBJECT_CROP else object_full,
        "context": context_full,
    }, metadata


def load_model(device, dtype):
    processor = InstructBlipProcessor.from_pretrained(
        str(Config.MODEL_PATH), local_files_only=True,
    )
    model = InstructBlipForConditionalGeneration.from_pretrained(
        str(Config.MODEL_PATH), torch_dtype=dtype, local_files_only=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    for component in (model, getattr(model, "language_model", None)):
        generation_config = getattr(component, "generation_config", None)
        if generation_config is not None:
            generation_config.do_sample = False
            generation_config.num_beams = 1
            generation_config.num_return_sequences = 1
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50
            generation_config.repetition_penalty = 1.0
            generation_config.min_length = 0
            generation_config.min_new_tokens = None
    print("Model loaded:", Config.MODEL_PATH)
    print("Device:", device)
    print("Dtype:", dtype)
    print("Transformers version:", transformers.__version__)
    print("Text model type:", model.config.text_config.model_type)
    return processor, model


@torch.inference_mode()
def answer_question(image, question, processor, model, device, dtype):
    inputs = processor(images=image, text=question["question"], return_tensors="pt")
    prepared = {
        name: value.to(device=device, dtype=dtype) if value.is_floating_point()
        else value.to(device=device)
        for name, value in inputs.items()
    }
    generated = model.generate(
        **prepared, **GENERATION_ARGS, max_new_tokens=question["max_new_tokens"],
    )
    raw_text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return {
        "question": question["question"],
        "max_new_tokens": question["max_new_tokens"],
        "raw_text": raw_text,
        "text": " ".join(raw_text.split()).strip(),
    }


def parse_count(text):
    value = text.lower().strip().rstrip(".!?")
    if re.fullmatch(r"\d+", value):
        return int(value)
    return {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }.get(value)


def run(views, templates, processor, model, device, dtype):
    raw_answers = {}
    structured = {}
    description = []
    if Config.INCLUDE_FULL_IMAGE:
        answer = answer_question(views["full_image"], templates["full_image"], processor, model, device, dtype)
        raw_answers["full_image"] = answer
        structured["full_image_caption"] = answer["text"]
        description.append("Full image: " + answer["text"])
        print("Full image:", answer["text"])
    for stream_name in ("object", "context"):
        raw_answers[stream_name] = {}
        structured[stream_name] = {}
        for question in templates[stream_name]:
            field = question["field"]
            answer = answer_question(views[stream_name], question, processor, model, device, dtype)
            raw_answers[stream_name][field] = answer
            structured[stream_name][field] = parse_count(answer["text"]) if field == "count" else answer["text"]
            description.append("{}/{}: {}".format(stream_name, field, answer["text"]))
            print("{}/{}: {}".format(stream_name, field, answer["text"]))
    return {
        "raw_answers": raw_answers,
        "structured": structured,
        "description_text": "\n".join(description),
    }


def main():
    device, dtype = setup()
    templates = load_templates()
    views, mask_metadata = prepare_views()
    print("Object region:", mask_metadata["object_region"])
    print("Object input:", mask_metadata["object_input"])
    print("Object area: {:.2f}%".format(mask_metadata["object_fraction"] * 100))
    processor, model = load_model(device, dtype)
    result = run(views, templates, processor, model, device, dtype)
    result["configuration"] = {
        "model_path": str(Config.MODEL_PATH),
        "image_path": str(Config.IMAGE_PATH),
        "mask_path": str(Config.MASK_PATH),
        "template_path": str(Config.TEMPLATE_PATH),
        "seed": Config.SEED,
        "dtype": str(dtype),
        "generation_args": GENERATION_ARGS,
    }
    result["mask_metadata"] = mask_metadata
    result_path = Config.OUTPUT_DIR / "result.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    text_path = Config.OUTPUT_DIR / "description.txt"
    text_path.write_text(result["description_text"] + "\n", encoding="utf-8")
    print("\nStructured descriptions:")
    print(json.dumps(result["structured"], ensure_ascii=False, indent=2))
    print("Result saved:", result_path)
    print("Description saved:", text_path)


if __name__ == "__main__":
    main()

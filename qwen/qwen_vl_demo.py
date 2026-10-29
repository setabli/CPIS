import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    MODEL_PATH = BASE_DIR
    TEMPLATE_PATH = BASE_DIR / "qwen_vl_prompts.json"
    IMAGE_PATH = PROJECT_DIR / "images" / "input.png"
    MASK_PATH = PROJECT_DIR / "images" / "mask.png"
    OUTPUT_DIR = PROJECT_DIR / "qwen_vl_output"

    GPU = "3"
    FORCE_CPU = False
    DTYPE = "auto"
    SEED = 0
    LOCAL_FILES_ONLY = True
    MIN_PIXELS = 4 * 28 * 28
    MAX_PIXELS = 512 * 512

    MASK_THRESHOLD = 127
    OBJECT_REGION = "white"
    USE_OBJECT_CROP = False
    CROP_MARGIN_RATIO = 0.10
    CROP_MIN_PADDING = 4
    INCLUDE_FULL_IMAGE = False


os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.GPU) if not Config.FORCE_CPU else ""
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import json
import random
import time

import numpy as np
from PIL import Image


GENERATION_ARGS = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "repetition_penalty": 1.0,
    "use_cache": True,
}


def reject_duplicate_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate JSON key: {}".format(key))
        output[key] = value
    return output


def reject_nonfinite(value):
    raise ValueError("Non-finite JSON constant: {}".format(value))


def make_decoder():
    return json.JSONDecoder(object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def setup():
    if Config.LOCAL_FILES_ONLY:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required. Use Python 3.10 or 3.11 with compatible dependencies.") from error
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
    for name, value in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False), ("enable_math_sdp", True)):
        if hasattr(torch.backends.cuda, name):
            getattr(torch.backends.cuda, name)(value)
    torch.set_num_threads(1)
    if Config.OBJECT_REGION not in ("black", "white"):
        raise ValueError("OBJECT_REGION must be black or white.")
    if not 0 <= Config.MASK_THRESHOLD <= 255:
        raise ValueError("MASK_THRESHOLD must be between 0 and 255.")
    if Config.CROP_MARGIN_RATIO < 0 or Config.CROP_MIN_PADDING < 0:
        raise ValueError("Crop margins must be non-negative.")
    if Config.DTYPE not in ("auto", "float16", "float32", "bfloat16"):
        raise ValueError("Unsupported DTYPE.")
    if Config.MIN_PIXELS < 4 * 28 * 28 or Config.MAX_PIXELS < Config.MIN_PIXELS:
        raise ValueError("Invalid image pixel limits.")
    for value in (Config.IMAGE_PATH, Config.MASK_PATH, Config.TEMPLATE_PATH):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError("Input file not found: {}".format(path))
    if not (Path(Config.MODEL_PATH) / "config.json").is_file():
        raise FileNotFoundError("MODEL_PATH must contain the complete local Qwen2.5-VL-Instruct model.")
    use_cuda = torch.cuda.is_available() and not Config.FORCE_CPU
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if not use_cuda:
        dtype = torch.float32
    elif Config.DTYPE == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = getattr(torch, Config.DTYPE)
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return device, dtype


def load_templates():
    templates = make_decoder().decode(Path(Config.TEMPLATE_PATH).read_text(encoding="utf-8-sig"))
    if not isinstance(templates, dict):
        raise ValueError("The template file must contain a JSON object.")
    system = templates.get("system_prompt")
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system_prompt must be non-empty.")
    for stream_name in ("object", "context"):
        entry = templates.get(stream_name)
        if not isinstance(entry, dict):
            raise ValueError("Each stream requires a single query object.")
        fields = entry.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise ValueError("Each stream must define its output fields.")
        for name, field_type in fields.items():
            if not isinstance(name, str) or not name or field_type not in ("string", "integer_or_null"):
                raise ValueError("Invalid field specification in {}.".format(stream_name))
    names = ["object", "context"]
    if Config.INCLUDE_FULL_IMAGE:
        names.append("full_image")
    for name in names:
        entry = templates.get(name)
        if not isinstance(entry, dict):
            raise ValueError("Missing template: {}".format(name))
        if not isinstance(entry.get("question"), str) or not entry["question"].strip():
            raise ValueError("Question text must be non-empty.")
        limit = entry.get("max_new_tokens")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
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
    object_region = ~black_region if Config.OBJECT_REGION == "white" else black_region
    context_region = ~object_region
    if not object_region.any() or not context_region.any():
        raise ValueError("The mask must contain both local and context regions.")
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
    destination = Path(Config.OUTPUT_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    object_full.save(destination / "object_full.png")
    object_crop.save(destination / "object_crop.png")
    context_full.save(destination / "context_full.png")
    Image.fromarray(object_region.astype(np.uint8) * 255).save(destination / "object_mask.png")
    metadata = {
        "image_size_wh": list(image.size),
        "object_region": Config.OBJECT_REGION,
        "object_fraction": float(object_region.mean()),
        "context_fraction": float(context_region.mean()),
        "object_crop_box_xyxy": list(box),
        "mask_threshold": Config.MASK_THRESHOLD,
        "object_input": "object_crop" if Config.USE_OBJECT_CROP else "object_full",
        "context_input": "context_full",
        "object_role": "local mask-region evidence (J_in)",
        "context_role": "complementary visible-context evidence (J_out)",
        "object_key_is_not_an_object_class_constraint": True,
    }
    return {
        "full_image": image,
        "object": object_crop if Config.USE_OBJECT_CROP else object_full,
        "context": context_full,
    }, metadata


def load_model(device, dtype):
    try:
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import transformers
    except ImportError as error:
        raise RuntimeError("Qwen2.5-VL support is required. Install transformers==4.51.3 and accelerate.") from error
    model_config = AutoConfig.from_pretrained(str(Config.MODEL_PATH), local_files_only=Config.LOCAL_FILES_ONLY)
    if getattr(model_config, "model_type", None) != "qwen2_5_vl":
        raise ValueError("MODEL_PATH must point to Qwen2.5-VL, not a text-only Qwen model.")
    processor = AutoProcessor.from_pretrained(
        str(Config.MODEL_PATH),
        local_files_only=Config.LOCAL_FILES_ONLY,
        min_pixels=Config.MIN_PIXELS,
        max_pixels=Config.MAX_PIXELS,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(Config.MODEL_PATH),
        config=model_config,
        torch_dtype=dtype,
        local_files_only=Config.LOCAL_FILES_ONLY,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval().requires_grad_(False)
    generation = model.generation_config
    for name, value in GENERATION_ARGS.items():
        setattr(generation, name, value)
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    generation.min_length = 0
    generation.min_new_tokens = None
    print("Model loaded:", Config.MODEL_PATH)
    print("Device:", device)
    print("Dtype:", dtype)
    print("Transformers version:", transformers.__version__)
    print("Model type:", model.config.model_type)
    return processor, model


def answer_question(image, question, system_prompt, processor, model, device, dtype):
    import torch
    query = question["question"]
    if "fields" in question:
        query += "\nRequired fields and value types:\n" + json.dumps(question["fields"], ensure_ascii=False, indent=2)
        query += "\nReturn exactly those fields with their observed values. Do not return the field-type specification itself."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = question["max_new_tokens"]
    with torch.inference_mode():
        generated = model.generate(**inputs, generation_config=generation_config)
    generated_ids = generated[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "question": question["question"],
        "max_new_tokens": question["max_new_tokens"],
        "raw_text": raw_text,
        "text": " ".join(raw_text.split()).strip(),
        "runtime_seconds": time.perf_counter() - started,
    }


def parse_description(raw_text, fields):
    decoder = make_decoder()
    candidates = []
    try:
        candidates.append((decoder.decode(raw_text.strip()), "strict"))
    except ValueError:
        pass
    if not candidates:
        for offset, char in enumerate(raw_text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw_text, offset)
            except ValueError:
                continue
            if isinstance(value, dict) and set(value) == set(fields):
                candidates.append((value, "extracted"))
                break
    if not candidates:
        raise ValueError("No valid description JSON object was found.")
    parsed, mode = candidates[0]
    if not isinstance(parsed, dict) or set(parsed) != set(fields):
        raise ValueError("The response fields do not match the template.")
    normalized = {}
    for field, field_type in fields.items():
        value = parsed[field]
        if field_type == "string":
            if not isinstance(value, str):
                raise ValueError("Field {} must be a string.".format(field))
            normalized[field] = " ".join(value.split()).strip()
        elif field_type == "integer_or_null":
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError("Field {} must be a non-negative integer or null.".format(field))
            normalized[field] = value
        else:
            raise ValueError("Unsupported field type: {}".format(field_type))
    return normalized, mode


def run(views, templates, processor, model, device, dtype):
    raw_answers = {}
    structured = {}
    description = []
    destination = Path(Config.OUTPUT_DIR)
    if Config.INCLUDE_FULL_IMAGE:
        answer = answer_question(
            views["full_image"], templates["full_image"], templates["system_prompt"], processor, model, device, dtype,
        )
        raw_answers["full_image"] = answer
        structured["full_image_caption"] = answer["text"]
        description.append("Full image: " + answer["text"])
        print("Full image:", answer["text"])
        save_json(destination / "full_image_raw.json", answer)
    for stream_name in ("object", "context"):
        question = templates[stream_name]
        answer = answer_question(views[stream_name], question, templates["system_prompt"], processor, model, device, dtype)
        (destination / (stream_name + "_raw.txt")).write_text(answer["raw_text"] + "\n", encoding="utf-8")
        save_json(destination / (stream_name + "_raw.json"), answer)
        try:
            parsed, parse_mode = parse_description(answer["raw_text"], question["fields"])
        except ValueError as error:
            raise ValueError("Invalid {} response. Inspect {}_raw.txt. {}".format(stream_name, stream_name, error)) from error
        answer["parse_mode"] = parse_mode
        raw_answers[stream_name] = answer
        structured[stream_name] = parsed
        print("\n{} description:".format(stream_name.capitalize()))
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print("Runtime: {:.3f} s".format(answer["runtime_seconds"]))
        for field, value in parsed.items():
            description.append("{}/{}: {}".format(stream_name, field, value if value is not None else "unknown"))
    return {
        "raw_answers": raw_answers,
        "structured": structured,
        "description_text": "\n".join(description),
    }


def main():
    device, dtype = setup()
    templates = load_templates()
    views, mask_metadata = prepare_views()
    print("Local mask region:", mask_metadata["object_region"])
    print("Local input:", mask_metadata["object_input"])
    print("Local area: {:.2f}%".format(mask_metadata["object_fraction"] * 100))
    processor, model = load_model(device, dtype)
    result = run(views, templates, processor, model, device, dtype)
    result["configuration"] = {
        "model_path": str(Config.MODEL_PATH),
        "model_type": "qwen2_5_vl",
        "image_path": str(Config.IMAGE_PATH),
        "mask_path": str(Config.MASK_PATH),
        "template_path": str(Config.TEMPLATE_PATH),
        "template_version": templates.get("version", ""),
        "seed": Config.SEED,
        "dtype": str(dtype),
        "generation_args": GENERATION_ARGS,
        "temperature": None,
        "min_pixels": Config.MIN_PIXELS,
        "max_pixels": Config.MAX_PIXELS,
        "include_full_image": Config.INCLUDE_FULL_IMAGE,
        "use_object_crop": Config.USE_OBJECT_CROP,
    }
    result["mask_metadata"] = mask_metadata
    destination = Path(Config.OUTPUT_DIR)
    result_path = destination / "result.json"
    save_json(result_path, result)
    text_path = destination / "description.txt"
    text_path.write_text(result["description_text"] + "\n", encoding="utf-8")
    print("\nResult saved:", result_path)
    print("Description saved:", text_path)


if __name__ == "__main__":
    main()

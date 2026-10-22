import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    MODEL_PATH = BASE_DIR
    TEMPLATE_PATH = BASE_DIR / "semantic_aggregation.md"
    INPUT_PATH = PROJECT_DIR / "inblip_output" / "result.json"
    OUTPUT_DIR = PROJECT_DIR / "qwen_output"
    GPU = "3"
    FORCE_CPU = False
    DTYPE = "auto"
    SEED = 0
    MAX_INPUT_TOKENS = 8192
    MAX_NEW_TOKENS = 768
    TRUST_REMOTE_CODE = False
    REQUIRE_SUBJECT = True
    LEGACY_OBJECT_REGION = "black"


os.environ["CUDA_VISIBLE_DEVICES"] = Config.GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import json
import random
from datetime import datetime

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


FIELDS = ("subject", "action", "location", "relation")
ROOT_FIELDS = set(FIELDS) | {"environment"}
UNKNOWN_VALUES = {
    "unknown", "unspecified", "uncertain", "not visible", "not specified",
    "n/a", "none", "null",
}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def reject_nonfinite_json(value):
    raise ValueError("Non-finite JSON constant is not allowed: {}".format(value))


def strict_json_loads(text):
    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite_json,
    )


def save_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


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
    for path in (Config.INPUT_PATH, Config.TEMPLATE_PATH, Config.MODEL_PATH / "config.json"):
        if not path.is_file():
            raise FileNotFoundError("Required file not found: {}".format(path))
    if Config.MAX_INPUT_TOKENS < 1 or Config.MAX_NEW_TOKENS < 1:
        raise ValueError("Token limits must be positive.")
    if Config.LEGACY_OBJECT_REGION not in ("black", "white"):
        raise ValueError("LEGACY_OBJECT_REGION must be 'black' or 'white'.")
    use_cuda = torch.cuda.is_available() and not Config.FORCE_CPU
    return torch.device("cuda:0" if use_cuda else "cpu")


def load_evidence():
    payload = strict_json_loads(Config.INPUT_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("The InstructBLIP result must be a JSON object.")
    structured = payload.get("structured", payload)
    if not isinstance(structured, dict):
        raise TypeError("The structured descriptions must be a JSON object.")
    metadata = payload.get("mask_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if "object" in structured and "context" in structured:
        object_evidence = structured["object"]
        context_evidence = structured["context"]
        object_region = metadata.get("object_region", "")
    elif "black_region" in structured and "white_region" in structured:
        object_region = Config.LEGACY_OBJECT_REGION
        context_region = "white" if object_region == "black" else "black"
        object_evidence = structured[object_region + "_region"]
        context_evidence = structured[context_region + "_region"]
    else:
        raise ValueError("Input must contain object/context or black_region/white_region descriptions.")
    if not isinstance(object_evidence, dict) or not isinstance(context_evidence, dict):
        raise TypeError("Object and context descriptions must be JSON objects.")
    caption = structured.get("full_image_caption", "")
    if caption is None:
        caption = ""
    if not isinstance(caption, str):
        raise TypeError("full_image_caption must be a string.")
    for stream in (object_evidence, context_evidence):
        for name, value in stream.items():
            if value is not None and not isinstance(value, (str, int, float)):
                raise TypeError("Unsupported evidence value in field {}.".format(name))
    return {
        "full_image_caption": caption,
        "object": object_evidence,
        "context": context_evidence,
        "source_roles": {
            "object_region_mask_color": object_region,
            "object_view": metadata.get("object_input", "object view"),
            "context_view": metadata.get("context_input", "complementary context view"),
        },
    }


def choose_dtype(device, model_config):
    if device.type == "cpu":
        return torch.float32
    if Config.DTYPE == "auto":
        value = getattr(model_config, "torch_dtype", None)
        if isinstance(value, str):
            value = getattr(torch, value.replace("torch.", ""), None)
        if value not in (torch.float16, torch.bfloat16, torch.float32):
            value = torch.float16
        if value == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            value = torch.float32
        return value
    if Config.DTYPE not in ("float16", "bfloat16", "float32"):
        raise ValueError("DTYPE must be auto, float16, bfloat16, or float32.")
    value = getattr(torch, Config.DTYPE)
    if value == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16. Set DTYPE to float16 or float32.")
    return value


def load_model(device):
    model_config = AutoConfig.from_pretrained(
        str(Config.MODEL_PATH), local_files_only=True,
        trust_remote_code=Config.TRUST_REMOTE_CODE,
    )
    dtype = choose_dtype(device, model_config)
    tokenizer = AutoTokenizer.from_pretrained(
        str(Config.MODEL_PATH), local_files_only=True,
        trust_remote_code=Config.TRUST_REMOTE_CODE,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The local tokenizer must provide an instruction-model chat template.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("The tokenizer must define a padding or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(Config.MODEL_PATH), config=model_config,
        torch_dtype=dtype, local_files_only=True,
        trust_remote_code=Config.TRUST_REMOTE_CODE,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    print("Model loaded:", Config.MODEL_PATH)
    print("Model type:", model_config.model_type)
    print("Device:", device)
    print("Dtype:", dtype)
    print("Transformers version:", transformers.__version__)
    return tokenizer, model, dtype


def build_messages(template, evidence):
    return [
        {"role": "system", "content": template},
        {
            "role": "user",
            "content": "Normalize the following visual evidence according to the system specification.\n"
                       "<visual_evidence>\n"
                       + json.dumps(evidence, ensure_ascii=False, indent=2)
                       + "\n</visual_evidence>",
        },
    ]


@torch.inference_mode()
def generate_structure(tokenizer, model, messages, device):
    template_text = str(tokenizer.chat_template)
    chat_kwargs = {"enable_thinking": False} if "enable_thinking" in template_text else {}
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False, truncation=False,
    )
    input_tokens = inputs["input_ids"].shape[1]
    if input_tokens > Config.MAX_INPUT_TOKENS:
        raise ValueError("Input has {} tokens and exceeds MAX_INPUT_TOKENS. No input was truncated.".format(input_tokens))
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and input_tokens + Config.MAX_NEW_TOKENS > context_limit:
        raise ValueError("The requested input/output token budget exceeds the model context limit.")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.num_beams = 1
    generation.num_return_sequences = 1
    generation.temperature = 1.0
    generation.top_p = 1.0
    generation.top_k = 50
    generation.repetition_penalty = 1.0
    generation.min_length = 0
    generation.min_new_tokens = None
    generation.max_new_tokens = Config.MAX_NEW_TOKENS
    generation.pad_token_id = tokenizer.pad_token_id
    generation.use_cache = True
    generated = model.generate(**inputs, generation_config=generation)
    new_tokens = generated[0, input_tokens:]
    raw_text = tokenizer.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return raw_text, {
        "input_tokens": int(input_tokens),
        "generated_tokens": int(new_tokens.numel()),
        "token_limit_reached": bool(new_tokens.numel() >= Config.MAX_NEW_TOKENS),
        "thinking_disabled_via_template": bool(chat_kwargs),
    }


def normalize_field(value, name):
    if not isinstance(value, str):
        raise TypeError("Semantic field {} must be a string.".format(name))
    normalized = " ".join(value.lower().split())
    if normalized in UNKNOWN_VALUES:
        raise ValueError("Use an empty string for unknown semantic field {}.".format(name))
    return normalized


def normalize_semantics(value):
    if not isinstance(value, dict) or set(value) != ROOT_FIELDS:
        raise ValueError("Output must contain exactly subject, action, location, relation, environment.")
    normalized = {name: normalize_field(value[name], name) for name in FIELDS}
    if Config.REQUIRE_SUBJECT and not normalized["subject"]:
        raise ValueError("No supported main subject was recovered. Inspect the visual evidence.")
    if normalized["relation"] and not normalized["location"]:
        raise ValueError("A main relation must have a location or spatial reference.")
    environment = value["environment"]
    if not isinstance(environment, list):
        raise TypeError("environment must be a list.")
    elements = []
    for index, element in enumerate(environment):
        if not isinstance(element, dict) or set(element) != set(FIELDS):
            raise ValueError("Each environmental element must have the same four fixed fields.")
        item = {name: normalize_field(element[name], "environment[{}].{}".format(index, name)) for name in FIELDS}
        if not item["subject"]:
            raise ValueError("An environmental element must have a supported subject.")
        if item["relation"] and not item["location"]:
            raise ValueError("An environmental relation must have a spatial reference.")
        elements.append(item)
    elements.sort(key=lambda element: tuple(element[name] for name in FIELDS))
    normalized["environment"] = elements
    return normalized


def serialize_semantics(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def main():
    device = setup()
    evidence = load_evidence()
    template = Config.TEMPLATE_PATH.read_text(encoding="utf-8-sig").strip()
    if not template:
        raise ValueError("The Markdown instruction template is empty.")
    output_dir = Config.OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "input_evidence.json", evidence)
    tokenizer, model, dtype = load_model(device)
    messages = build_messages(template, evidence)
    print("Instruction template:", Config.TEMPLATE_PATH)
    print("Input descriptions:", Config.INPUT_PATH)
    print("Running one semantic-aggregation generation...")
    raw_text, token_info = generate_structure(tokenizer, model, messages, device)
    (output_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    info = {
        "status": "pending_validation",
        "model_path": str(Config.MODEL_PATH),
        "model_type": model.config.model_type,
        "input_path": str(Config.INPUT_PATH),
        "template_path": str(Config.TEMPLATE_PATH),
        "seed": Config.SEED,
        "dtype": str(dtype),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "do_sample": False,
        "num_beams": 1,
        "generation_calls": 1,
        "raw_json_valid": False,
        "schema_valid": False,
        **token_info,
    }
    try:
        parsed = strict_json_loads(raw_text.strip())
        info["raw_json_valid"] = True
        normalized = normalize_semantics(parsed)
        info["schema_valid"] = True
    except (ValueError, TypeError) as exc:
        info["status"] = "invalid_output"
        info["error"] = str(exc)
        save_json(output_dir / "run_info.json", info)
        raise RuntimeError("Qwen output did not satisfy the schema. Raw output saved to {}. No retry was performed.".format(output_dir / "raw_response.txt")) from exc
    save_json(output_dir / "semantic.json", normalized)
    (output_dir / "semantic_serialized.txt").write_text(serialize_semantics(normalized) + "\n", encoding="utf-8")
    info["status"] = "success"
    save_json(output_dir / "run_info.json", info)
    print("\nNormalized semantic structure:")
    print(json.dumps(normalized, ensure_ascii=False, indent=2))
    print("Raw JSON valid:", info["raw_json_valid"])
    print("Schema valid:", info["schema_valid"])
    print("Generated tokens:", info["generated_tokens"])
    print("Token limit reached:", info["token_limit_reached"])
    print("Semantic JSON saved:", output_dir / "semantic.json")
    print("Serialized semantics saved:", output_dir / "semantic_serialized.txt")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()

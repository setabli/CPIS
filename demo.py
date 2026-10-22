import argparse
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import importlib.util
import hashlib
import json
import math
import random
import sys
import types
from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from tqdm.auto import tqdm


BASE_DIR = Path(__file__).resolve().parent
UNET_DIR = BASE_DIR / "unet"


def build_parser():
    parser = argparse.ArgumentParser(description="Masked image hiding and recovery with correct keys.")
    parser.add_argument("--image_path", type=str,
                        default="path/to/your/secret.png")
    parser.add_argument("--mask_path", type=str,
                        default="path/to/your/mask.png")
    parser.add_argument("--private_key", type=str,
                        default=" ")
    parser.add_argument("--public_key", type=str,
                        default=" ")
    parser.add_argument("--save_path", type=str, default=str(BASE_DIR / "output"))
    parser.add_argument("--model_path_diffusion", type=str,
                        default=str(BASE_DIR / "sd2"))
    parser.add_argument("--unet_arch_path", type=str, default=str(UNET_DIR / "model.py"))
    parser.add_argument("--unet_checkpoint", type=str,
                        default=str(UNET_DIR / "training_output" / "your_run" / "checkpoints" / "final_model.pt"))
    parser.add_argument("--unet_run_config", type=str, default="",
                        help="Optional saved run config.json; inferred from the checkpoint when empty.")
    parser.add_argument("--len_key", type=str, default="",
                        help="Training-time 32-character hexadecimal LEN key; read from the saved run config when empty.")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--shuffle_step", type=int, default=25,
                        help="Zero-based public-denoising index; shuffle is applied after this step.")
    parser.add_argument("--inject_step", type=int, default=25,
                        help="Zero-based public-denoising index; LEN injection follows any shuffle at this step.")
    parser.add_argument("--recover_step", type=int, default=None,
                        help="Zero-based public-inversion index; correction is applied before this step.")
    parser.add_argument("--inject_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def configure_steps(args):
    if args.num_steps < 2:
        raise ValueError("num_steps must be at least 2.")
    for name in ("shuffle_step", "inject_step"):
        if not 0 <= getattr(args, name) < args.num_steps:
            raise ValueError("{} must be a valid zero-based loop index.".format(name))
    args.unshuffle_step = args.num_steps - args.shuffle_step - 1
    if args.recover_step is not None and not 0 <= args.recover_step < args.num_steps:
        raise ValueError("recover_step must be a valid zero-based loop index.")
    if not math.isfinite(args.inject_scale) or not math.isfinite(args.guidance_scale):
        raise ValueError("inject_scale and guidance_scale must be finite.")


def setup(args):
    configure_steps(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_tensor_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_external_architecture(path):
    path = project_path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("External UNet architecture file not found: {}".format(path))
    package_name = "_cpis_corrector_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package
    module_name = package_name + "." + path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_path = list(sys.path)
    previous_config = sys.modules.get("config")
    local_config_path = path.parent / "config.py"
    config_replaced = local_config_path.is_file()
    try:
        sys.path.insert(0, str(path.parent))
        if config_replaced:
            config_spec = importlib.util.spec_from_file_location("config", str(local_config_path))
            config_module = importlib.util.module_from_spec(config_spec)
            sys.modules["config"] = config_module
            config_spec.loader.exec_module(config_module)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
        if config_replaced:
            if previous_config is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = previous_config
    return module


def read_training_config(args, checkpoint_path):
    explicit = args.unet_run_config.strip()
    config_path = (project_path(explicit) if explicit
                   else checkpoint_path.parent.parent / "config.json")
    if not config_path.is_file():
        if explicit:
            raise FileNotFoundError("Saved UNet run config not found: {}".format(config_path))
        return {}, None
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise TypeError("The saved UNet run config must be a JSON object.")
    return config, config_path


def restore_network_config(module, checkpoint, run_config):
    config = getattr(module, "Config", None)
    if config is None:
        raise AttributeError("The external UNet module must import its local Config class.")
    for name, field in (("channels", "CHANNELS"), ("size", "SIZE"),
                        ("widths", "WIDTHS"), ("groups", "GROUPS")):
        value = checkpoint.get(name, run_config.get(field, getattr(config, field)))
        normalize = (lambda item: tuple(int(x) for x in item)) if field == "WIDTHS" else int
        value = normalize(value)
        if name in checkpoint and field in run_config:
            if value != normalize(run_config[field]):
                raise ValueError("Checkpoint and run config disagree on {}.".format(field))
        setattr(config, field, value)
    if config.CHANNELS < 1 or config.SIZE < 8 or config.SIZE % 8 != 0:
        raise ValueError("UNet input channels must be positive and size must be divisible by 8.")
    if len(config.WIDTHS) != 4 or config.GROUPS < 1:
        raise ValueError("CorrectionUNet requires four widths and a positive GroupNorm group count.")
    if any(width < 1 or width % config.GROUPS != 0 for width in config.WIDTHS):
        raise ValueError("Each UNet width must be positive and divisible by the GroupNorm group count.")
    return config


def delta_from_key(key, channels, size, delta_std, device):
    try:
        key_bytes = bytes.fromhex(key)
    except ValueError as exc:
        raise ValueError("The LEN key must contain 32 hexadecimal characters.") from exc
    if len(key_bytes) != 16:
        raise ValueError("The LEN key must contain 32 hexadecimal characters.")
    if not math.isfinite(delta_std) or delta_std <= 0:
        raise ValueError("The saved DELTA_STD must be a positive finite value.")
    digest = hashlib.sha256(b"unet-demo-delta|" + key_bytes).digest()
    seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    delta = torch.randn((1, channels, size, size), generator=generator, dtype=torch.float32)
    return (delta_std * delta).to(device)


def load_corrector(args, device):
    checkpoint_path = project_path(args.unet_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("UNet checkpoint not found: {}".format(checkpoint_path))
    checkpoint = load_tensor_file(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint must contain a state dictionary.")
    run_config, run_config_path = read_training_config(args, checkpoint_path)
    state = next((checkpoint[name] for name in ("model_state", "model_state_dict", "state_dict")
                  if name in checkpoint), None)
    if state is None:
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            state = checkpoint
        else:
            raise ValueError("No supported model state was found in the checkpoint.")
    state = {name[7:] if name.startswith("module.") else name: value for name, value in state.items()}
    module = load_external_architecture(args.unet_arch_path)
    config = restore_network_config(module, checkpoint, run_config)
    model_class = getattr(module, "CorrectionUNet", None)
    if model_class is None:
        raise AttributeError("CorrectionUNet was not found in the architecture file.")
    model = model_class()
    model.load_state_dict(state, strict=True)
    if "groups" in checkpoint:
        groups = {layer.num_groups for layer in model.modules() if isinstance(layer, torch.nn.GroupNorm)}
        if groups and groups != {int(checkpoint["groups"])}:
            raise ValueError("The external architecture uses different GroupNorm groups from the checkpoint.")
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    if args.recover_step is None:
        args.recover_step = int(checkpoint.get("recover_step", args.num_steps - args.inject_step - 1))
    if args.shuffle_seed is None:
        args.shuffle_seed = int(checkpoint.get("shuffle_seed", args.seed))
    if not 0 <= args.recover_step < args.num_steps:
        raise ValueError("recover_step must be a valid zero-based loop index.")
    for name, expected in (
        ("num_steps", args.num_steps), ("inject_step", args.inject_step),
        ("shuffle_step", args.shuffle_step), ("unshuffle_step", args.unshuffle_step),
        ("recover_step", args.recover_step), ("inject_scale", args.inject_scale),
        ("guidance_scale", args.guidance_scale), ("shuffle_seed", args.shuffle_seed),
        ("background_mode", "fixed"),
        ("shuffle_phase", "after_denoising_step"),
        ("recovery_event_phase", "before_inverse_step"),
    ):
        if name in checkpoint and checkpoint[name] != expected:
            raise ValueError("Checkpoint setting mismatch: {}.".format(name))
    if args.len_key.strip():
        key = args.len_key.strip()
        key_source = "parser len_key"
    elif str(run_config.get("CORRECT_KEY", "")).strip():
        key = str(run_config["CORRECT_KEY"]).strip()
        key_source = "saved run config.json"
    elif run_config_path is None:
        key = str(getattr(config, "CORRECT_KEY", "")).strip()
        key_source = "local UNet config.py"
    else:
        key = ""
        key_source = "not set"
    if not key:
        raise ValueError("Set len_key to the training-time key, or retain CORRECT_KEY in the saved run config.json.")
    delta_std = float(run_config.get("DELTA_STD", getattr(config, "DELTA_STD", 0.25)))
    key_input = delta_from_key(key, config.CHANNELS, config.SIZE, delta_std, device)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "run_config": str(run_config_path) if run_config_path else None,
        "pair_id": str(checkpoint.get("pair_id", run_config.get("PAIR_ID", ""))).strip(),
        "training_iteration": checkpoint.get("iteration"),
        "input_shape": list(key_input.shape),
        "delta_std": delta_std,
    }
    print("External UNet class: CorrectionUNet")
    print("UNet checkpoint:", checkpoint_path)
    print("UNet run config:", run_config_path if run_config_path else "local Config fallback")
    print("UNet training pair:", metadata["pair_id"] or "not recorded")
    print("UNet training iteration:", metadata["training_iteration"])
    print("UNet parameters:", sum(parameter.numel() for parameter in model.parameters()))
    print("LEN key source:", key_source)
    print("LEN input shape:", tuple(key_input.shape))
    print("LEN input standard-deviation scale:", delta_std)
    print("Shuffle seed:", args.shuffle_seed)
    print("Shuffle index:", args.shuffle_step)
    print("Inverse shuffle index:", args.unshuffle_step)
    print("Injection index:", args.inject_step)
    print("Correction index:", args.recover_step)
    print("All step indices are zero-based.")
    return model, key_input, metadata


def read_inputs(image_path, mask_path):
    image_path = project_path(image_path)
    mask_path = project_path(mask_path)
    if not image_path.is_file():
        raise FileNotFoundError("Secret image not found: {}".format(image_path))
    if not mask_path.is_file():
        raise FileNotFoundError("Editable-region mask not found: {}".format(mask_path))
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)).copy()
    with Image.open(mask_path) as source:
        mask = source.convert("L").resize((512, 512), Image.Resampling.NEAREST)
        mask_hr = np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5
    if not mask_hr.any():
        raise ValueError("The editable mask is empty. White pixels identify the editable region.")
    return image, mask_hr


def project_background(latent, mask, reference):
    if latent.shape != reference.shape:
        raise ValueError("Latent and reference shapes must match.")
    return torch.where(mask.bool(), latent, reference)


def composite_pixels(generated, reference, mask_hr):
    if generated.shape != reference.shape or generated.shape[:2] != mask_hr.shape:
        raise ValueError("Image and mask shapes must match for pixel composition.")
    return np.where(mask_hr[:, :, None], generated, reference).astype(np.uint8)


def channel_permutation(channels, seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randperm(channels, generator=generator, device=device)


def shuffle_latent_channels_with_mask(latent, mask, indices):
    return torch.where(mask.bool(), latent[:, indices], latent)


def unshuffle_latent_channels_with_mask(latent, mask, indices):
    inverse = torch.argsort(indices)
    return torch.where(mask.bool(), latent[:, inverse], latent)


class ODESolve:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = model.device
        self.dtype = next(model.unet.parameters()).dtype
        self.scheduler = DDIMScheduler.from_config(
            model.scheduler.config, clip_sample=False, thresholding=False,
            set_alpha_to_one=False, steps_offset=1, timestep_spacing="leading",
        )
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
        self.scheduler.set_timesteps(args.num_steps)
        self.inverse_scheduler.set_timesteps(args.num_steps)
        self.scaling_factor = float(model.vae.config.scaling_factor)

    @torch.inference_mode()
    def image2latent(self, image):
        value = torch.from_numpy(np.ascontiguousarray(image)).to(self.device, self.dtype)
        value = value.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return self.model.vae.encode(value).latent_dist.mode() * self.scaling_factor

    @torch.inference_mode()
    def latent2image(self, latent):
        image = self.model.vae.decode(latent / self.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    @torch.inference_mode()
    def init_prompt(self, prompt):
        texts = [prompt] if self.args.guidance_scale == 1.0 else ["", prompt]
        tokens = self.model.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.model.tokenizer.model_max_length, return_tensors="pt",
        )
        kwargs = {}
        if getattr(self.model.text_encoder.config, "use_attention_mask", False):
            kwargs["attention_mask"] = tokens.attention_mask.to(self.device)
        return self.model.text_encoder(tokens.input_ids.to(self.device), **kwargs)[0]

    @torch.inference_mode()
    def get_noise_pred(self, latent, timestep, context):
        if self.args.guidance_scale == 1.0:
            return self.model.unet(latent, timestep, encoder_hidden_states=context).sample
        value = torch.cat((latent, latent), dim=0)
        output = self.model.unet(value, timestep, encoder_hidden_states=context).sample
        unconditional, conditional = output.chunk(2)
        return unconditional + self.args.guidance_scale * (conditional - unconditional)

    def injection_coefficient(self, timestep, latent):
        timestep = int(timestep)
        previous = timestep - self.scheduler.config.num_train_timesteps // self.args.num_steps
        alpha = self.scheduler.alphas_cumprod[timestep].to(latent.device, latent.dtype)
        previous_alpha = (self.scheduler.alphas_cumprod[previous] if previous >= 0
                          else self.scheduler.final_alpha_cumprod).to(latent.device, latent.dtype)
        return (1 - previous_alpha).sqrt() - (previous_alpha / alpha).sqrt() * (1 - alpha).sqrt()

    @torch.inference_mode()
    def invert(self, prompt, start_latent, mask_latent, background_reference, is_forward,
               inject_noise=None, unet_correction=None, shuffle_indices=None,
               unshuffle_indices=None, description="Diffusion"):
        if inject_noise is not None and is_forward:
            raise ValueError("LEN injection is only applied during hiding denoising.")
        if unet_correction is not None and not is_forward:
            raise ValueError("LEN correction is only applied during public inversion.")
        if shuffle_indices is not None and is_forward:
            raise ValueError("Channel shuffle is only applied during public denoising.")
        if unshuffle_indices is not None and not is_forward:
            raise ValueError("Inverse channel shuffle is only applied during public inversion.")
        context = self.init_prompt(prompt)
        scheduler = self.inverse_scheduler if is_forward else self.scheduler
        latent = project_background(start_latent.detach().clone(), mask_latent, background_reference)
        for index, timestep in enumerate(tqdm(scheduler.timesteps, desc=description)):
            if unet_correction is not None and index == self.args.recover_step:
                latent = latent - unet_correction * mask_latent
            if unshuffle_indices is not None and index == self.args.unshuffle_step:
                latent = unshuffle_latent_channels_with_mask(latent, mask_latent, unshuffle_indices)
            model_output = self.get_noise_pred(latent, timestep, context)
            if is_forward:
                latent = scheduler.step(model_output, timestep, latent).prev_sample
            else:
                latent = scheduler.step(model_output, timestep, latent, eta=0.0).prev_sample
            if shuffle_indices is not None and index == self.args.shuffle_step:
                latent = shuffle_latent_channels_with_mask(latent, mask_latent, shuffle_indices)
            if inject_noise is not None and index == self.args.inject_step:
                coefficient = self.injection_coefficient(timestep, latent)
                latent = latent + self.args.inject_scale * coefficient * inject_noise * mask_latent
            latent = project_background(latent, mask_latent, background_reference)
            if not torch.isfinite(latent).all().item():
                raise RuntimeError("A non-finite latent was encountered.")
        return latent


@torch.inference_mode()
def main(args):
    device = setup(args)
    output_dir = project_path(args.save_path)
    diffusion_path = project_path(args.model_path_diffusion)
    if not diffusion_path.is_dir():
        raise FileNotFoundError("Local diffusion model directory not found: {}".format(diffusion_path))
    image_gt, mask_hr = read_inputs(args.image_path, args.mask_path)
    corrector, len_input, _ = load_corrector(args, device)
    print("Loading the local diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        str(diffusion_path), torch_dtype=torch.float32, local_files_only=True,
    ).to(device)
    pipeline.unet.eval().requires_grad_(False)
    pipeline.vae.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)
    try:
        pipeline.disable_xformers_memory_efficient_attention()
    except (AttributeError, ModuleNotFoundError):
        pass
    ode = ODESolve(pipeline, args)
    image_gt_latent = ode.image2latent(image_gt)
    mask_latent = torch.from_numpy(mask_hr.astype(np.float32))[None, None].to(device)
    mask_latent = F.interpolate(mask_latent, size=image_gt_latent.shape[-2:], mode="nearest")
    if not mask_latent.any().item():
        raise ValueError("The editable region disappeared during latent-mask downsampling.")
    if len_input.shape != image_gt_latent.shape:
        raise ValueError("LEN input shape {} differs from latent shape {}.".format(tuple(len_input.shape), tuple(image_gt_latent.shape)))
    correction = corrector(len_input)
    if correction.shape != image_gt_latent.shape or not torch.isfinite(correction).all().item():
        raise ValueError("The correction network returned an invalid tensor.")
    correction = correction * mask_latent
    output_dir.mkdir(parents=True, exist_ok=True)
    sender_reference = image_gt_latent.detach().clone()
    print("Device:", device)
    print("Scheduler prediction type:", ode.scheduler.config.prediction_type)
    latent_noise = ode.invert(
        args.private_key, image_gt_latent, mask_latent, sender_reference, True,
        description="Private inversion",
    )
    indices = channel_permutation(latent_noise.shape[1], args.shuffle_seed, device)
    stego_latent = ode.invert(
        args.public_key, latent_noise, mask_latent, sender_reference, False,
        inject_noise=len_input, shuffle_indices=indices, description="Public denoising",
    )
    stego_image = composite_pixels(ode.latent2image(stego_latent), image_gt, mask_hr)
    Image.fromarray(image_gt).save(output_dir / "secret.png")
    Image.fromarray(mask_hr.astype(np.uint8) * 255).save(output_dir / "mask.png")
    Image.fromarray(stego_image).save(output_dir / "stego.png")
    with Image.open(output_dir / "stego.png") as source:
        received_image = np.asarray(source.convert("RGB")).copy()
    received_latent = ode.image2latent(received_image)
    receiver_reference = received_latent.detach().clone()
    receiver_indices = channel_permutation(received_latent.shape[1], args.shuffle_seed, device)
    latent_after_correction = ode.invert(
        args.public_key, received_latent, mask_latent, receiver_reference, True,
        unet_correction=correction, unshuffle_indices=receiver_indices,
        description="Public inversion with correct key",
    )
    recovered_latent = ode.invert(
        args.private_key, latent_after_correction, mask_latent, receiver_reference, False,
        description="Private denoising with correct key",
    )
    recovered_image = composite_pixels(ode.latent2image(recovered_latent), received_image, mask_hr)
    Image.fromarray(recovered_image).save(output_dir / "recover.png")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main(build_parser().parse_args())

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
torch = None


def ddim_coefficients(alpha_t, alpha_previous):
    if not (0.0 < alpha_t <= 1.0 and 0.0 < alpha_previous <= 1.0):
        raise ValueError("DDIM alpha products must be in (0, 1].")
    a = math.sqrt(alpha_previous / alpha_t)
    b = math.sqrt(max(0.0, 1.0 - alpha_previous))
    b -= a * math.sqrt(max(0.0, 1.0 - alpha_t))
    return a, b


def prediction_to_epsilon(model_output, sample, alpha_t, prediction_type):
    if prediction_type == "epsilon":
        return model_output
    if prediction_type == "v_prediction":
        return math.sqrt(alpha_t) * model_output + math.sqrt(1.0 - alpha_t) * sample
    if prediction_type == "sample":
        denominator = math.sqrt(max(1e-20, 1.0 - alpha_t))
        return (sample - math.sqrt(alpha_t) * model_output) / denominator
    raise ValueError("Unsupported scheduler prediction type: " + prediction_type)


def solve_inverse_step(target, initial, forward, project, measure, a,
                       max_iterations, atol, rtol, relaxation, max_backtracking):
    estimate = project(initial)
    residual = forward(estimate) - target
    error = measure(residual)
    initial_error = error
    tolerance = atol + rtol * measure(target)
    evaluations = 1
    accepted_updates = 0
    status = "max_iterations"
    if not math.isfinite(error):
        raise RuntimeError("The initial inversion residual is not finite. Try float32.")
    for _ in range(max_iterations):
        if error <= tolerance:
            status = "converged"
            break
        accepted = False
        rate = relaxation
        for _ in range(max_backtracking + 1):
            candidate = project(estimate - rate * residual / a)
            candidate_residual = forward(candidate) - target
            candidate_error = measure(candidate_residual)
            evaluations += 1
            if math.isfinite(candidate_error) and candidate_error < error:
                estimate = candidate
                residual = candidate_residual
                error = candidate_error
                accepted_updates += 1
                accepted = True
                break
            rate *= 0.5
        if not accepted:
            status = "stalled"
            break
    if error <= tolerance:
        status = "converged"
    return estimate, {
        "initial_residual_rms": initial_error,
        "final_residual_rms": error,
        "tolerance": tolerance,
        "accepted_updates": accepted_updates,
        "forward_evaluations": evaluations,
        "converged": status == "converged",
        "status": status,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--image_path", default="")
    parser.add_argument("--mask_path", default="")
    parser.add_argument("--save_path", default=str(BASE_DIR / "output"))
    parser.add_argument("--private_prompt", "--private_key", dest="private_prompt",
                        default="")
    parser.add_argument("--public_prompt", "--public_key", dest="public_prompt",
                        default="")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force_cpu", action="store_true", default=False)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perm_key", "--lep_key", dest="perm_key", default="1234")
    parser.add_argument("--perm_ratio", type=float, default=1.0,
                        help="Fraction of editable spatial positions selected for channel permutation.")
    parser.add_argument("--perm_step", type=int, default=1,
                        help="Apply permutation BEFORE this one-based sender denoising step.")
    parser.add_argument("--wrong_perm_keys", "--wrong_lep_keys", dest="wrong_perm_keys", nargs="*",
                        default=["incorrect-perm-key-01", "incorrect-perm-key-02", "incorrect-perm-key-03"])
    parser.add_argument("--len_key", default="replace-this-with-your-key")
    parser.add_argument("--wrong_len_keys", nargs="*",
                        default=["incorrect-len-key-01", "incorrect-len-key-02", "incorrect-len-key-03"])
    parser.add_argument("--skip_wrong_both", action="store_true", default=False,
                        help="Skip paired incorrect LEP/LEN trials; otherwise pair the two key lists with zip.")
    parser.add_argument("--test_without_perm", action="store_true", default=False,
                        help="Also recover the same stego without inverse permutation, using the correct LEN key.")
    parser.add_argument("--len_step", type=int, default=25,
                        help="One-based position in the sender's descending denoising loop.")
    parser.add_argument("--len_strength", type=float, default=1.0)
    parser.add_argument("--inverse_iters", type=int, default=12)
    parser.add_argument("--inverse_atol", type=float, default=1e-5)
    parser.add_argument("--inverse_rtol", type=float, default=1e-5)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--max_backtracking", type=int, default=3)
    args = parser.parse_args()
    if args.num_steps < 1 or not 1 <= args.len_step <= args.num_steps:
        parser.error("len_step must be between 1 and num_steps.")
    if not 1 <= args.perm_step <= args.num_steps:
        parser.error("perm_step must be between 1 and num_steps.")
    if not 0.0 <= args.perm_ratio <= 1.0:
        parser.error("perm_ratio must be in [0, 1].")
    if args.image_size < 8 or args.image_size % 8:
        parser.error("image_size must be a positive multiple of 8.")
    if args.inverse_iters < 1 or args.max_backtracking < 0:
        parser.error("inverse_iters must be positive; max_backtracking cannot be negative.")
    if not 0.0 < args.relaxation <= 1.0:
        parser.error("relaxation must be in (0, 1].")
    if args.inverse_atol < 0 or args.inverse_rtol < 0 or args.len_strength < 0:
        parser.error("Tolerances and len_strength cannot be negative.")
    if not args.len_key or not args.perm_key:
        parser.error("len_key and perm_key must not be empty.")
    for label, correct, incorrect in (
        ("LEN", args.len_key, args.wrong_len_keys),
        ("permutation", args.perm_key, args.wrong_perm_keys),
    ):
        if any(not key for key in incorrect):
            parser.error("Incorrect %s keys must not be empty." % label)
        if correct in incorrect:
            parser.error("An incorrect %s key equals the correct key." % label)
        if len(set(incorrect)) != len(incorrect):
            parser.error("Incorrect %s keys must be unique." % label)
    return args


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def masked_rms(value, mask):
    denominator = float(mask.sum().item()) * value.shape[1]
    if denominator <= 0:
        raise ValueError("The mask contains no editable latent positions.")
    return math.sqrt(float((value.double().square() * mask.double()).sum().item()) / denominator)


def latent_metrics(reference, estimate, mask):
    difference = estimate - reference
    return {
        "global_rmse": math.sqrt(float(difference.double().square().mean().item())),
        "editable_rmse": masked_rms(difference, mask),
        "editable_mae": float((difference.double().abs() * mask.double()).sum().item())
        / (float(mask.sum().item()) * difference.shape[1]),
    }


def image_metrics(reference, estimate, mask):
    squared = (reference.astype(np.float64) - estimate.astype(np.float64)) ** 2
    selected = mask.astype(bool)
    def to_psnr(mse):
        return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
    preserved = ~selected
    return {
        "global_psnr_db": to_psnr(float(squared.mean())),
        "editable_psnr_db": to_psnr(float(squared[selected].mean())),
        "preserved_max_absolute_error": int(np.abs(
            reference.astype(np.int16) - estimate.astype(np.int16)
        )[preserved].max()) if preserved.any() else None,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def key_fingerprint(key):
    if key is None:
        return "none"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def build_channel_permutation(shape, key, participate_ratio):
    if len(shape) != 4 or shape[1] < 1:
        raise ValueError("The latent shape must be (N, C, H, W).")
    if not 0.0 <= participate_ratio <= 1.0:
        raise ValueError("participate_ratio must be in [0, 1].")
    _, channels, height, width = shape
    description = "CPIS-LEP-v1|%s|%d,%d,%d" % (key, channels, height, width)
    digest = hashlib.sha256(description.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:16], "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = np.stack([generator.permutation(channels) for _ in range(height * width)])
    indices = indices.T.reshape(1, channels, height, width).astype(np.int64)
    inverse_indices = np.argsort(indices, axis=1).astype(np.int64)
    participation = generator.random((1, 1, height, width)) < participate_ratio
    return indices, inverse_indices, participation


def build_key_trials(args):
    trials = [("correct_keys", args.perm_key, args.len_key, "recover.png")]
    trials.extend(("wrong_lep_%02d" % index, key, args.len_key,
                   "recover_wrong_lep_%02d.png" % index)
                  for index, key in enumerate(args.wrong_perm_keys, 1))
    trials.extend(("wrong_len_%02d" % index, args.perm_key, key,
                   "recover_wrong_len_%02d.png" % index)
                  for index, key in enumerate(args.wrong_len_keys, 1))
    if not args.skip_wrong_both:
        trials.extend(("wrong_both_%02d" % index, perm_key, len_key,
                       "recover_wrong_both_%02d.png" % index)
                      for index, (perm_key, len_key) in enumerate(
                          zip(args.wrong_perm_keys, args.wrong_len_keys), 1))
    if args.test_without_perm:
        trials.append(("without_inverse_lep", None, args.len_key, "recover_without_lep.png"))
    return trials


def load_inputs(args):
    image_path = resolve_path(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError("Set the --image_path parser default: " + str(image_path))
    size = (args.image_size, args.image_size)
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(image_path) as image:
        secret = np.asarray(image.convert("RGB").resize(size, resampling.LANCZOS)).copy()
    if args.mask_path:
        mask_path = resolve_path(args.mask_path)
        if not mask_path.is_file():
            raise FileNotFoundError("Set the --mask_path parser default, or use an empty string: " + str(mask_path))
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L").resize(size, resampling.NEAREST)) >= 128
    else:
        mask = np.ones(size[::-1], dtype=bool)
    if not mask.any():
        raise ValueError("The mask contains no editable pixels. White means editable.")
    return secret, mask


class ODESolve:
    def __init__(self, model, args):
        from tqdm.auto import tqdm
        self.progress = tqdm
        self.model = model
        self.args = args
        self.device = model.device
        self.dtype = next(model.unet.parameters()).dtype
        self.scheduler = model.scheduler
        self.scheduler.set_timesteps(args.num_steps)
        self.timesteps = [int(item) for item in self.scheduler.timesteps.cpu().tolist()]
        self.alpha = [float(self.scheduler.alphas_cumprod[t]) for t in self.timesteps]
        self.alpha.append(float(self.scheduler.final_alpha_cumprod))
        self.prediction_type = self.scheduler.config.prediction_type
        self.scaling_factor = float(getattr(model.vae.config, "scaling_factor", 0.18215))
        self.context = {}
        self.permutation_cache = {}
        self.trace = []
        self.unet_calls = 0

    def encode_image(self, image):
        tensor = torch.from_numpy(image.copy()).to(device=self.device, dtype=self.dtype)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return self.model.vae.encode(tensor).latent_dist.mode() * self.scaling_factor

    def decode_image(self, latent):
        image = self.model.vae.decode(latent / self.scaling_factor).sample
        image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
        image = image[0].permute(1, 2, 0).float().cpu().numpy()
        return np.rint(image * 255.0).astype(np.uint8)

    def mask_tensor(self, high_resolution_mask, latent):
        mask = torch.from_numpy(high_resolution_mask.astype(np.float32))[None, None]
        mask = torch.nn.functional.interpolate(mask, size=latent.shape[-2:], mode="nearest")
        mask = mask.to(device=self.device, dtype=self.dtype)
        if not bool(mask.any().item()):
            raise ValueError("The editable mask disappeared after latent downsampling.")
        return mask

    def prompt_embeddings(self, prompt):
        if prompt not in self.context:
            tokenizer = self.model.tokenizer
            tokens = tokenizer([prompt], padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt")
            encoder_args = {}
            if getattr(self.model.text_encoder.config, "use_attention_mask", False):
                encoder_args["attention_mask"] = tokens.attention_mask.to(self.device)
            self.context[prompt] = self.model.text_encoder(
                tokens.input_ids.to(self.device), **encoder_args
            )[0]
        return self.context[prompt]

    def predict_epsilon(self, latent, index, prompt):
        timestep = self.timesteps[index]
        conditional = self.model.unet(
            latent, timestep, encoder_hidden_states=self.prompt_embeddings(prompt)
        ).sample
        self.unet_calls += 1
        if self.args.guidance_scale != 1.0:
            unconditional = self.model.unet(
                latent, timestep, encoder_hidden_states=self.prompt_embeddings("")
            ).sample
            self.unet_calls += 1
            conditional = unconditional + self.args.guidance_scale * (conditional - unconditional)
        return prediction_to_epsilon(conditional, latent, self.alpha[index], self.prediction_type)

    def key_noise(self, shape, key):
        description = "CPIS-LEN-v1|" + key + "|" + ",".join(map(str, shape))
        digest = hashlib.sha256(description.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(tuple(shape), generator=generator, dtype=torch.float32, device="cpu")
        return noise.to(device=self.device, dtype=self.dtype) * self.args.len_strength

    def permutation_plan(self, shape, key):
        identifier = (tuple(shape[1:]), key, self.args.perm_ratio)
        if identifier not in self.permutation_cache:
            indices, inverse_indices, participation = build_channel_permutation(
                shape, key, self.args.perm_ratio
            )
            self.permutation_cache[identifier] = (
                torch.from_numpy(indices).to(device=self.device, dtype=torch.long),
                torch.from_numpy(inverse_indices).to(device=self.device, dtype=torch.long),
                torch.from_numpy(participation).to(device=self.device),
            )
        return self.permutation_cache[identifier]

    def permute_channels(self, latent, mask, key, inverse=False):
        indices, inverse_indices, participation = self.permutation_plan(latent.shape, key)
        selected = participation & mask.bool()
        gather_indices = inverse_indices if inverse else indices
        gather_indices = gather_indices.expand(latent.shape[0], -1, -1, -1)
        transformed = torch.gather(latent, dim=1, index=gather_indices)
        return torch.where(selected.expand_as(latent), transformed, latent)

    def permutation_statistics(self, shape, mask, key):
        indices, _, participation = build_channel_permutation(shape, key, self.args.perm_ratio)
        editable = mask.detach().float().cpu().numpy().astype(bool)
        selected = participation & editable
        identity = np.arange(shape[1], dtype=np.int64)[None, :, None, None]
        changed_channels = (indices != identity) & selected
        denominator = int(editable.sum())
        return {
            "selected_position_fraction": float(selected.sum()) / denominator,
            "nonidentity_position_fraction": float(changed_channels.any(axis=1, keepdims=True).sum())
            / denominator,
            "changed_channel_index_fraction": float(changed_channels.sum()) / (denominator * shape[1]),
            "generator": "SHA-256 domain-separated seed and NumPy PCG64 on CPU",
        }

    def step(self, latent, index, prompt, mask, reference, delta):
        a, b = ddim_coefficients(self.alpha[index], self.alpha[index + 1])
        epsilon = self.predict_epsilon(latent, index, prompt)
        if delta is not None and index + 1 == self.args.len_step:
            epsilon = epsilon + delta * mask
        output = a * latent + b * epsilon
        return output * mask + reference * (1.0 - mask)

    def denoise(self, latent, prompt, mask, stage, delta=None, perm_key=None):
        current = latent.clone()
        reference = latent.clone()
        for index in self.progress(range(len(self.timesteps)), desc=stage):
            if perm_key is not None and index + 1 == self.args.perm_step:
                current = self.permute_channels(current, mask, perm_key)
            current = self.step(current, index, prompt, mask, reference, delta)
            if not bool(torch.isfinite(current).all().item()):
                raise RuntimeError("Non-finite latent during " + stage)
        return current

    def invert(self, latent, prompt, mask, stage, delta=None, perm_key=None):
        current = latent.clone()
        reference = latent.clone()
        iterator = self.progress(range(len(self.timesteps) - 1, -1, -1), desc=stage)
        for reverse_index, index in enumerate(iterator):
            a, _ = ddim_coefficients(self.alpha[index], self.alpha[index + 1])
            def project(value):
                return value * mask + reference * (1.0 - mask)
            def forward(value):
                return self.step(value, index, prompt, mask, reference, delta)
            def measure(value):
                return masked_rms(value, mask)
            current, info = solve_inverse_step(
                current, current, forward, project, measure, a,
                self.args.inverse_iters, self.args.inverse_atol, self.args.inverse_rtol,
                self.args.relaxation, self.args.max_backtracking,
            )
            perm_active = perm_key is not None and index + 1 == self.args.perm_step
            if perm_active:
                current = self.permute_channels(current, mask, perm_key, inverse=True)
            info.update({
                "stage": stage,
                "inverse_step": reverse_index + 1,
                "sender_step": index + 1,
                "timestep": self.timesteps[index],
                "len_active": delta is not None and index + 1 == self.args.len_step,
                "perm_inverse_active": perm_active,
            })
            self.trace.append(info)
            iterator.set_postfix(rms="%.2e" % info["final_residual_rms"],
                                 updates=info["accepted_updates"], status=info["status"])
        rows = [row for row in self.trace if row["stage"] == stage]
        summary = {
            "steps": len(rows),
            "converged_steps": sum(row["converged"] for row in rows),
            "stalled_steps": sum(row["status"] == "stalled" for row in rows),
            "max_iterations_steps": sum(row["status"] == "max_iterations" for row in rows),
            "worst_final_step_residual_rms": max(row["final_residual_rms"] for row in rows),
        }
        print(stage + ": " + json.dumps(summary))
        return current

    def receive(self, received_latent, mask, public_prompt, private_prompt,
                len_key, perm_key, stage):
        delta = self.key_noise(received_latent.shape, len_key)
        recovered_noise = self.invert(received_latent, public_prompt, mask,
                                      stage + "/public_inverse", delta, perm_key)
        recovered_latent = self.denoise(recovered_noise, private_prompt, mask,
                                        stage + "/private_denoise")
        return recovered_noise, recovered_latent


def load_model(args):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    model_path = resolve_path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError("Set the --model_path parser default: " + str(model_path))
    device = "cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    dtype = torch.float32 if args.dtype == "float32" or device == "cpu" else torch.float16
    print("Loading local diffusion model...")
    model = StableDiffusionPipeline.from_pretrained(
        str(model_path), torch_dtype=dtype, local_files_only=True
    )
    model.scheduler = DDIMScheduler.from_config(
        model.scheduler.config, clip_sample=False, thresholding=False, timestep_spacing="leading"
    )
    model = model.to(device)
    for component in (model.unet, model.vae, model.text_encoder):
        component.eval()
        component.requires_grad_(False)
    if hasattr(model, "disable_xformers_memory_efficient_attention"):
        model.disable_xformers_memory_efficient_attention()
    model.set_progress_bar_config(disable=True)
    return model


def save_trace(path, rows):
    if not rows:
        return
    fields = ["stage", "inverse_step", "sender_step", "timestep", "len_active", "perm_inverse_active",
              "initial_residual_rms", "final_residual_rms", "tolerance",
              "accepted_updates", "forward_evaluations", "converged", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, report, trace):
    save_trace(output_dir / "inversion_trace.csv", trace)
    fields = ["condition", "perm_key_fingerprint", "len_key_fingerprint", "status", "image_file", "global_psnr_db",
              "editable_psnr_db", "preserved_max_absolute_error", "noise_editable_rmse",
              "secret_latent_editable_rmse", "inverse_steps", "inverse_converged_steps",
              "inverse_not_converged_steps", "worst_step_residual_rms", "unet_forward_calls",
              "elapsed_seconds", "error"]
    with (output_dir / "key_recovery_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in report["key_recovery_results"]:
            writer.writerow({name: json_safe(result.get(name, "")) for name in fields})
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, ensure_ascii=False, indent=2, allow_nan=False)


def synchronize_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_key(ode, received_latent, received_image, mask, mask_hr,
                 private_prompt, public_prompt, len_key, perm_key, condition,
                 image_file, secret, sender_noise, secret_latent, output_dir):
    start_trace = len(ode.trace)
    start_calls = ode.unet_calls
    synchronize_device()
    started = time.perf_counter()
    result = {
        "condition": condition,
        "perm_key_fingerprint": key_fingerprint(perm_key),
        "len_key_fingerprint": key_fingerprint(len_key),
        "status": "running",
        "image_file": image_file,
    }
    print("Testing %s; LEP key fingerprint: %s; LEN key fingerprint: %s" % (
        condition, result["perm_key_fingerprint"], result["len_key_fingerprint"]))
    try:
        recovered_noise, recovered_latent = ode.receive(
            received_latent, mask, public_prompt, private_prompt, len_key, perm_key,
            "receiver/" + condition,
        )
        recovered_raw = ode.decode_image(recovered_latent)
        recovered = np.where(mask_hr[:, :, None], recovered_raw, received_image).astype(np.uint8)
        Image.fromarray(recovered).save(output_dir / image_file)
        metrics = image_metrics(secret, recovered, mask_hr)
        noise_error = latent_metrics(sender_noise, recovered_noise, mask)
        secret_error = latent_metrics(secret_latent, recovered_latent, mask)
        result.update(metrics)
        result.update({
            "status": "completed",
            "noise_editable_rmse": noise_error["editable_rmse"],
            "secret_latent_editable_rmse": secret_error["editable_rmse"],
        })
        print("%s: Global PSNR = %.4f dB; Editable PSNR = %.4f dB; Preserved max error = %s" % (
            condition, metrics["global_psnr_db"], metrics["editable_psnr_db"],
            str(metrics["preserved_max_absolute_error"]),
        ))
    except RuntimeError as error:
        result["status"] = "failed"
        result["error"] = str(error)
        result["image_file"] = ""
        print("%s failed: %s" % (condition, str(error)))
        if "out of memory" in str(error).lower():
            raise
    finally:
        synchronize_device()
        rows = ode.trace[start_trace:]
        result.update({
            "elapsed_seconds": time.perf_counter() - started,
            "unet_forward_calls": ode.unet_calls - start_calls,
            "inverse_steps": len(rows),
            "inverse_converged_steps": sum(row["converged"] for row in rows),
            "inverse_not_converged_steps": sum(not row["converged"] for row in rows),
            "worst_step_residual_rms": max(
                (row["final_residual_rms"] for row in rows), default=None
            ),
        })
    return result


def main():
    global torch
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.force_cpu else args.gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch as torch_module
    torch = torch_module
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
            if hasattr(torch.backends.cuda, name):
                getattr(torch.backends.cuda, name)(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    secret, mask_hr = load_inputs(args)
    output_dir = resolve_path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(secret).save(output_dir / "secret.png")
    Image.fromarray(mask_hr.astype(np.uint8) * 255).save(output_dir / "mask.png")
    model = load_model(args)
    ode = ODESolve(model, args)
    print("Device: " + str(model.device))
    print("Dtype: " + str(ode.dtype))
    print("Scheduler prediction type: " + ode.prediction_type)
    print("Position-wise channel permutation: enabled. Extra correction network: disabled.")
    print("Private prompt: " + args.private_prompt)
    print("Public prompt: " + args.public_prompt)
    print("LEP step: %d/%d; scheduler timestep: %d; receiver inverse position: %d" % (
        args.perm_step, args.num_steps, ode.timesteps[args.perm_step - 1],
        args.num_steps - args.perm_step + 1,
    ))
    print("LEP participation ratio: %.4f; correct key fingerprint: %s" % (
        args.perm_ratio, key_fingerprint(args.perm_key)))
    print("LEN step: %d/%d; scheduler timestep: %d; receiver inverse position: %d" % (
        args.len_step, args.num_steps, ode.timesteps[args.len_step - 1],
        args.num_steps - args.len_step + 1,
    ))
    print("LEN strength: %.6g; correct key fingerprint: %s" % (
        args.len_strength, key_fingerprint(args.len_key)))
    trials = build_key_trials(args)
    print("Recovery trials: %d" % len(trials))
    print("All key conditions use the same saved stego.png, prompts, mask, and solver settings.")
    report = {
        "settings": {key: value for key, value in vars(args).items()
                     if key not in ("len_key", "wrong_len_keys", "perm_key", "wrong_perm_keys")},
        "correct_len_key_fingerprint": key_fingerprint(args.len_key),
        "correct_perm_key_fingerprint": key_fingerprint(args.perm_key),
        "incorrect_len_key_count": len(args.wrong_len_keys),
        "incorrect_perm_key_count": len(args.wrong_perm_keys),
        "recovery_trial_count": len(trials),
        "scheduler_prediction_type": ode.prediction_type,
        "scheduler_timesteps": ode.timesteps,
        "vae_scaling_factor": ode.scaling_factor,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "reference_policy": "Freeze the unedited latent positions to the input of each pass.",
        "receiver_reference_policy": "Use only the re-encoded received PNG; no sender reference latent.",
        "evaluation_protocol": "Change only the indicated LEP/LEN receiver subkeys; use the same received PNG and other conditions.",
        "permutation_type": "Independent channel permutation at selected editable spatial positions; spatial coordinates do not move.",
        "permutation_placement": "Before the raw DDIM denoising update at perm_step.",
        "inverse_order": "Numerically invert the raw DDIM update including LEN, then undo LEP at the matching step.",
        "residual_scope": "Forward residuals describe the raw DDIM inverse before inverse permutation.",
        "key_recovery_results": [],
        "status": "running",
    }
    synchronize_device()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            secret_latent = ode.encode_image(secret)
            mask = ode.mask_tensor(mask_hr, secret_latent)
            report["editable_pixel_fraction"] = float(mask_hr.mean())
            report["editable_latent_fraction"] = float(mask.float().mean().item())
            print("Editable pixel fraction: %.2f%%" % (100.0 * mask_hr.mean()))
            sender_noise = ode.invert(secret_latent, args.private_prompt, mask,
                                      "sender/private_inverse")
            report["permutation_statistics"] = ode.permutation_statistics(
                sender_noise.shape, mask, args.perm_key
            )
            print("LEP statistics: " + json.dumps(report["permutation_statistics"]))
            sender_delta = ode.key_noise(sender_noise.shape, args.len_key)
            stego_latent = ode.denoise(sender_noise, args.public_prompt, mask,
                                       "sender/public_denoise", sender_delta, args.perm_key)
            stego_raw = ode.decode_image(stego_latent)
            stego = np.where(mask_hr[:, :, None], stego_raw, secret).astype(np.uint8)
            Image.fromarray(stego).save(output_dir / "stego.png")
            print("Saved stego.png. The PNG receiver uses no sender intermediate latent.")
            with Image.open(output_dir / "stego.png") as image:
                received_image = np.asarray(image.convert("RGB")).copy()
            received_latent = ode.encode_image(received_image)
            del stego_latent, sender_delta
            for condition, perm_key, len_key, image_file in trials:
                result = evaluate_key(
                    ode, received_latent, received_image, mask, mask_hr,
                    args.private_prompt, args.public_prompt, len_key, perm_key, condition, image_file,
                    secret, sender_noise, secret_latent, output_dir,
                )
                report["key_recovery_results"].append(result)
                save_results(output_dir, report, ode.trace)
            report["status"] = "completed" if all(
                result["status"] == "completed" for result in report["key_recovery_results"]
            ) else "completed_with_failed_trials"
    finally:
        synchronize_device()
        report["elapsed_seconds_excluding_model_loading"] = time.perf_counter() - started
        report["unet_forward_calls"] = ode.unet_calls
        report["inversion_steps"] = len(ode.trace)
        report["inversion_steps_not_converged"] = sum(not row["converged"] for row in ode.trace)
        save_results(output_dir, report, ode.trace)
    print("\nKey recovery comparison")
    print("%-18s %17s %19s %18s" % (
        "Condition", "Global PSNR (dB)", "Editable PSNR (dB)", "Unconverged steps"))
    for result in report["key_recovery_results"]:
        if result["status"] == "completed":
            print("%-18s %17.4f %19.4f %18d" % (
                result["condition"], result["global_psnr_db"], result["editable_psnr_db"],
                result["inverse_not_converged_steps"],
            ))
        else:
            print("%-18s %17s %19s %18s" % (result["condition"], "failed", "failed", "N/A"))
    print("Inversion steps not converged: %d/%d" % (
        report["inversion_steps_not_converged"], report["inversion_steps"]))
    print("Output directory: " + str(output_dir))
    print("Saved recovery images and key_recovery_results.csv, report.json, inversion_trace.csv")


if __name__ == "__main__":
    main()

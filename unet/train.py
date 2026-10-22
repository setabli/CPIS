import os

from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = Config.GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hashlib
import json
import random
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

from model import CorrectionUNet


def setup():
    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.set_num_threads(Config.CPU_THREADS)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    if Config.SIZE < 8 or Config.SIZE % 8 != 0:
        raise ValueError("SIZE must be at least 8 and divisible by 8.")

    if Config.ITERATIONS < 1:
        raise ValueError("ITERATIONS must be positive.")

    if Config.TRAIN_NEGATIVE_KEYS < 1:
        raise ValueError("TRAIN_NEGATIVE_KEYS must be positive.")

    if Config.PRINT_EVERY < 1 or Config.SAVE_EVERY < 1:
        raise ValueError("PRINT_EVERY and SAVE_EVERY must be positive.")

    if len(bytes.fromhex(Config.CORRECT_KEY)) != 16:
        raise ValueError("CORRECT_KEY must contain 32 hexadecimal characters.")

    use_cuda = torch.cuda.is_available() and not Config.FORCE_CPU
    return torch.device("cuda:0" if use_cuda else "cpu")


def seed_for(tag):
    text = "{}|{}".format(Config.SEED, tag)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def random_tensor(tag, shape):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_for(tag))
    return torch.randn(
        shape,
        generator=generator,
        dtype=torch.float32,
    )


def key_for(tag):
    text = "{}|key|{}".format(Config.SEED, tag)
    return hashlib.sha256(text.encode("utf-8")).digest()[:16].hex()


def delta_from_key(key, device):
    key_bytes = bytes.fromhex(key)
    digest = hashlib.sha256(b"unet-demo-delta|" + key_bytes).digest()
    seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    delta = torch.randn(
        (1, Config.CHANNELS, Config.SIZE, Config.SIZE),
        generator=generator,
        dtype=torch.float32,
    )

    return (Config.DELTA_STD * delta).to(device)


def make_mask(device):
    mask = torch.zeros(
        (1, 1, Config.SIZE, Config.SIZE),
        dtype=torch.float32,
        device=device,
    )
    margin = Config.SIZE // 8
    mask[
        :,
        :,
        margin:Config.SIZE - margin,
        margin:Config.SIZE - margin,
    ] = 1.0
    return mask


def make_pair_context(device):
    coarse = random_tensor(
        "pair-context-coarse",
        (1, Config.CHANNELS, 8, 8),
    ).to(device)

    smooth = F.interpolate(
        coarse,
        size=(Config.SIZE, Config.SIZE),
        mode="bilinear",
        align_corners=False,
    )

    detail = random_tensor(
        "pair-context-detail",
        (1, Config.CHANNELS, Config.SIZE, Config.SIZE),
    ).to(device)

    return 0.6 * smooth + 0.04 * detail


def simulated_displacement(delta, pair_context):
    local_response = F.avg_pool2d(
        delta,
        kernel_size=5,
        stride=1,
        padding=2,
    )

    return (
        1.6 * delta
        + 0.35 * local_response
        + 0.12 * torch.sin(4.0 * delta)
        + 0.08 * torch.tanh(pair_context)
    )


def masked_mse(prediction, target, mask):
    expanded_mask = mask.expand(
        prediction.shape[0],
        1,
        prediction.shape[2],
        prediction.shape[3],
    )
    squared_error = ((prediction - target) * expanded_mask).square()
    count = expanded_mask.sum() * prediction.shape[1]

    return squared_error.sum() / (count + Config.NUMERICAL_EPS)


def config_dict():
    result = {}

    for name in dir(Config):
        if name.isupper():
            value = getattr(Config, name)
            result[name] = str(value) if isinstance(value, Path) else value

    return result


def start_tensorboard(log_dir):
    if not Config.START_TENSORBOARD:
        return None

    try:
        server = program.TensorBoard()
        server.configure(
            argv=[
                "tensorboard",
                "--logdir",
                str(log_dir),
                "--host",
                Config.TENSORBOARD_HOST,
                "--port",
                str(Config.TENSORBOARD_PORT),
            ]
        )
        url = server.launch()
        print("TensorBoard URL:", url)

        if Config.OPEN_BROWSER:
            webbrowser.open(url)

        return server

    except Exception as exc:
        print("TensorBoard could not be started:", exc)
        print("Training logs will still be saved to:", log_dir)
        return None


def save_checkpoint(path, model, optimizer, iteration, losses):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": iteration,
            "losses": losses,
            "pair_id": Config.PAIR_ID,
            "channels": Config.CHANNELS,
            "size": Config.SIZE,
            "widths": list(Config.WIDTHS),
            "groups": Config.GROUPS,
            "seed": Config.SEED,
            "learning_rate": Config.LEARNING_RATE,
        },
        path,
    )


def main():
    device = setup()

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Config.OUTPUT_DIR / run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "tensorboard"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    settings = config_dict()

    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2, ensure_ascii=False)

    tensorboard_server = start_tensorboard(Config.OUTPUT_DIR)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=5)
    writer.add_text("Configuration", json.dumps(settings, indent=2), 0)

    try:
        mask = make_mask(device)
        pair_context = make_pair_context(device)

        correct_delta = delta_from_key(Config.CORRECT_KEY, device)
        correct_target = simulated_displacement(
            correct_delta,
            pair_context,
        )

        negative_keys = [
            key_for("train-negative-{}".format(i))
            for i in range(Config.TRAIN_NEGATIVE_KEYS)
        ]

        if Config.CORRECT_KEY in negative_keys:
            raise ValueError("A training negative key matches the correct key.")

        negative_inputs = torch.cat(
            [delta_from_key(key, device) for key in negative_keys],
            dim=0,
        )
        negative_targets = simulated_displacement(
            negative_inputs,
            pair_context,
        )

        model = CorrectionUNet().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print("=" * 72)
        print("Single-pair synthetic latent compensation training")
        print("Device:", device)
        print("Trainable parameters:", parameter_count)
        print("Learning rate:", Config.LEARNING_RATE)
        print("Training iterations:", Config.ITERATIONS)
        print("Training negative keys:", len(negative_keys))
        print("Output directory:", run_dir)
        print("TensorBoard log directory:", log_dir)
        print("=" * 72)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        started = time.perf_counter()
        final_losses = {}

        model.train()

        for step in range(1, Config.ITERATIONS + 1):
            index = (step - 1) % len(negative_keys)

            batch_delta = torch.cat(
                (
                    correct_delta,
                    negative_inputs[index:index + 1],
                ),
                dim=0,
            )

            batch_target = torch.cat(
                (
                    correct_target,
                    negative_targets[index:index + 1],
                ),
                dim=0,
            )

            optimizer.zero_grad(set_to_none=True)

            prediction = model(batch_delta)

            genuine_loss = masked_mse(
                prediction[:1],
                batch_target[:1],
                mask,
            )

            negative_loss = masked_mse(
                prediction[1:],
                batch_target[1:],
                mask,
            )

            loss = 0.5 * (genuine_loss + negative_loss)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss contains NaN or Inf.")

            loss.backward()
            optimizer.step()

            final_losses = {
                "total": loss.detach().item(),
                "genuine": genuine_loss.detach().item(),
                "negative": negative_loss.detach().item(),
            }

            writer.add_scalar("Loss/Total", final_losses["total"], step)
            writer.add_scalar("Loss/Genuine", final_losses["genuine"], step)
            writer.add_scalar("Loss/Negative", final_losses["negative"], step)
            writer.add_scalar(
                "Training/LearningRate",
                optimizer.param_groups[0]["lr"],
                step,
            )

            if (
                step == 1
                or step % Config.PRINT_EVERY == 0
                or step == Config.ITERATIONS
            ):
                elapsed = time.perf_counter() - started

                print(
                    "Iteration {}/{} | Loss: {:.8e} | "
                    "Genuine: {:.8e} | Negative: {:.8e} | "
                    "Elapsed: {:.2f}s".format(
                        step,
                        Config.ITERATIONS,
                        final_losses["total"],
                        final_losses["genuine"],
                        final_losses["negative"],
                        elapsed,
                    ),
                    flush=True,
                )

                writer.flush()

            if step % Config.SAVE_EVERY == 0:
                checkpoint_path = checkpoint_dir / (
                    "step_{:06d}.pt".format(step)
                )

                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step,
                    final_losses,
                )

                print("Checkpoint saved:", checkpoint_path)

        final_path = checkpoint_dir / "final_model.pt"

        save_checkpoint(
            final_path,
            model,
            optimizer,
            Config.ITERATIONS,
            final_losses,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        print("=" * 72)
        print("Training completed.")
        print("Final model:", final_path)
        print("Total training time: {:.2f}s".format(
            time.perf_counter() - started
        ))
        print("TensorBoard logs:", log_dir)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
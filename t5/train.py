import os

from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = Config.GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from transformers import DataCollatorForSeq2Seq

from model import build_model


def setup():
    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    if Config.EPOCHS < 1:
        raise ValueError("EPOCHS must be positive.")

    if Config.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be positive.")

    if Config.PRINT_EVERY < 1:
        raise ValueError("PRINT_EVERY must be positive.")

    use_cuda = torch.cuda.is_available() and not Config.FORCE_CPU
    return torch.device("cuda:0" if use_cuda else "cpu")


def normalize_text(value):
    if not isinstance(value, str):
        raise TypeError("Semantic field values must be strings.")

    return " ".join(value.lower().split())


def normalize_semantics(value):
    if not isinstance(value, dict):
        raise TypeError("Semantics must be a JSON object.")

    fields = (
        "subject",
        "action",
        "location",
        "relation",
    )

    if set(value) != set(fields) | {"environment"}:
        raise ValueError(
            "Semantics must contain exactly: "
            "subject, action, location, relation, environment."
        )

    normalized = {
        name: normalize_text(value[name])
        for name in fields
    }

    environment = value["environment"]

    if not isinstance(environment, list):
        raise TypeError("Environment must be a list.")

    elements = []

    for element in environment:
        if not isinstance(element, dict):
            raise TypeError(
                "Each environmental element must be a JSON object."
            )

        if set(element) != set(fields):
            raise ValueError(
                "Each environmental element must contain exactly: "
                "subject, action, location, relation."
            )

        elements.append(
            {
                name: normalize_text(element[name])
                for name in fields
            }
        )

    elements.sort(
        key=lambda element: tuple(
            element[name] for name in fields
        )
    )

    normalized["environment"] = elements
    return normalized


def serialize_semantics(value):
    return json.dumps(
        normalize_semantics(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


class SemanticDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                "Training dataset not found: {}".format(file_path)
            )

        with file_path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)

        if not isinstance(records, list) or not records:
            raise ValueError(
                "Training data must be a non-empty JSON list."
            )

        self.group_count = len(records)
        self.samples = []
        seen_targets = {}

        for group_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(
                    "Each dataset record must be a JSON object."
                )

            prompt = record["prompt"]
            variants = record.get("variants", [])

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    "Each record must contain a non-empty prompt."
                )

            if not isinstance(variants, list):
                raise TypeError("Variants must be a list.")

            target_text = serialize_semantics(
                record["semantics"]
            )

            target_encoding = tokenizer(
                text_target=target_text,
                add_special_tokens=True,
                truncation=False,
            )

            target_ids = target_encoding["input_ids"]

            if len(target_ids) > Config.MAX_TARGET_LENGTH:
                raise ValueError(
                    "Target exceeds MAX_TARGET_LENGTH in group {}.".format(
                        group_index
                    )
                )

            if tokenizer.unk_token_id in target_ids:
                raise ValueError(
                    "Target contains unknown tokens in group {}.".format(
                        group_index
                    )
                )

            for expression in [prompt] + variants:
                if (
                    not isinstance(expression, str)
                    or not expression.strip()
                ):
                    raise ValueError(
                        "Prompt expressions must be non-empty strings."
                    )

                expression = " ".join(expression.split())

                if expression in seen_targets:
                    if seen_targets[expression] != target_text:
                        raise ValueError(
                            "Conflicting annotations for expression: "
                            + expression
                        )

                    continue

                seen_targets[expression] = target_text

                input_encoding = tokenizer(
                    Config.INPUT_PREFIX + expression,
                    add_special_tokens=True,
                    truncation=False,
                )

                if (
                    len(input_encoding["input_ids"])
                    > Config.MAX_INPUT_LENGTH
                ):
                    raise ValueError(
                        "Input exceeds MAX_INPUT_LENGTH in group {}.".format(
                            group_index
                        )
                    )

                self.samples.append(
                    {
                        "input_ids": input_encoding["input_ids"],
                        "attention_mask": input_encoding["attention_mask"],
                        "labels": list(target_ids),
                    }
                )

        if not self.samples:
            raise ValueError("No training samples were loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {
            name: list(value)
            for name, value in self.samples[index].items()
        }


def config_dict():
    result = {}

    for name in dir(Config):
        if name.isupper():
            value = getattr(Config, name)
            result[name] = (
                str(value)
                if isinstance(value, Path)
                else value
            )

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


def save_model(
    output_dir,
    model,
    tokenizer,
    optimizer,
    epoch,
    global_step,
    epoch_loss,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(
        {
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "epoch_loss": epoch_loss,
        },
        output_dir / "training_state.pt",
    )

    with (output_dir / "training_config.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            config_dict(),
            stream,
            indent=2,
            ensure_ascii=False,
        )


def main():
    device = setup()

    run_name = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_dir = Config.OUTPUT_DIR / run_name
    log_dir = run_dir / "tensorboard"

    log_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = build_model()

    dataset = SemanticDataset(
        Config.TRAIN_FILE,
        tokenizer,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    generator = torch.Generator()
    generator.manual_seed(Config.SEED)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    tensorboard_server = start_tensorboard(
        Config.OUTPUT_DIR
    )

    writer = SummaryWriter(
        log_dir=str(log_dir),
        flush_secs=5,
    )

    writer.add_text(
        "Configuration",
        json.dumps(config_dict(), indent=2),
        0,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 72)
    print("T5 semantic parser training")
    print("Device:", device)
    print("Pretrained model:", Config.MODEL_PATH)
    print("Semantic groups:", dataset.group_count)
    print("Training expressions:", len(dataset))
    print("Trainable parameters:", parameter_count)
    print("Epochs:", Config.EPOCHS)
    print("Batch size:", Config.BATCH_SIZE)
    print("Learning rate:", Config.LEARNING_RATE)
    print("Output directory:", run_dir)
    print("TensorBoard log directory:", log_dir)
    print("=" * 72)

    global_step = 0
    started = time.perf_counter()
    epoch_loss = None

    try:
        for epoch in range(1, Config.EPOCHS + 1):
            model.train()

            total_loss = 0.0
            total_tokens = 0

            for batch in loader:
                batch = {
                    name: value.to(
                        device,
                        non_blocking=device.type == "cuda",
                    )
                    for name, value in batch.items()
                }

                optimizer.zero_grad(set_to_none=True)

                outputs = model(**batch)
                loss = outputs.loss

                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        "Training loss contains NaN or Inf."
                    )

                loss.backward()

                gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        Config.MAX_GRAD_NORM,
                        error_if_nonfinite=True,
                    )
                )

                optimizer.step()

                valid_tokens = (
                    batch["labels"] != -100
                ).sum().item()

                loss_value = loss.detach().item()

                total_loss += loss_value * valid_tokens
                total_tokens += valid_tokens
                global_step += 1

                writer.add_scalar(
                    "Loss/Step",
                    loss_value,
                    global_step,
                )

                writer.add_scalar(
                    "Training/GradientNorm",
                    gradient_norm.detach().item(),
                    global_step,
                )

                writer.add_scalar(
                    "Training/LearningRate",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )

            epoch_loss = total_loss / total_tokens

            writer.add_scalar(
                "Loss/Epoch",
                epoch_loss,
                epoch,
            )

            if (
                epoch == 1
                or epoch % Config.PRINT_EVERY == 0
                or epoch == Config.EPOCHS
            ):
                print(
                    "Epoch {}/{} | Loss: {:.8f} | "
                    "Steps: {} | Elapsed: {:.2f}s".format(
                        epoch,
                        Config.EPOCHS,
                        epoch_loss,
                        global_step,
                        time.perf_counter() - started,
                    ),
                    flush=True,
                )

                writer.flush()

            if (
                Config.SAVE_EVERY > 0
                and epoch % Config.SAVE_EVERY == 0
                and epoch != Config.EPOCHS
            ):
                checkpoint_dir = run_dir / (
                    "epoch_{:04d}".format(epoch)
                )

                save_model(
                    checkpoint_dir,
                    model,
                    tokenizer,
                    optimizer,
                    epoch,
                    global_step,
                    epoch_loss,
                )

                print(
                    "Checkpoint saved:",
                    checkpoint_dir,
                )

        final_dir = run_dir / "final_model"

        save_model(
            final_dir,
            model,
            tokenizer,
            optimizer,
            Config.EPOCHS,
            global_step,
            epoch_loss,
        )

        print("=" * 72)
        print("Training completed.")
        print(
            "Final training loss: {:.8f}".format(
                epoch_loss
            )
        )
        print("Final model:", final_dir)
        print("TensorBoard logs:", log_dir)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
# CPIS

Controllable Partial Image Steganography.

## Project Structure

| Path | Purpose |
| --- | --- |
| `demo.py` | Image hiding and recovery with correct keys |
| `sd2/` | Stable Diffusion model files |
| `unet/` | Correction network, training script, and checkpoints |
| `seg/` | GroundingDINO and SAM region selection |
| `inblip/` | InstructBLIP model, description script, and query templates |
| `qwen/` | Qwen model, semantic aggregation script, and instruction template |
| `t5/` | T5 semantic-parser training code and data |

Place pretrained model files in their corresponding directories as shown above in "Project Structure". Set the T5 model source in `t5/config.py` and GPU settings in the relevant configuration files.

## Experimentation

`demo.py` provides the core steps of image hiding and recovery. To run your own experiment, modify the arguments in the parser (for example, `image_path`, `mask_path`, `private_key`, `public_key`, `unet_checkpoint`, `len_key`, `shuffle_step`, and `inject_step`) and replace the model loading in `demo.py` with your own trained model/checkpoint.

In the demo mask, white pixels are editable and black pixels are preserved. `shuffle_step` and `inject_step` control channel shuffling and LEN injection during public denoising.

## Other Scripts

Each module can be verified in its corresponding directory, where example data are provided for reference. After configuring their inputs, run `seg/run_tgrs.py`, `inblip/inblip_demo.py`, or `qwen/qwen_demo.py` separately. Qwen reads `inblip_output/result.json` by default. These modules are not automatically invoked by `demo.py`.

For training, configure and run `t5/train.py` or `unet/train.py`. The provided UNet training script uses synthetic correction targets; actual image recovery requires pair-specific weights trained with real diffusion correction targets and matching settings.
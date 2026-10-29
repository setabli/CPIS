# CPIS

Controllable Partial Image Steganography.

## Project Structure

| Path | Purpose |
| --- | --- |
| `demo.py` | Image hiding and recovery with correct keys |
| `verify_lre.py` | Standalone verification of latent representation encryption |
| `sd2/` | Stable Diffusion model files |
| `seg/` | Rule-guided region selection using Qwen2.5-VL and SAM |
| `qwen/` | Visual-description and semantic-aggregation scripts, prompt templates, and local Qwen model files |
| `t5/` | T5 semantic-parser training code, data, and checkpoints |

Place pretrained model files in their corresponding directories and configure their paths in the relevant scripts. Visual description and region localization require Qwen2.5-VL, while semantic aggregation uses the text-only Qwen model. Set the T5 model source and training options in `t5/config.py`.

## Experimentation

`demo.py` provides the core image hiding and recovery process using mask-guided diffusion, key-controlled latent transformations, and iterative numerical inversion.

Before running the script, modify the parser defaults for `model_path`, `image_path`, `mask_path`, `private_prompt`, `public_prompt`, and GPU settings. Configure the independent LEP and LEN keys through `perm_key` and `len_key`, or use the local key file specified by `key_file`. Explicit keys must each contain 32 hexadecimal characters, representing 128 bits.

White mask pixels are editable and black pixels are preserved. `perm_step` controls position-wise channel permutation, and `len_step` controls noise-perturbation injection during public-branch denoising. Both are counted from the start of the descending denoising sequence. Use matching keys and diffusion settings for hiding and recovery.

## Other Scripts

The modules can be run separately after configuring their model paths and inputs:

- `seg/`: locate a region using the shared selection policy and generate masks with SAM.
- `qwen/`: extract masked-region and contextual visual descriptions with Qwen2.5-VL, then aggregate them with Qwen. Set the aggregation input to the visual-description script's output `result.json`.
- `t5/train.py`: train the T5 semantic parser using the configuration in `t5/config.py`.
- `verify_lre.py`: run standalone LRE verification.

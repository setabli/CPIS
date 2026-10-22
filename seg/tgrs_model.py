import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import box_convert, nms

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from segment_anything import SamPredictor, sam_model_registry

from tgrs_config import TGRSConfig


class TGRS:
    def __init__(self, device):
        self.device = torch.device(device)
        self.check_paths()

        args = SLConfig.fromfile(
            str(TGRSConfig.DINO_CONFIG)
        )

        args.device = str(self.device)
        args.text_encoder_type = str(
            TGRSConfig.BERT_MODEL_PATH
        )
        args.use_checkpoint = False
        args.use_transformer_ckpt = False

        self.detector = build_model(args)

        checkpoint = torch.load(
            str(TGRSConfig.DINO_CHECKPOINT),
            map_location="cpu",
            weights_only=False,
        )

        state = clean_state_dict(checkpoint["model"])

        load_result = self.detector.load_state_dict(
            state,
            strict=False,
        )

        print(
            "GroundingDINO missing keys:",
            len(load_result.missing_keys),
        )
        print(
            "GroundingDINO unexpected keys:",
            len(load_result.unexpected_keys),
        )

        self.detector.to(self.device)
        self.detector.eval()
        self.detector.requires_grad_(False)

        sam = sam_model_registry[
            TGRSConfig.SAM_MODEL_TYPE
        ](
            checkpoint=str(TGRSConfig.SAM_CHECKPOINT)
        )

        sam.to(self.device)
        sam.eval()
        sam.requires_grad_(False)

        self.predictor = SamPredictor(sam)

        self.transform = T.Compose(
            [
                T.RandomResize(
                    [800],
                    max_size=1333,
                ),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def check_paths(self):
        for path in (
            TGRSConfig.DINO_CONFIG,
            TGRSConfig.DINO_CHECKPOINT,
            TGRSConfig.SAM_CHECKPOINT,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    "Model file not found: {}".format(path)
                )

        if not TGRSConfig.BERT_MODEL_PATH.is_dir():
            raise FileNotFoundError(
                "Local BERT directory not found: {}".format(
                    TGRSConfig.BERT_MODEL_PATH
                )
            )

        if (
            TGRSConfig.SAM_MODEL_TYPE
            not in sam_model_registry
        ):
            raise ValueError("Unsupported SAM model type.")

    @torch.inference_mode()
    def select(self, image, target_text, region_mode):
        if region_mode not in ("edit", "preserve"):
            raise ValueError(
                "REGION_MODE must be 'edit' or 'preserve'."
            )

        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError("TARGET_TEXT must be non-empty.")

        image = image.convert("RGB")
        image_array = np.asarray(image).copy()
        width, height = image.size

        transformed_image, _ = self.transform(
            image,
            None,
        )

        boxes, scores, phrases = predict(
            model=self.detector,
            image=transformed_image,
            caption=target_text,
            box_threshold=TGRSConfig.BOX_THRESHOLD,
            text_threshold=TGRSConfig.TEXT_THRESHOLD,
            device=str(self.device),
        )

        if boxes.shape[0] == 0:
            raise RuntimeError(
                "No target was detected. Check TARGET_TEXT "
                "or the detection thresholds."
            )

        boxes = boxes.detach().cpu()
        scores = scores.detach().cpu()

        scale = torch.tensor(
            [width, height, width, height],
            dtype=boxes.dtype,
        )

        boxes = box_convert(
            boxes * scale,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)

        valid = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )

        valid_indices = torch.where(valid)[0]

        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        phrases = [
            phrases[index]
            for index in valid_indices.tolist()
        ]

        if boxes.shape[0] == 0:
            raise RuntimeError(
                "No valid detection boxes remain."
            )

        keep = nms(
            boxes,
            scores,
            TGRSConfig.NMS_THRESHOLD,
        )

        boxes = boxes[keep]
        scores = scores[keep]
        phrases = [
            phrases[index]
            for index in keep.tolist()
        ]

        self.predictor.set_image(image_array)

        transformed_boxes = (
            self.predictor.transform.apply_boxes_torch(
                boxes.to(self.device),
                image_array.shape[:2],
            )
        )

        target_probability = torch.zeros(
            (height, width),
            dtype=torch.float32,
            device=self.device,
        )

        quality_scores = []

        for start in range(
            0,
            len(transformed_boxes),
            TGRSConfig.SAM_BATCH_SIZE,
        ):
            batch_boxes = transformed_boxes[
                start:start + TGRSConfig.SAM_BATCH_SIZE
            ]

            mask_logits, predicted_quality, _ = (
                self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=batch_boxes,
                    multimask_output=False,
                    return_logits=True,
                )
            )

            batch_probability = (
                mask_logits[:, 0]
                .sigmoid()
                .amax(dim=0)
            )

            target_probability = torch.maximum(
                target_probability,
                batch_probability,
            )

            quality_scores.extend(
                predicted_quality[:, 0]
                .detach()
                .cpu()
                .tolist()
            )

        if region_mode == "edit":
            editable_probability = target_probability
        else:
            editable_probability = 1.0 - target_probability

        mask_hr = (
            editable_probability
            >= TGRSConfig.MASK_THRESHOLD
        ).float()

        if mask_hr.sum().item() == 0:
            raise RuntimeError(
                "The editable mask is empty."
            )

        mask_latent = F.interpolate(
            mask_hr[None, None],
            size=TGRSConfig.LATENT_SIZE,
            mode="nearest",
        )

        if mask_latent.sum().item() == 0:
            raise RuntimeError(
                "The editable region disappeared after "
                "latent-mask downsampling."
            )

        return {
            "image": image_array,
            "mask_hr": mask_hr.cpu().numpy(),
            "mask_latent": mask_latent.cpu().numpy(),
            "confidence": editable_probability.cpu().numpy(),
            "boxes": boxes.numpy(),
            "scores": scores.numpy(),
            "phrases": phrases,
            "sam_quality_estimates": quality_scores,
        }
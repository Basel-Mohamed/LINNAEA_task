"""
extractor.py
============
Spatial-aware OCR pipeline for handwritten medical prescriptions.

Pipeline:
    Image → YOLO (region detection + bounding boxes) → TrOCR (text recognition)
    → SpatialToken list  (text + x, y, w, h for every detected region)

Each detected region is returned as a SpatialToken so downstream
tasks (spatial hallucination detection, graph building) always know
WHERE on the original scan a piece of text came from.
"""

import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image, ImageOps

from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Data contract – every OCR result carries its spatial origin
# ---------------------------------------------------------------------------

@dataclass
class SpatialToken:
    """
    A single detected text region with its position on the original scan.

    Attributes
    ----------
    text        : recognized text from TrOCR
    x_min       : left edge   (pixels, in 640×640 resized space)
    y_min       : top edge
    x_max       : right edge
    y_max       : bottom edge
    x_center    : normalized center-x  (0-1)  from YOLO
    y_center    : normalized center-y  (0-1)  from YOLO
    width_norm  : normalized width     (0-1)  from YOLO
    height_norm : normalized height    (0-1)  from YOLO
    confidence  : YOLO detection confidence score
    region_id   : reading-order index (sorted top-to-bottom)
    source_path : path to the original image
    """
    text:         str
    x_min:        int
    y_min:        int
    x_max:        int
    y_max:        int
    x_center:     float
    y_center:     float
    width_norm:   float
    height_norm:  float
    confidence:   float
    region_id:    int
    source_path:  str = ""

    # ---- helpers -----------------------------------------------------------

    @property
    def bbox(self) -> dict:
        """Return bounding box as a plain dict (handy for serialisation)."""
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    @property
    def zone(self) -> str:
        """
        Classify vertical position into document zones.
        Useful for spatial hallucination detection (Task 1).

        Zones (in a 640-px tall image):
            header  : top    20 %   (y_center < 0.20)
            body    : middle 60 %   (0.20 ≤ y_center < 0.80)
            footer  : bottom 20 %   (y_center ≥ 0.80)
        """
        if self.y_center < 0.20:
            return "header"
        elif self.y_center < 0.80:
            return "body"
        else:
            return "footer"

    def to_dict(self) -> dict:
        """Full serialisable representation."""
        return {
            "region_id":   self.region_id,
            "text":        self.text,
            "bbox":        self.bbox,
            "x_center":   round(self.x_center, 4),
            "y_center":   round(self.y_center, 4),
            "width_norm":  round(self.width_norm, 4),
            "height_norm": round(self.height_norm, 4),
            "confidence":  round(self.confidence, 4),
            "zone":        self.zone,
            "source_path": self.source_path,
        }


# ---------------------------------------------------------------------------
# OCR Extractor
# ---------------------------------------------------------------------------

class PrescriptionExtractor:
    """
    Spatial-aware OCR for handwritten medical prescriptions.

    Parameters
    ----------
    yolo_repo   : HuggingFace repo that hosts the YOLO weights
    trocr_model : HuggingFace model id for the fine-tuned TrOCR
    trocr_proc  : HuggingFace model id for the TrOCR processor
    hf_token    : HuggingFace token (needed for gated models)
    conf        : YOLO confidence threshold  (default 0.5)
    iou         : YOLO IoU threshold         (default 0.6)
    max_det     : maximum detections per image (default 50)
    img_size    : YOLO inference size in pixels (default 640)
    """

    IMG_SIZE   = 640    # YOLO input resolution
    CROP_SIZE  = 384    # TrOCR input resolution

    def __init__(
        self,
        yolo_repo:   str = "wahdan2003/YOLO_handwritten_medical",
        trocr_model: str = "David-Magdy/TROCR_MASTER_V2",
        trocr_proc:  str = "microsoft/trocr-base-handwritten",
        hf_token:    Optional[str] = None,
        conf:        float = 0.5,
        iou:         float = 0.6,
        max_det:     int   = 50,
    ):
        self.conf    = conf
        self.iou     = iou
        self.max_det = max_det
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Extractor] device  : {self.device}")
        print(f"[Extractor] loading YOLO from {yolo_repo} …")
        model_path   = hf_hub_download(repo_id=yolo_repo, filename="best.pt")
        self.yolo    = YOLO(model_path)

        print(f"[Extractor] loading TrOCR processor from {trocr_proc} …")
        self.processor = TrOCRProcessor.from_pretrained(
            trocr_proc, use_safetensors=True, token=hf_token
        )

        print(f"[Extractor] loading TrOCR model from {trocr_model} …")
        self.trocr = VisionEncoderDecoderModel.from_pretrained(
            trocr_model, use_safetensors=True, token=hf_token
        ).to(self.device)

        print("[Extractor] ready.\n")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(self, image_paths: list[str]) -> list[list[SpatialToken]]:
        """
        Run the full pipeline on a list of image paths.

        Returns
        -------
        List of SpatialToken lists — one inner list per image,
        ordered top-to-bottom (reading order).
        """
        results = []
        for path in image_paths:
            tokens = self._process_single(path)
            results.append(tokens)
        return results

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """Resize to IMG_SIZE × IMG_SIZE with gray letterbox padding."""
        s = self.IMG_SIZE
        ratio  = min(s / image.width, s / image.height)
        nw, nh = int(image.width * ratio), int(image.height * ratio)
        resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
        return ImageOps.pad(resized, (s, s), color=(114, 114, 114))

    def _run_yolo(self, image: Image.Image):
        """Run YOLO detection and return raw results."""
        return self.yolo.predict(
            source      = image,
            imgsz       = self.IMG_SIZE,
            device      = 0 if self.device == "cuda" else "cpu",
            save        = False,
            conf        = self.conf,
            iou         = self.iou,
            max_det     = self.max_det,
            verbose     = False,
        )

    def _crop_region(
        self, image: Image.Image, x_min: int, y_min: int,
        x_max: int, y_max: int
    ) -> Image.Image:
        """Crop a detected region and pad to CROP_SIZE for TrOCR."""
        crop = image.crop((x_min, y_min, x_max, y_max))
        return ImageOps.pad(
            crop, (self.CROP_SIZE, self.CROP_SIZE), color=(255, 255, 255)
        )

    def _run_trocr(self, crop: Image.Image) -> str:
        """Run TrOCR on a single cropped region and return text."""
        pixel_values = self.processor(
            crop, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.trocr.generate(pixel_values)

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def _process_single(self, image_path: str) -> list[SpatialToken]:
        """Full pipeline for a single image → list of SpatialTokens."""
        try:
            image   = Image.open(image_path).convert("RGB")
            resized = self._resize_with_padding(image)
            s       = self.IMG_SIZE

            yolo_results = self._run_yolo(resized)
            boxes        = yolo_results[0].boxes

            if boxes is None or len(boxes) == 0:
                print(f"[Extractor] no regions detected in {image_path}")
                return []

            # xywhn → normalised [x_center, y_center, width, height]
            coords_norm = boxes.xywhn.cpu().numpy()   # shape (N, 4)
            confs       = boxes.conf.cpu().numpy()     # shape (N,)

            # sort top-to-bottom by y_center (reading order)
            order          = coords_norm[:, 1].argsort()
            coords_norm    = coords_norm[order]
            confs          = confs[order]

            tokens: list[SpatialToken] = []

            for idx, (coord, conf_score) in enumerate(zip(coords_norm, confs)):
                xc, yc, wn, hn = coord

                # pixel coordinates in the resized 640×640 space
                x_min = int((xc - wn / 2) * s)
                x_max = int((xc + wn / 2) * s)
                y_min = int((yc - hn / 2) * s)
                y_max = int((yc + hn / 2) * s)

                # clamp to image bounds
                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, s), min(y_max, s)

                crop = self._crop_region(resized, x_min, y_min, x_max, y_max)
                text = self._run_trocr(crop)

                tokens.append(SpatialToken(
                    text        = text,
                    x_min       = x_min,
                    y_min       = y_min,
                    x_max       = x_max,
                    y_max       = y_max,
                    x_center    = float(xc),
                    y_center    = float(yc),
                    width_norm  = float(wn),
                    height_norm = float(hn),
                    confidence  = float(conf_score),
                    region_id   = idx,
                    source_path = image_path,
                ))

            return tokens

        except Exception as exc:
            print(f"[Extractor] ERROR processing {image_path}: {exc}")
            return []
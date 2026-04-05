"""
extractor.py  (v2)
==================
Key improvements over v1:
  1. Line-level box merging  — TrOCR gets full-line crops, not word fragments.
  2. Zone-aware preprocessing — letterhead zone uses a contrast-boosted crop;
     handwritten body uses the original cleaned image.
  3. Adaptive crop padding   — very thin boxes get extra top/bottom padding.
  4. Source coords stored    — token x/y always refer to the ORIGINAL image,
     ready for VLM cropping without any extra scaling.
"""

import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from huggingface_hub import hf_hub_download

from src.ocr.vision_utils import ClinicalImageEnhancer


# ─────────────────────────────────────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpatialToken:
    text:         str
    x_min:        int
    y_min:        int
    x_max:        int
    y_max:        int
    x_center:     float   # normalised to original image dims
    y_center:     float
    width_norm:   float
    height_norm:  float
    confidence:   float
    region_id:    int
    source_path:  str = ""
    zone:         str = "body"

    @property
    def bbox(self) -> dict:
        return dict(x_min=self.x_min, y_min=self.y_min,
                    x_max=self.x_max, y_max=self.y_max)

    def to_dict(self) -> dict:
        return {
            "region_id":   self.region_id,
            "text":        self.text,
            "bbox":        self.bbox,
            "x_center":    round(self.x_center,   4),
            "y_center":    round(self.y_center,   4),
            "width_norm":  round(self.width_norm,  4),
            "height_norm": round(self.height_norm, 4),
            "confidence":  round(self.confidence,  4),
            "zone":        self.zone,
            "source_path": self.source_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Line merger  —  FIX #1: give TrOCR full lines, not word fragments
# ─────────────────────────────────────────────────────────────────────────────

class LineMerger:
    """
    Clusters YOLO word-boxes into line-level groups.

    Two boxes belong to the same line when their vertical centres differ by
    less than `y_tol` × the average box height.  Within a line, boxes are
    merged left-to-right and their bounding union is used as the crop.
    """

    def __init__(self, y_tol: float = 0.55, x_gap_tol: float = 0.08):
        self.y_tol     = y_tol      # fraction of avg height
        self.x_gap_tol = x_gap_tol  # max horizontal gap (normalised 0-1)

    def merge(self, boxes_norm: np.ndarray, confs: np.ndarray,
              img_w: int, img_h: int):
        """
        Parameters
        ----------
        boxes_norm : (N, 4) xywhn in [0, 1]  (from YOLO)
        confs      : (N,)   detection scores
        img_w/h    : original image pixel dimensions

        Returns
        -------
        List of dicts, each with keys:
            x_min, y_min, x_max, y_max  (pixel coords in orig image)
            conf                         (mean confidence)
        """
        if len(boxes_norm) == 0:
            return []

        # Convert to pixel absolute coords
        abs_boxes = []
        for (xc, yc, w, h), c in zip(boxes_norm, confs):
            abs_boxes.append({
                "x_min": (xc - w / 2) * img_w,
                "x_max": (xc + w / 2) * img_w,
                "y_min": (yc - h / 2) * img_h,
                "y_max": (yc + h / 2) * img_h,
                "yc":    yc * img_h,
                "h":     h  * img_h,
                "conf":  float(c),
            })

        # Sort top-to-bottom then left-to-right
        abs_boxes.sort(key=lambda b: (b["yc"], b["x_min"]))

        lines = []
        for box in abs_boxes:
            placed = False
            for line in lines:
                avg_h   = np.mean([b["h"] for b in line])
                line_yc = np.mean([b["yc"] for b in line])
                if abs(box["yc"] - line_yc) < self.y_tol * avg_h:
                    # Check horizontal gap — don't merge across large gaps
                    rightmost = max(b["x_max"] for b in line)
                    gap = (box["x_min"] - rightmost) / img_w
                    if gap < self.x_gap_tol:
                        line.append(box)
                        placed = True
                        break
            if not placed:
                lines.append([box])

        merged = []
        for line in lines:
            merged.append({
                "x_min": int(max(0,     min(b["x_min"] for b in line) - 4)),
                "y_min": int(max(0,     min(b["y_min"] for b in line) - 4)),
                "x_max": int(min(img_w, max(b["x_max"] for b in line) + 4)),
                "y_max": int(min(img_h, max(b["y_max"] for b in line) + 4)),
                "conf":  float(np.mean([b["conf"] for b in line])),
            })

        # Sort final lines top-to-bottom
        merged.sort(key=lambda b: b["y_min"])
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────────────────────

class PrescriptionExtractor:
    IMG_SIZE  = 640   # YOLO input resolution
    CROP_W    = 768   # TrOCR input width  (wider = better for full lines)
    CROP_H    = 128   # TrOCR input height

    def __init__(
        self,
        yolo_model:  str   = "RoyRud1902/yolo11n-text",
        trocr_model: str   = "microsoft/trocr-large-handwritten",
        trocr_proc:  str   = "microsoft/trocr-large-handwritten",
        hf_token:    str   = None,
        conf:        float = 0.45,   # slightly lower — we merge anyway
        iou:         float = 0.5,
        max_det:     int   = 100,
        y_tol:       float = 0.55,   # LineMerger sensitivity
    ):
        self.conf      = conf
        self.iou       = iou
        self.max_det   = max_det
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.merger    = LineMerger(y_tol=y_tol)

        print(f"[Extractor] device  : {self.device}")
        print(f"[Extractor] loading YOLO …")
        if "/" in yolo_model and not yolo_model.endswith(".pt"):
            path = hf_hub_download(repo_id=yolo_model,
                                   filename="best.pt", token=hf_token)
            self.yolo = YOLO(path)
        else:
            self.yolo = YOLO(yolo_model)

        print(f"[Extractor] loading TrOCR …")
        self.processor = TrOCRProcessor.from_pretrained(
            trocr_proc, use_safetensors=True, token=hf_token
        )
        self.trocr = VisionEncoderDecoderModel.from_pretrained(
            trocr_model, use_safetensors=True, token=hf_token
        ).to(self.device)
        print("[Extractor] ready.\n")

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self, image_paths: list[str]) -> list[list[SpatialToken]]:
        return [self._process_single(p) for p in image_paths]

    # ── internal ──────────────────────────────────────────────────────────────

    def _run_yolo(self, pil_img: Image.Image):
        resized = self._letterbox(pil_img)
        results = self.yolo.predict(
            source=resized, imgsz=self.IMG_SIZE,
            device=0 if self.device == "cuda" else "cpu",
            save=False, conf=self.conf, iou=self.iou,
            max_det=self.max_det, verbose=False,
        )
        return results, resized

    @staticmethod
    def _letterbox(img: Image.Image, size: int = 640) -> Image.Image:
        ratio  = min(size / img.width, size / img.height)
        nw, nh = int(img.width * ratio), int(img.height * ratio)
        return ImageOps.pad(img.resize((nw, nh), Image.Resampling.LANCZOS),
                            (size, size), color=(114, 114, 114))

    def _preprocess_crop(self, crop: Image.Image, is_printed: bool) -> Image.Image:
        """
        Zone-aware enhancement before TrOCR.
        Printed letterhead → contrast boost + sharpen.
        Handwritten body   → mild sharpen only.
        """
        if is_printed:
            crop = ImageEnhance.Contrast(crop).enhance(2.0)
            crop = ImageEnhance.Sharpness(crop).enhance(2.0)
        else:
            crop = crop.filter(ImageFilter.SHARPEN)
        return crop

    def _run_trocr(self, crop: Image.Image) -> str:
        # Pad to fixed rectangular shape that TrOCR likes
        crop_padded = ImageOps.pad(
            crop, (self.CROP_W, self.CROP_H), color=(255, 255, 255)
        )
        pv = self.processor(
            crop_padded, return_tensors="pt"
        ).pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.trocr.generate(pv, max_new_tokens=128)
        return self.processor.batch_decode(
            ids, skip_special_tokens=True
        )[0].strip()

    def _process_single(self, image_path: str) -> list[SpatialToken]:
        try:
            raw   = Image.open(image_path).convert("RGB")
            clean = ClinicalImageEnhancer.remove_stamps_from_image(raw)
            w_orig, h_orig = clean.size

            # ── YOLO on 640×640 letterboxed image ────────────────────────────
            results, _ = self._run_yolo(clean)
            boxes_obj  = results[0].boxes
            if boxes_obj is None or len(boxes_obj) == 0:
                print(f"[Extractor] no regions in {image_path}")
                return []

            coords_norm = boxes_obj.xywhn.cpu().numpy()
            confs       = boxes_obj.conf.cpu().numpy()

            # ── FIX #1: merge word-boxes → line-boxes ─────────────────────────
            # YOLO normalised coords are in 640-space; convert to orig-image space
            s     = self.IMG_SIZE
            ratio = min(s / w_orig, s / h_orig)
            pad_l = (s - int(w_orig * ratio)) // 2
            pad_t = (s - int(h_orig * ratio)) // 2

            def to_orig(xc_n, yc_n, wn, hn):
                """Map normalised 640-space → original pixel coords."""
                x_min_640 = (xc_n - wn / 2) * s
                y_min_640 = (yc_n - hn / 2) * s
                x_max_640 = (xc_n + wn / 2) * s
                y_max_640 = (yc_n + hn / 2) * s
                return (
                    int(max(0, (x_min_640 - pad_l) / ratio)),
                    int(max(0, (y_min_640 - pad_t) / ratio)),
                    int(min(w_orig, (x_max_640 - pad_l) / ratio)),
                    int(min(h_orig, (y_max_640 - pad_t) / ratio)),
                )

            # Build orig-space normalised array for LineMerger
            orig_boxes_norm = []
            for (xc, yc, wn, hn) in coords_norm:
                x0, y0, x1, y1 = to_orig(xc, yc, wn, hn)
                orig_boxes_norm.append([
                    ((x0 + x1) / 2) / w_orig,
                    ((y0 + y1) / 2) / h_orig,
                    (x1 - x0)       / w_orig,
                    (y1 - y0)       / h_orig,
                ])
            orig_boxes_norm = np.array(orig_boxes_norm)

            line_regions = self.merger.merge(orig_boxes_norm, confs,
                                             w_orig, h_orig)

            # ── Build SpatialTokens ───────────────────────────────────────────
            tokens: list[SpatialToken] = []

            # Simple heuristic: top 15% of page = letterhead (printed)
            letterhead_y_thresh = h_orig * 0.15

            for idx, region in enumerate(line_regions):
                x0, y0 = region["x_min"], region["y_min"]
                x1, y1 = region["x_max"], region["y_max"]

                crop = clean.crop((x0, y0, x1, y1))

                # Ensure minimum height for TrOCR (avoids squeezed crops)
                crop_h = y1 - y0
                if crop_h < 24:
                    pad_px = (24 - crop_h) // 2
                    crop = ImageOps.expand(crop, border=(0, pad_px), fill=255)

                is_printed = (y0 < letterhead_y_thresh)
                crop = self._preprocess_crop(crop, is_printed)
                text = self._run_trocr(crop)

                xc_n = ((x0 + x1) / 2) / w_orig
                yc_n = ((y0 + y1) / 2) / h_orig

                tokens.append(SpatialToken(
                    text=text,
                    x_min=x0, y_min=y0, x_max=x1, y_max=y1,
                    x_center=xc_n, y_center=yc_n,
                    width_norm=(x1 - x0) / w_orig,
                    height_norm=(y1 - y0) / h_orig,
                    confidence=region["conf"],
                    region_id=idx,
                    source_path=image_path,
                ))

            return tokens

        except Exception as exc:
            print(f"[Extractor] ERROR {image_path}: {exc}")
            return []
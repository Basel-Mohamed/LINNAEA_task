"""Text detection with EasyOCR and recognition with TrOCR."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import logging as transformers_logging
from src.ocr.vision_utils import ClinicalImageEnhancer


@dataclass
class SpatialToken:
    """OCR output tied to a crop in the source document."""

    text: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    x_center: float
    y_center: float
    width_norm: float
    height_norm: float
    confidence: float
    region_id: int
    source_path: str = ""
    zone: str = "body"

    @property
    def bbox(self) -> dict:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "text": self.text,
            "bbox": self.bbox,
            "x_center": round(self.x_center, 4),
            "y_center": round(self.y_center, 4),
            "width_norm": round(self.width_norm, 4),
            "height_norm": round(self.height_norm, 4),
            "confidence": round(self.confidence, 4),
            "zone": self.zone,
            "source_path": self.source_path,
        }


class LineMerger:
    """Merge nearby word boxes into line-level regions for TrOCR."""

    def __init__(self, y_tol: float = 0.55, x_gap_tol: float = 0.08):
        self.y_tol = y_tol
        self.x_gap_tol = x_gap_tol

    def merge(self, boxes_norm: np.ndarray, confs: np.ndarray, img_w: int, img_h: int) -> list[dict]:
        """Group normalized detection boxes into line crops in image coordinates."""
        if len(boxes_norm) == 0:
            return []

        abs_boxes = []
        for (xc, yc, w, h), confidence in zip(boxes_norm, confs):
            abs_boxes.append(
                {
                    "x_min": (xc - w / 2) * img_w,
                    "x_max": (xc + w / 2) * img_w,
                    "y_min": (yc - h / 2) * img_h,
                    "y_max": (yc + h / 2) * img_h,
                    "yc": yc * img_h,
                    "h": h * img_h,
                    "conf": float(confidence),
                }
            )

        abs_boxes.sort(key=lambda box: (box["yc"], box["x_min"]))

        lines: list[list[dict]] = []
        for box in abs_boxes:
            placed = False
            for line in lines:
                avg_h = float(np.mean([entry["h"] for entry in line]))
                line_yc = float(np.mean([entry["yc"] for entry in line]))
                if abs(box["yc"] - line_yc) < self.y_tol * avg_h:
                    rightmost = max(entry["x_max"] for entry in line)
                    gap = (box["x_min"] - rightmost) / img_w
                    if gap < self.x_gap_tol:
                        line.append(box)
                        placed = True
                        break
            if not placed:
                lines.append([box])

        merged = []
        for line in lines:
            merged.append(
                {
                    "x_min": int(max(0, min(box["x_min"] for box in line) - 4)),
                    "y_min": int(max(0, min(box["y_min"] for box in line) - 4)),
                    "x_max": int(min(img_w, max(box["x_max"] for box in line) + 4)),
                    "y_max": int(min(img_h, max(box["y_max"] for box in line) + 4)),
                    "conf": float(np.mean([box["conf"] for box in line])),
                }
            )

        merged.sort(key=lambda box: box["y_min"])
        return merged


class PrescriptionExtractor:
    """Run EasyOCR detection and TrOCR recognition on medical document pages."""

    CROP_W = 768
    CROP_H = 128

    def __init__(
        self,
        trocr_model: str = "microsoft/trocr-large-handwritten",
        trocr_proc: str = "microsoft/trocr-large-handwritten",
        hf_token: str | None = None,
        easyocr_lang_list: list[str] | None = None,
        easyocr_gpu: bool = True,
        min_size: int = 10,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        add_margin: float = 0.1,
        y_tol: float = 0.55,
    ):
        self._require_runtime_dependencies()
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.merger = LineMerger(y_tol=y_tol)
        self.easyocr_lang_list = easyocr_lang_list or ["en", "ar"]
        self.detect_kwargs = {
            "min_size": min_size,
            "text_threshold": text_threshold,
            "low_text": low_text,
            "link_threshold": link_threshold,
            "canvas_size": canvas_size,
            "mag_ratio": mag_ratio,
            "slope_ths": slope_ths,
            "ycenter_ths": ycenter_ths,
            "height_ths": height_ths,
            "width_ths": width_ths,
            "add_margin": add_margin,
        }
        self.detector = easyocr.Reader(
            self.easyocr_lang_list,
            gpu=easyocr_gpu and self.device == "cuda",
            detector=True,
            recognizer=False,
            download_enabled=True,
        )

        with self._suppress_transformers_startup_noise():
            self.processor = TrOCRProcessor.from_pretrained(
                trocr_proc,
                use_safetensors=True,
                token=hf_token,
            )
            self.trocr = VisionEncoderDecoderModel.from_pretrained(
                trocr_model,
                use_safetensors=True,
                token=hf_token,
            ).to(self.device)

    @staticmethod
    def _require_runtime_dependencies() -> None:
        missing = []
        if torch is None:
            missing.append("torch")
        if easyocr is None:
            missing.append("easyocr")
        if TrOCRProcessor is None or VisionEncoderDecoderModel is None:
            missing.append("transformers")
        if missing:
            raise ImportError(
                "PrescriptionExtractor requires the following packages at runtime: "
                + ", ".join(sorted(set(missing)))
            )

    @staticmethod
    @contextmanager
    def _suppress_transformers_startup_noise():
        """Hide harmless TrOCR startup warnings during model initialization."""
        if transformers_logging is None:
            yield
            return
        previous_verbosity = transformers_logging.get_verbosity()
        try:
            transformers_logging.set_verbosity_error()
            yield
        finally:
            transformers_logging.set_verbosity(previous_verbosity)

    def extract(self, image_paths: list[str]) -> list[list[SpatialToken]]:
        """Extract OCR tokens from each image path."""
        return [self._process_single(path) for path in image_paths]

    @staticmethod
    def _normalize_box(x_min: int, x_max: int, y_min: int, y_max: int, img_w: int, img_h: int) -> list[float]:
        """Convert an absolute rectangle into normalized xywh format."""
        return [
            ((x_min + x_max) / 2) / img_w,
            ((y_min + y_max) / 2) / img_h,
            (x_max - x_min) / img_w,
            (y_max - y_min) / img_h,
        ]

    @staticmethod
    def _freeform_to_rect(points: list[list[int]]) -> tuple[int, int, int, int]:
        """Collapse a quadrilateral box into an axis-aligned rectangle."""
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), max(xs), min(ys), max(ys)

    def _run_detector(self, pil_img: Image.Image) -> tuple[list[list[float]], list[float]]:
        """Run EasyOCR detection and normalize all returned boxes."""
        image_array = np.array(pil_img)
        horizontal_list, free_list = self.detector.detect(image_array, **self.detect_kwargs)
        horizontal_boxes = horizontal_list[0] if horizontal_list else []
        freeform_boxes = free_list[0] if free_list else []

        boxes_norm: list[list[float]] = []
        confidences: list[float] = []

        for x_min, x_max, y_min, y_max in horizontal_boxes:
            boxes_norm.append(
                self._normalize_box(int(x_min), int(x_max), int(y_min), int(y_max), pil_img.width, pil_img.height)
            )
            confidences.append(1.0)

        for points in freeform_boxes:
            x_min, x_max, y_min, y_max = self._freeform_to_rect(points)
            boxes_norm.append(
                self._normalize_box(int(x_min), int(x_max), int(y_min), int(y_max), pil_img.width, pil_img.height)
            )
            confidences.append(1.0)

        return boxes_norm, confidences

    def _preprocess_crop(self, crop: Image.Image, is_printed: bool) -> Image.Image:
        """Apply lightweight enhancement tailored to printed or handwritten regions."""
        if is_printed:
            crop = ImageEnhance.Contrast(crop).enhance(2.0)
            crop = ImageEnhance.Sharpness(crop).enhance(2.0)
        else:
            crop = crop.filter(ImageFilter.SHARPEN)
        return crop

    def _run_trocr(self, crop: Image.Image) -> str:
        """Recognize one cropped line image with TrOCR."""
        crop_padded = ImageOps.pad(crop, (self.CROP_W, self.CROP_H), color=(255, 255, 255))
        pixel_values = self.processor(crop_padded, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.trocr.generate(pixel_values, max_new_tokens=128)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _process_single(self, image_path: str) -> list[SpatialToken]:
        """Detect line regions on one page, then recognize each region with TrOCR."""
        try:
            raw = Image.open(image_path).convert("RGB")
            clean = ClinicalImageEnhancer.remove_stamps_from_image(raw)
            orig_width, orig_height = clean.size

            coords_norm, confs = self._run_detector(clean)
            if not coords_norm:
                return []
            line_regions = self.merger.merge(np.array(coords_norm), np.array(confs), orig_width, orig_height)

            tokens: list[SpatialToken] = []
            letterhead_y_thresh = orig_height * 0.15

            for idx, region in enumerate(line_regions):
                x0, y0, x1, y1 = region["x_min"], region["y_min"], region["x_max"], region["y_max"]
                crop = clean.crop((x0, y0, x1, y1))
                crop_h = y1 - y0
                if crop_h < 24:
                    pad_px = (24 - crop_h) // 2
                    crop = ImageOps.expand(crop, border=(0, pad_px), fill=255)

                crop = self._preprocess_crop(crop, is_printed=y0 < letterhead_y_thresh)
                text = self._run_trocr(crop)

                tokens.append(
                    SpatialToken(
                        text=text,
                        x_min=x0,
                        y_min=y0,
                        x_max=x1,
                        y_max=y1,
                        x_center=((x0 + x1) / 2) / orig_width,
                        y_center=((y0 + y1) / 2) / orig_height,
                        width_norm=(x1 - x0) / orig_width,
                        height_norm=(y1 - y0) / orig_height,
                        confidence=region["conf"],
                        region_id=idx,
                        source_path=image_path,
                    )
                )
            return tokens
        except Exception as exc:  # pragma: no cover - runtime path
            print(f"[Extractor] ERROR {image_path}: {exc}")
            return []

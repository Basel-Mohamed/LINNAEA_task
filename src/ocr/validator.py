"""Rule-based and VLM-assisted validation for OCR tokens."""
from __future__ import annotations
import asyncio
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional
from PIL import Image, ImageFilter, ImageOps
import torch
from groq import Groq
from src.ocr.extractor import SpatialToken


DOSAGE_RANGES: dict[str, tuple[float, float]] = {
    "paracetamol": (100, 1000),
    "amoxicillin": (125, 875),
    "ibuprofen": (100, 800),
    "metformin": (500, 2000),
    "omeprazole": (10, 40),
    "default": (0.1, 5000),
}

ARABIC_RE = re.compile(r"[؀-ۿ]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
WORD_RE = re.compile(r"[\w؀-ۿ]+")


def _contains_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def _latin_words(text: str) -> list[str]:
    return LATIN_WORD_RE.findall(text)


def _normalize_text(text: str) -> str:
    lowered = text.casefold().strip()
    return re.sub(r"\s+", " ", lowered)


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        for j, rch in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (lch != rch)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _normalized_edit_distance(left: str, right: str) -> float:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    longest = max(len(normalized_left), len(normalized_right), 1)
    return _levenshtein_distance(normalized_left, normalized_right) / longest


@dataclass
class ValidationFlag:
    """A deterministic validation outcome for one token."""

    rule: str
    token: SpatialToken
    detail: str
    action: str
    severity: str


@dataclass
class VLMVerdict:
    """Blind-read comparison between the crop and OCR text."""

    ocr_text: str
    vlm_reading: str
    match: bool
    confidence: str
    vlm_reasoning: str
    suggested_fix: str = ""


@dataclass
class ValidationReport:
    """Combined deterministic and VLM validation result for one token."""

    token: SpatialToken
    flags: list[ValidationFlag] = field(default_factory=list)
    vlm_verdict: Optional[VLMVerdict] = None
    final_text: str = ""


class ConfidenceGate:
    """Escalate low-confidence OCR tokens for review or rescanning."""

    def __init__(self, low: float = 0.92, critical: float = 0.78):
        self.low = low
        self.critical = critical

    def check(self, token: SpatialToken) -> Optional[ValidationFlag]:
        if token.confidence < self.critical:
            return ValidationFlag(
                "ConfidenceGate",
                token,
                f"conf {token.confidence:.2f} < {self.critical} (RESCAN)",
                "RESCAN",
                "HIGH",
            )
        if token.confidence < self.low:
            return ValidationFlag(
                "ConfidenceGate",
                token,
                f"conf {token.confidence:.2f} < {self.low} (WARN)",
                "WARN",
                "MEDIUM",
            )
        return None


class GarbageTextRule:
    """Detect clearly corrupted Latin OCR while ignoring Arabic-only tokens."""

    VOWELS = set("aeiou")
    CONSONANT_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]{5,}", re.IGNORECASE)
    COMMON_SHORT = {
        "the",
        "a",
        "an",
        "to",
        "for",
        "of",
        "in",
        "on",
        "is",
        "mr",
        "dr",
        "no",
        "id",
        "by",
        "we",
        "or",
        "at",
    }

    def check(self, token: SpatialToken) -> Optional[ValidationFlag]:
        text = token.text.strip()
        if len(text) < 4:
            return None

        latin_words = _latin_words(text)
        if not latin_words:
            return None

        for word in latin_words:
            lowered = word.lower()
            if len(lowered) <= 3 and lowered in self.COMMON_SHORT:
                continue
            if self.CONSONANT_RE.search(lowered):
                return ValidationFlag(
                    "GarbageTextRule",
                    token,
                    f"Consonant cluster in '{word}' indicates OCR corruption",
                    "RESCAN",
                    "HIGH",
                )
            if len(lowered) >= 6:
                vowel_ratio = sum(1 for char in lowered if char in self.VOWELS) / len(lowered)
                if vowel_ratio < 0.15:
                    return ValidationFlag(
                        "GarbageTextRule",
                        token,
                        f"'{word}' has an implausibly low vowel ratio",
                        "RESCAN",
                        "HIGH",
                    )

        parts = [part for part in text.split("-") if part]
        if len(parts) >= 3 and all(not _contains_arabic(part) for part in parts):
            return ValidationFlag(
                "GarbageTextRule",
                token,
                f"Suspicious multi-hyphen token: '{text[:40]}'",
                "RESCAN",
                "MEDIUM",
            )

        return None


class DosageRangeRule:
    """Flag implausible medication doses."""

    _DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mg", re.IGNORECASE)

    def check(self, token: SpatialToken, drug_hint: str = "default") -> Optional[ValidationFlag]:
        matches = self._DOSE_RE.findall(token.text)
        if not matches:
            return None
        dose = float(matches[0])
        low, high = DOSAGE_RANGES.get(drug_hint.lower(), DOSAGE_RANGES["default"])
        if low <= dose <= high:
            return None
        action = "RESCAN" if dose > high * 2 else "WARN"
        return ValidationFlag(
            "DosageRangeRule",
            token,
            f"Dose {dose}mg outside [{low}-{high}mg]",
            action,
            "HIGH" if action == "RESCAN" else "MEDIUM",
        )


class FrequencyPlausibilityRule:
    """Warn when dosing intervals are unusually frequent."""

    _PATTERN = re.compile(r"(?:every\s*)?(\d+)\s*(h|hr|hrs|hour|hours|\u0633\u0627\u0639(?:\u0629)?)", re.IGNORECASE)

    def check(self, token: SpatialToken) -> Optional[ValidationFlag]:
        match = self._PATTERN.search(token.text)
        if not match:
            return None
        interval = int(match.group(1))
        if interval <= 4:
            return ValidationFlag(
                "FrequencyRule",
                token,
                f"Interval of every {interval}h is unusually frequent",
                "WARN",
                "MEDIUM",
            )
        return None


class DrugNameSanityRule:
    """Catch numeric-only or out-of-vocabulary drug names."""

    def check(self, token: SpatialToken, drug_dict: dict) -> Optional[ValidationFlag]:
        text = token.text.strip()
        if re.fullmatch(r"[\d\s\u0660-\u0669]+", text):
            return ValidationFlag(
                "DrugNameRule",
                token,
                "Drug name is purely numeric",
                "RESCAN",
                "HIGH",
            )
        if len(text) > 1 and text.lower() not in drug_dict:
            return ValidationFlag(
                "DrugNameRule",
                token,
                f"'{text}' not in drug dictionary",
                "WARN",
                "LOW",
            )
        return None


class VLMValidator:
    """Blindly read a crop with the VLM and compare it to OCR output."""

    MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    BLIND_READ_PROMPT = (
        "You are reading a cropped medical document image. Read only the text visible in the crop. "
        "Do not infer missing text. Return valid JSON only with keys: "
        '{"reading": "<text>", "confidence": "high|medium|low", "reasoning": "<brief note>"}.'
    )

    def __init__(self, api_key: str, mismatch_threshold: float = 0.22):
        if Groq is None:
            raise ImportError("groq is required to use VLMValidator")
        self.client = Groq(api_key=api_key)
        self.mismatch_threshold = mismatch_threshold
        self._executor = ThreadPoolExecutor(max_workers=6)

    def _crop_to_base64(self, token: SpatialToken) -> str:
        """Encode the token crop as inline JPEG for the VLM call."""
        image = Image.open(token.source_path).convert("RGB")
        x0 = max(token.x_min - 10, 0)
        y0 = max(token.y_min - 10, 0)
        x1 = min(token.x_max + 10, image.width)
        y1 = min(token.y_max + 10, image.height)
        crop = image.crop((x0, y0, x1, y1)).filter(ImageFilter.SHARPEN)
        buffer = BytesIO()
        crop.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _blind_read_crop(self, b64: str) -> dict:
        """Ask the VLM to read the crop without showing OCR text."""
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.BLIND_READ_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        }
                    ],
                },
            ],
            temperature=0.0,
            max_completion_tokens=192,
            stream=False,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    def _call_groq(self, b64: str, ocr_text: str) -> VLMVerdict:
        """Run blind reading first, then compare the result to OCR text."""
        try:
            blind_read = self._blind_read_crop(b64)
            vlm_reading = str(blind_read.get("reading", "")).strip()
            confidence = str(blind_read.get("confidence", "low")).lower()
            reasoning = str(blind_read.get("reasoning", "blind read complete")).strip()
            distance = _normalized_edit_distance(vlm_reading, ocr_text)
            match = distance <= self.mismatch_threshold
            return VLMVerdict(
                ocr_text=ocr_text,
                vlm_reading=vlm_reading,
                match=match,
                confidence=confidence,
                vlm_reasoning=f"{reasoning}; normalized_edit_distance={distance:.3f}",
                suggested_fix=vlm_reading if not match and vlm_reading else "",
            )
        except Exception as exc:
            return VLMVerdict(
                ocr_text=ocr_text,
                vlm_reading="",
                match=True,
                confidence="low",
                vlm_reasoning=str(exc),
                suggested_fix="",
            )

    async def verify_async(self, token: SpatialToken) -> VLMVerdict:
        loop = asyncio.get_event_loop()
        b64 = self._crop_to_base64(token)
        return await loop.run_in_executor(self._executor, self._call_groq, b64, token.text)


class CropReScanAgent:
    """Retry OCR on suspicious crops, preferring high-confidence VLM fixes."""

    RESCAN_SIZE = 512

    def __init__(self, processor, model, device):
        self.processor = processor
        self.model = model
        self.device = device

    def rescan(self, token: SpatialToken, vlm_verdict: Optional[VLMVerdict] = None) -> str:
        if (
            vlm_verdict
            and not vlm_verdict.match
            and vlm_verdict.confidence == "high"
            and vlm_verdict.suggested_fix
        ):
            return vlm_verdict.suggested_fix

        if torch is None:
            return token.text

        try:
            image = Image.open(token.source_path).convert("RGB")
            x0 = max(token.x_min - 5, 0)
            y0 = max(token.y_min - 5, 0)
            x1 = min(token.x_max + 5, image.width)
            y1 = min(token.y_max + 5, image.height)
            crop = image.crop((x0, y0, x1, y1))
            crop = crop.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
            crop = ImageOps.pad(crop, (self.RESCAN_SIZE, self.RESCAN_SIZE), color=(255, 255, 255))
            pixel_values = self.processor(crop, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                ids = self.model.generate(pixel_values, max_new_tokens=128)
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        except Exception:
            return token.text


class ValidatorAgent:
    """Apply deterministic rules and optional VLM verification across tokens."""

    def __init__(
        self,
        processor,
        trocr_model,
        device,
        drug_dict: dict,
        groq_api_key: str | None,
        vlm_on_warn: bool = True,
        always_vlm_body: bool = False,
    ):
        self.drug_dict = drug_dict
        self.vlm_on_warn = vlm_on_warn
        self.always_vlm_body = always_vlm_body

        self.confidence_gate = ConfidenceGate()
        self.garbage_rule = GarbageTextRule()
        self.dosage_rule = DosageRangeRule()
        self.frequency_rule = FrequencyPlausibilityRule()
        self.drug_rule = DrugNameSanityRule()

        self.vlm = VLMValidator(api_key=groq_api_key) if groq_api_key else None
        self.rescan_agent = CropReScanAgent(processor, trocr_model, device)

    async def validate_token_async(
        self,
        token: SpatialToken,
        drug_hint: str = "default",
        is_drug: bool = False,
    ) -> ValidationReport:
        """Validate a single OCR token and optionally rescan it."""
        report = ValidationReport(token=token, final_text=token.text)
        flags: list[ValidationFlag] = []

        for flag in (
            self.confidence_gate.check(token),
            self.garbage_rule.check(token),
            self.dosage_rule.check(token, drug_hint),
            self.frequency_rule.check(token),
            self.drug_rule.check(token, self.drug_dict) if is_drug else None,
        ):
            if flag is not None:
                flags.append(flag)

        report.flags = flags
        has_rescan = any(flag.action == "RESCAN" for flag in flags)
        has_warn = any(flag.action == "WARN" for flag in flags)

        send_to_vlm = self.vlm is not None and (
            has_rescan
            or (self.vlm_on_warn and has_warn)
            or (self.always_vlm_body and token.zone == "body")
        )

        if send_to_vlm and self.vlm is not None:
            verdict = await self.vlm.verify_async(token)
            report.vlm_verdict = verdict
            report.final_text = self.rescan_agent.rescan(token, verdict) if not verdict.match else token.text

        return report

    async def validate_all_async(
        self,
        tokens: list[SpatialToken],
        drug_pairs: list[tuple[str, str]],
    ) -> list[ValidationReport]:
        """Validate a page worth of OCR tokens concurrently."""
        drug_names = {drug.lower() for drug, _ in drug_pairs}
        tasks = []
        for token in tokens:
            words = WORD_RE.findall(token.text.lower())
            is_drug = any(word in drug_names for word in words)
            hint = next((word for word in words if word in DOSAGE_RANGES), "default")
            tasks.append(self.validate_token_async(token, drug_hint=hint, is_drug=is_drug))
        return list(await asyncio.gather(*tasks))

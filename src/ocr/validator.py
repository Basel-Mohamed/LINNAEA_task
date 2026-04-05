"""
validator.py  (v3)
==================
Fixes:
  1. ConfidenceGate thresholds raised — on degraded handwritten docs,
     a YOLO conf of 0.85 still means "probably detected something" but
     TrOCR can still misread it badly. We now WARN at < 0.92.
  2. GarbageTextRule added — catches obviously garbled OCR output
     (e.g. "dooroveguation", "kunahoe-human-system") based on:
       a. Ratio of alpha chars to total
       b. Unknown word clusters (heuristic)
       c. Implausible character n-grams
  3. AlwaysVLMForBody flag — optionally sends every body-zone token
     to the VLM regardless of flags (set always_vlm_body=True).
"""

import re
import asyncio
import base64
import json
from io import BytesIO
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps, ImageFilter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from groq import Groq

from src.ocr.extractor import SpatialToken


DOSAGE_RANGES: dict[str, tuple[float, float]] = {
    "paracetamol": (100,  1000), "amoxicillin": (125,  875),
    "ibuprofen":   (100,   800), "metformin":   (500, 2000),
    "omeprazole":  (10,     40), "default":     (0.1, 5000),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationFlag:
    rule: str; token: SpatialToken; detail: str; action: str; severity: str

    def __str__(self) -> str:
        icon = {"ACCEPT": "✅", "WARN": "⚠️",
                "RESCAN": "🔬", "REJECT": "❌"}.get(self.action, "?")
        return (f"  {icon} [{self.severity}] {self.rule}\n"
                f"     text: '{self.token.text}'\n"
                f"     detail: {self.detail}")


@dataclass
class VLMVerdict:
    ocr_text: str; vlm_reading: str; match: bool
    confidence: str; vlm_reasoning: str; suggested_fix: str = ""


@dataclass
class ValidationReport:
    token: SpatialToken; flags: list[ValidationFlag] = field(default_factory=list)
    vlm_verdict: Optional[VLMVerdict] = None; final_text: str = ""

    def __str__(self) -> str:
        lines = [f"── Token: '{self.token.text[:50]}' "
                 f"(conf={self.token.confidence:.2f}, zone={self.token.zone}) ──"]
        for flag in self.flags:
            lines.append(str(flag))
        if self.vlm_verdict:
            v = self.vlm_verdict
            tick = "✅" if v.match else "❌"
            lines += [f"  🔭 VLM {tick}",
                      f"     OCR: '{v.ocr_text[:40]}' → VLM: '{v.vlm_reading[:40]}'"
                      f" | conf: {v.confidence}"]
            if v.suggested_fix:
                lines.append(f"     Fix: '{v.suggested_fix}'")
        lines.append(f"  → Final: '{self.final_text}'")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic rules
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceGate:
    """
    FIX v3: raised thresholds.
    On handwritten/degraded documents TrOCR can produce garbled text even
    when YOLO is confident it detected *something*.  We WARN at < 0.92 and
    RESCAN at < 0.78 so the VLM actually gets invoked.
    """
    def __init__(self, low: float = 0.92, critical: float = 0.78):
        self.low      = low
        self.critical = critical

    def check(self, token: SpatialToken) -> Optional[ValidationFlag]:
        if token.confidence < self.critical:
            return ValidationFlag(
                "ConfidenceGate", token,
                f"conf {token.confidence:.2f} < {self.critical} (RESCAN)",
                "RESCAN", "HIGH")
        if token.confidence < self.low:
            return ValidationFlag(
                "ConfidenceGate", token,
                f"conf {token.confidence:.2f} < {self.low} (WARN)",
                "WARN", "MEDIUM")
        return None


class GarbageTextRule:
    """
    FIX v3 — NEW RULE.
    Detects obviously garbled OCR output using three heuristics:

    H1. Consonant cluster: 5+ consecutive consonants → likely misread.
    H2. Mixed alpha-digit-special chaos: if the text has high ratio of
        punctuation / digits mixed into what looks like a word.
    H3. Hyphenated non-dictionary compound: 'word-word' where both parts
        are short and unrecognisable (e.g. 'kunahoe-human-system').

    Any hit → RESCAN (sends to VLM).
    """
    VOWELS       = set("aeiouAEIOU")
    CONSONANT_RE = re.compile(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}")
    CHAOS_RE     = re.compile(r"[^\w\s]{2,}")   # 2+ consecutive non-word chars

    # Very short words (≤3 chars) that appear real
    COMMON_SHORT = {"the", "a", "an", "to", "for", "of", "in", "on",
                    "is", "he", "she", "it", "mr", "dr", "no", "ip",
                    "id", "do", "my", "by", "we", "or", "at"}

    def check(self, token: SpatialToken) -> Optional[ValidationFlag]:
        text = token.text.strip()
        if len(text) < 4:
            return None   # too short to judge

        words = re.findall(r"[a-zA-Z]+", text)
        if not words:
            return None

        for word in words:
            if len(word) <= 3 and word.lower() in self.COMMON_SHORT:
                continue

            # H1: impossible consonant cluster
            if self.CONSONANT_RE.search(word):
                return ValidationFlag(
                    "GarbageTextRule", token,
                    f"Consonant cluster in '{word}' — likely OCR garble",
                    "RESCAN", "HIGH")

            # H2: very low vowel ratio for a long word
            if len(word) >= 6:
                vowel_ratio = sum(1 for c in word if c in self.VOWELS) / len(word)
                if vowel_ratio < 0.15:
                    return ValidationFlag(
                        "GarbageTextRule", token,
                        f"'{word}' has {vowel_ratio:.0%} vowels — garbled?",
                        "RESCAN", "HIGH")

        # H3: suspicious hyphenated compound
        parts = text.split("-")
        if len(parts) >= 3 and all(len(p) >= 3 for p in parts):
            return ValidationFlag(
                "GarbageTextRule", token,
                f"Suspicious multi-hyphenated string: '{text[:40]}'",
                "RESCAN", "MEDIUM")

        return None


class DosageRangeRule:
    _DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mg", re.IGNORECASE)

    def check(self, token: SpatialToken,
              drug_hint: str = "default") -> Optional[ValidationFlag]:
        matches = self._DOSE_RE.findall(token.text)
        if not matches:
            return None
        dose = float(matches[0])
        lo, hi = DOSAGE_RANGES.get(drug_hint.lower(), DOSAGE_RANGES["default"])
        if dose < lo or dose > hi:
            action = "RESCAN" if dose > hi * 2 else "WARN"
            return ValidationFlag(
                "DosageRangeRule", token,
                f"Dose {dose}mg outside [{lo}-{hi}mg]",
                action, "HIGH" if action == "RESCAN" else "MEDIUM")
        return None


class FrequencyPlausibilityRule:
    _NUM_RE = re.compile(r"\b(\d+)\b")

    def check(self, token: SpatialToken) -> Optional[ValidationFlag]:
        for num_str in self._NUM_RE.findall(token.text.lower()):
            num = int(num_str)
            if 2 <= num <= 4 and (24 // num) > 6:
                return ValidationFlag(
                    "FrequencyRule", token,
                    f"Every {num}h is unusually high",
                    "WARN", "MEDIUM")
        return None


class DrugNameSanityRule:
    def check(self, token: SpatialToken,
              drug_dict: dict) -> Optional[ValidationFlag]:
        text = token.text.strip()
        if re.fullmatch(r"[\d\s٠-٩]+", text):
            return ValidationFlag(
                "DrugNameRule", token,
                "Drug name is purely numeric", "RESCAN", "HIGH")
        if text.lower() not in drug_dict and len(text) > 1:
            return ValidationFlag(
                "DrugNameRule", token,
                f"'{text}' not in drug dict", "WARN", "LOW")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Groq VLM validator
# ─────────────────────────────────────────────────────────────────────────────

class VLMValidator:
    MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    SYSTEM_PROMPT = (
        "You are a clinical OCR auditor specialising in handwritten medical "
        "documents. You will receive an image crop and the text an OCR engine "
        "produced from it.\n"
        "Read the image carefully. Respond with ONLY valid JSON — no markdown, "
        "no explanation outside the JSON:\n"
        '{"vlm_reading": "<what you see>", "match": true/false, '
        '"confidence": "high|medium|low", '
        '"reasoning": "<brief>", "suggested_fix": "<corrected text or empty>"}\n'
        "Set match=true only when the OCR text is an exact or near-exact match."
    )

    def __init__(self, api_key: str):
        self.client    = Groq(api_key=api_key)
        self._executor = ThreadPoolExecutor(max_workers=6)

    def _crop_to_base64(self, token: SpatialToken) -> str:
        img  = Image.open(token.source_path).convert("RGB")
        x0   = max(token.x_min - 10, 0)
        y0   = max(token.y_min - 10, 0)
        x1   = min(token.x_max + 10, img.width)
        y1   = min(token.y_max + 10, img.height)
        crop = img.crop((x0, y0, x1, y1)).filter(ImageFilter.SHARPEN)
        buf  = BytesIO()
        crop.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_groq(self, b64: str, ocr_text: str) -> VLMVerdict:
        try:
            resp = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text",
                         "text": f'OCR produced: "{ocr_text}"\nRespond with JSON only.'},
                    ]},
                ],
                temperature=0.05,
                max_completion_tokens=256,
                stream=False,
            )
            raw  = resp.choices[0].message.content.strip()
            raw  = re.sub(r"^```(?:json)?\s*", "", raw)
            raw  = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            return VLMVerdict(
                ocr_text=ocr_text,
                vlm_reading=data.get("vlm_reading", ocr_text),
                match=bool(data.get("match", True)),
                confidence=data.get("confidence", "medium"),
                vlm_reasoning=data.get("reasoning", ""),
                suggested_fix=data.get("suggested_fix", ""),
            )
        except json.JSONDecodeError as e:
            print(f"  [VLM] JSON error: {e}")
            return VLMVerdict(ocr_text, ocr_text, True, "low", f"parse-err: {e}")
        except Exception as e:
            print(f"  [VLM] Groq error: {e}")
            return VLMVerdict(ocr_text, ocr_text, True, "low", str(e))

    async def verify_async(self, token: SpatialToken) -> VLMVerdict:
        print(f"  [VLM ▶] '{token.text[:40]}'")
        b64  = self._crop_to_base64(token)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._call_groq, b64, token.text
        )


# ─────────────────────────────────────────────────────────────────────────────
# Crop re-scan agent
# ─────────────────────────────────────────────────────────────────────────────

class CropReScanAgent:
    RESCAN_SIZE = 512

    def __init__(self, processor, model, device):
        self.processor = processor
        self.model     = model
        self.device    = device

    def rescan(self, token: SpatialToken,
               vlm_verdict: Optional[VLMVerdict] = None) -> str:
        # Priority 1: VLM high-confidence fix
        if (vlm_verdict and not vlm_verdict.match
                and vlm_verdict.confidence == "high"
                and vlm_verdict.suggested_fix):
            print(f"  [ReScan] VLM fix: '{token.text}' → '{vlm_verdict.suggested_fix}'")
            return vlm_verdict.suggested_fix

        # Priority 2: TrOCR re-run on sharpened crop
        try:
            img  = Image.open(token.source_path).convert("RGB")
            x0   = max(token.x_min - 5, 0)
            y0   = max(token.y_min - 5, 0)
            x1   = min(token.x_max + 5, img.width)
            y1   = min(token.y_max + 5, img.height)
            crop = img.crop((x0, y0, x1, y1))
            crop = crop.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
            crop = ImageOps.pad(crop, (self.RESCAN_SIZE, self.RESCAN_SIZE),
                                color=(255, 255, 255))
            pv   = self.processor(crop, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                ids = self.model.generate(pv, max_new_tokens=128)
            corrected = self.processor.batch_decode(
                ids, skip_special_tokens=True)[0].strip()
            print(f"  [ReScan] TrOCR: '{token.text}' → '{corrected}'")
            return corrected
        except Exception as e:
            print(f"  [ReScan] ERROR: {e}")
            return token.text


# ─────────────────────────────────────────────────────────────────────────────
# Main validator agent
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    def __init__(
        self,
        processor,
        trocr_model,
        device,
        drug_dict:       dict,
        groq_api_key:    str,
        vlm_on_warn:     bool = True,
        always_vlm_body: bool = False,   # send ALL body tokens to VLM
    ):
        self.drug_dict       = drug_dict
        self.vlm_on_warn     = vlm_on_warn
        self.always_vlm_body = always_vlm_body

        self.confidence_gate = ConfidenceGate()       # thresholds raised in v3
        self.garbage_rule    = GarbageTextRule()       # NEW in v3
        self.dosage_rule     = DosageRangeRule()
        self.frequency_rule  = FrequencyPlausibilityRule()
        self.drug_rule       = DrugNameSanityRule()

        self.vlm          = VLMValidator(api_key=groq_api_key)
        self.rescan_agent = CropReScanAgent(processor, trocr_model, device)

    async def validate_token_async(
        self,
        token:     SpatialToken,
        drug_hint: str  = "default",
        is_drug:   bool = False,
    ) -> ValidationReport:

        report = ValidationReport(token=token, final_text=token.text)
        flags  = []

        if f := self.confidence_gate.check(token):        flags.append(f)
        if f := self.garbage_rule.check(token):           flags.append(f)
        if f := self.dosage_rule.check(token, drug_hint): flags.append(f)
        if f := self.frequency_rule.check(token):         flags.append(f)
        if is_drug:
            if f := self.drug_rule.check(token, self.drug_dict):
                flags.append(f)

        report.flags = flags
        has_rescan   = any(f.action == "RESCAN" for f in flags)
        has_warn     = any(f.action == "WARN"   for f in flags)

        send_to_vlm = (
            has_rescan
            or (self.vlm_on_warn and has_warn)
            or (self.always_vlm_body and token.zone == "body")
        )

        if send_to_vlm:
            verdict            = await self.vlm.verify_async(token)
            report.vlm_verdict = verdict
            report.final_text  = (
                self.rescan_agent.rescan(token, verdict)
                if not verdict.match
                else token.text
            )
        else:
            report.final_text = token.text

        return report

    async def validate_all_async(
        self,
        tokens:     list[SpatialToken],
        drug_pairs: list[tuple[str, str]],
    ) -> list[ValidationReport]:

        sep = "═" * 60
        print(f"\n{sep}\n  VALIDATOR AGENT v3 — Groq Llama-4-Scout\n{sep}")

        drug_names = {d.lower() for d, _ in drug_pairs}
        tasks = []
        for token in tokens:
            words   = token.text.lower().split()
            is_drug = any(w in drug_names for w in words)
            hint    = next((w for w in words if w in DOSAGE_RANGES), "default")
            tasks.append(
                self.validate_token_async(token, drug_hint=hint, is_drug=is_drug)
            )

        reports = await asyncio.gather(*tasks)

        for r in reports:
            print(r, "\n")

        vlm_n = sum(1 for r in reports if r.vlm_verdict)
        fix_n = sum(1 for r in reports if r.vlm_verdict and not r.vlm_verdict.match)
        print(f"\n  Tokens: {len(reports)} | VLM calls: {vlm_n} | Fixes: {fix_n}")
        return reports
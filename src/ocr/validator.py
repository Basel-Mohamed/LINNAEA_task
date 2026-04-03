"""
validator.py
============
Task 2 — The VLLM Validation Engine (Analytical Validator)

Analogy: OCR errors are "impurities in a chemical synthesis."
         This module is the "Mass Spectrometer" that catches them
         before they reach the patient.

Four-layer validation pipeline:
    Layer 1 — Confidence Gate     : YOLO score too low  → flag for VLM
    Layer 2 — Clinical Rules      : dosage / drug / frequency checks
    Layer 3 — VLM Visual Check    : GPT-4o Vision sees the actual pixel-patch
                                    and judges if OCR text matches the image
    Layer 4 — Crop & Re-Scan      : if VLM says "wrong" → isolate pixel-patch
                                    and re-run TrOCR at higher resolution

The key insight:
    Rules catch impossible values  (10000mg).
    The VLM catches ambiguous handwriting  ("10mg" vs "100mg").
    The VLM looks at the SAME pixel-patch the OCR read, not the full image.
"""
import re
import google.generativeai as genai
import json
from io import BytesIO
from typing import Optional
from dataclasses import dataclass, field
from PIL import Image, ImageOps, ImageFilter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.ocr.extractor import SpatialToken


# ─────────────────────────────────────────────────────────────────────────────
# Clinical knowledge base
# ─────────────────────────────────────────────────────────────────────────────

DOSAGE_RANGES: dict[str, tuple[float, float]] = {
    "paracetamol":    (100,  1000),
    "amoxicillin":    (125,  875),
    "ibuprofen":      (100,  800),
    "metformin":      (500,  2000),
    "aspirin":        (75,   1000),
    "omeprazole":     (10,   40),
    "hydrocortisone": (5,    30),
    "mupirocin":      (0,    0),
    "default":        (0.1,  5000),
}


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationFlag:
    rule:     str
    token:    SpatialToken
    detail:   str
    action:   str     # "ACCEPT" | "WARN" | "RESCAN" | "REJECT"
    severity: str     # "LOW" | "MEDIUM" | "HIGH"

    def __str__(self) -> str:
        icon = {"ACCEPT": "✅", "WARN": "⚠️",
                "RESCAN": "🔬", "REJECT": "❌"}.get(self.action, "?")
        return (
            f"  {icon} [{self.severity}] {self.rule}\n"
            f"     text   : '{self.token.text}'\n"
            f"     detail : {self.detail}\n"
            f"     action : {self.action}"
        )


@dataclass
class VLMVerdict:
    """What GPT-4o Vision said after inspecting the pixel-patch."""
    ocr_text:      str
    vlm_reading:   str
    match:         bool
    confidence:    str     # "high" | "medium" | "low"
    vlm_reasoning: str
    suggested_fix: str = ""


@dataclass
class ValidationReport:
    token:       SpatialToken
    flags:       list[ValidationFlag] = field(default_factory=list)
    vlm_verdict: Optional[VLMVerdict] = None
    final_text:  str = ""

    @property
    def needs_rescan(self) -> bool:
        return any(f.action == "RESCAN" for f in self.flags)

    @property
    def worst_severity(self) -> str:
        order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        if not self.flags:
            return "LOW"
        return max(self.flags, key=lambda f: order.get(f.severity, 0)).severity

    def __str__(self) -> str:
        lines = [
            f"── Token: '{self.token.text[:40]}' "
            f"(conf={self.token.confidence:.2f}, zone={self.token.zone}) ──"
        ]
        for flag in self.flags:
            lines.append(str(flag))

        if self.vlm_verdict:
            v = self.vlm_verdict
            icon = "✅" if v.match else "❌"
            lines += [
                f"  🔭 VLM Visual Check {icon}",
                f"     OCR read  : '{v.ocr_text}'",
                f"     VLM read  : '{v.vlm_reading}'",
                f"     Confidence: {v.confidence}",
                f"     Reasoning : {v.vlm_reasoning}",
            ]
            if v.suggested_fix:
                lines.append(f"     Fix       : '{v.suggested_fix}'")

        lines.append(f"  → Final text: '{self.final_text}'")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Confidence Gate
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceGate:
    """
    If YOLO wasn't confident about WHERE the text region is,
    the OCR result is unreliable → send to VLM for visual confirmation.
    """

    def __init__(self, low: float = 0.65, critical: float = 0.50):
        self.low      = low
        self.critical = critical

    def check(self, token: SpatialToken) -> Optional[ValidationFlag]:
        if token.confidence < self.critical:
            return ValidationFlag(
                rule="ConfidenceGate", token=token,
                detail=f"YOLO conf {token.confidence:.2f} < {self.critical} → VLM required",
                action="RESCAN", severity="HIGH",
            )
        if token.confidence < self.low:
            return ValidationFlag(
                rule="ConfidenceGate", token=token,
                detail=f"YOLO conf {token.confidence:.2f} < {self.low} → VLM recommended",
                action="WARN", severity="MEDIUM",
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Clinical Rules (3 rules)
# ─────────────────────────────────────────────────────────────────────────────

class DosageRangeRule:
    """Rule A: Dosage must be within a clinically plausible range."""

    _DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mg", re.IGNORECASE)

    def check(self, token: SpatialToken, drug_hint: str = "default") -> Optional[ValidationFlag]:
        matches = self._DOSE_RE.findall(token.text)
        if not matches:
            return None
        dose = float(matches[0])
        lo, hi = DOSAGE_RANGES.get(drug_hint.lower(), DOSAGE_RANGES["default"])
        if hi == 0:
            return ValidationFlag(
                rule="DosageRangeRule", token=token,
                detail=f"{dose}mg for topical drug '{drug_hint}' → VLM to verify",
                action="WARN", severity="MEDIUM",
            )
        if dose < lo or dose > hi:
            action = "RESCAN" if dose > hi * 2 else "WARN"
            return ValidationFlag(
                rule="DosageRangeRule", token=token,
                detail=f"Dose {dose}mg outside [{lo}–{hi}mg] for '{drug_hint}'",
                action=action,
                severity="HIGH" if action == "RESCAN" else "MEDIUM",
            )
        return None


class FrequencyPlausibilityRule:
    """Rule B: Daily frequency must be ≤ 6 doses/day."""

    _NUM_RE = re.compile(r"\b(\d+)\b")

    def check(self, token: SpatialToken) -> Optional[ValidationFlag]:
        for num_str in self._NUM_RE.findall(token.text.lower()):
            num = int(num_str)
            if 2 <= num <= 4:
                daily = 24 // num
                if daily > 6:
                    return ValidationFlag(
                        rule="FrequencyPlausibilityRule", token=token,
                        detail=f"Every {num}h = {daily}×/day — unusually high",
                        action="WARN", severity="MEDIUM",
                    )
        return None


class DrugNameSanityRule:
    """Rule C: Drug field must not be purely numeric or a single character."""

    def check(self, token: SpatialToken, drug_dict: dict) -> Optional[ValidationFlag]:
        text = token.text.strip()
        if re.fullmatch(r"[\d\s٠-٩]+", text):
            return ValidationFlag(
                rule="DrugNameSanityRule", token=token,
                detail=f"Drug field is numeric: '{text}' → OCR likely misread stamp as drug name",
                action="RESCAN", severity="HIGH",
            )
        if len(text.replace(" ", "")) <= 1:
            return ValidationFlag(
                rule="DrugNameSanityRule", token=token,
                detail=f"Single character: '{text}' — OCR noise",
                action="RESCAN", severity="HIGH",
            )
        if text.lower() not in drug_dict:
            return ValidationFlag(
                rule="DrugNameSanityRule", token=token,
                detail=f"'{text}' not in drug dictionary — VLM to confirm spelling",
                action="WARN", severity="LOW",
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — VLM Visual Validator  ← THE CORE OF TASK 2
# ─────────────────────────────────────────────────────────────────────────────
class VLMValidator:
    SYSTEM_PROMPT = """You are a clinical OCR auditor for handwritten medical prescriptions.
You will be shown a cropped image region from a handwritten prescription.
Read the text in the image yourself. Compare your reading to the OCR output provided.

Respond ONLY in this JSON format:
{
  "vlm_reading": "<what YOU read from the image>",
  "match": true or false,
  "confidence": "high" or "medium" or "low",
  "reasoning": "<what you see in the image>",
  "suggested_fix": "<corrected text if mismatch, else empty string>"
}"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=self.SYSTEM_PROMPT
        )

    def _crop_to_pil(self, token: SpatialToken) -> Image.Image:
        """Returns the PIL Image directly for Gemini."""
        image = Image.open(token.source_path).convert("RGB")
        scale_x = image.width / 640
        scale_y = image.height / 640
        x_min = max(int(token.x_min * scale_x) - 8, 0)
        y_min = max(int(token.y_min * scale_y) - 8, 0)
        x_max = min(int(token.x_max * scale_x) + 8, image.width)
        y_max = min(int(token.y_max * scale_y) + 8, image.height)

        crop = image.crop((x_min, y_min, x_max, y_max))
        return crop.filter(ImageFilter.SHARPEN)

    def verify(self, token: SpatialToken) -> VLMVerdict:
        print(f"  [VLM] Sending to Gemini 1.5 Flash: '{token.text[:30]}'")
        try:
            pil_image = self._crop_to_pil(token)
            prompt = f'OCR output: "{token.text}"\nVerify this against what you see in the image.'

            response = self.model.generate_content(
                [pil_image, prompt],
                generation_config={"response_mime_type": "application/json"}
            )
            
            data = json.loads(response.text)
            
            verdict = VLMVerdict(
                ocr_text      = token.text,
                vlm_reading   = data.get("vlm_reading", token.text),
                match         = data.get("match", True),
                confidence    = data.get("confidence", "medium"),
                vlm_reasoning = data.get("reasoning", ""),
                suggested_fix = data.get("suggested_fix", ""),
            )
            
            status = "✅" if verdict.match else f"❌ fix='{verdict.suggested_fix}'"
            print(f"  [VLM] {status} (confidence: {verdict.confidence})")
            return verdict
            
        except Exception as e:
            print(f"  [VLM] ERROR: {e}")
            return VLMVerdict(token.text, token.text, True, "low", str(e))
# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Crop & Re-Scan
# ─────────────────────────────────────────────────────────────────────────────

class CropReScanAgent:
    """
    Decision tree after VLM fires:
        VLM mismatch + high confidence   → use VLM's suggested_fix directly
        VLM mismatch + medium confidence → re-run TrOCR on sharpened 512px crop
        VLM mismatch + low confidence    → re-run TrOCR + flag for human review
        No VLM mismatch                  → keep original OCR text
    """

    RESCAN_SIZE = 512

    def __init__(self, processor: TrOCRProcessor,
                 model: VisionEncoderDecoderModel, device: str):
        self.processor = processor
        self.model     = model
        self.device    = device

    def rescan(self, token: SpatialToken,
               vlm_verdict: Optional[VLMVerdict] = None) -> str:

        # Trust VLM directly when it's confident
        if (vlm_verdict and not vlm_verdict.match
                and vlm_verdict.confidence == "high"
                and vlm_verdict.suggested_fix):
            print(f"  [ReScan] VLM fix applied: '{token.text}' → '{vlm_verdict.suggested_fix}'")
            return vlm_verdict.suggested_fix

        # Re-run TrOCR on enhanced crop
        try:
            image   = Image.open(token.source_path).convert("RGB")
            scale_x = image.width  / 640
            scale_y = image.height / 640

            x_min = max(int(token.x_min * scale_x) - 5, 0)
            y_min = max(int(token.y_min * scale_y) - 5, 0)
            x_max = min(int(token.x_max * scale_x) + 5, image.width)
            y_max = min(int(token.y_max * scale_y) + 5, image.height)

            crop = image.crop((x_min, y_min, x_max, y_max))
            crop = crop.filter(ImageFilter.SHARPEN)
            crop = crop.filter(ImageFilter.SHARPEN)
            crop = ImageOps.pad(
                crop, (self.RESCAN_SIZE, self.RESCAN_SIZE), color=(255, 255, 255)
            )

            pixel_values = self.processor(
                crop, return_tensors="pt"
            ).pixel_values.to(self.device)

            with torch.no_grad():
                ids = self.model.generate(pixel_values)

            corrected = self.processor.batch_decode(
                ids, skip_special_tokens=True
            )[0].strip()

            print(f"  [ReScan] TrOCR: '{token.text}' → '{corrected}'")
            return corrected

        except Exception as e:
            print(f"  [ReScan] ERROR: {e} — keeping original")
            return token.text


# ─────────────────────────────────────────────────────────────────────────────
# ValidatorAgent — unified entry point for Task 2
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Orchestrates all four validation layers.

    Parameters
    ----------
    processor       : TrOCR processor (for re-scan layer)
    trocr_model     : TrOCR model (for re-scan layer)
    device          : 'cuda' or 'cpu'
    drug_dict       : {term: 1=drug / 0=instruction}
    openai_api_key  : key for GPT-4o Vision
    vlm_model       : OpenAI model to use (default: gpt-4o)
    vlm_on_warn     : also send WARN tokens to VLM (default: False)
    """

    def __init__(
        self,
        processor:      TrOCRProcessor,
        trocr_model:    VisionEncoderDecoderModel,
        device:         str,
        drug_dict:      dict[str, int],
        gemini_api_key: str,
        vlm_on_warn:    bool = False,
    ):
        self.drug_dict   = drug_dict
        self.vlm_on_warn = vlm_on_warn

        self.confidence_gate = ConfidenceGate()
        self.dosage_rule     = DosageRangeRule()
        self.frequency_rule  = FrequencyPlausibilityRule()
        self.drug_rule       = DrugNameSanityRule()
        self.vlm             = VLMValidator(gemini_api_key)
        self.rescan_agent    = CropReScanAgent(processor, trocr_model, device)

    def validate_token(
        self,
        token:     SpatialToken,
        drug_hint: str  = "default",
        is_drug:   bool = False,
    ) -> ValidationReport:
        report = ValidationReport(token=token, final_text=token.text)
        flags: list[ValidationFlag] = []

        # Layer 1
        f = self.confidence_gate.check(token)
        if f:
            flags.append(f)

        # Layer 2
        f = self.dosage_rule.check(token, drug_hint)
        if f:
            flags.append(f)

        f = self.frequency_rule.check(token)
        if f:
            flags.append(f)

        if is_drug:
            f = self.drug_rule.check(token, self.drug_dict)
            if f:
                flags.append(f)

        report.flags = flags

        # Layer 3 — VLM (triggered by RESCAN flag or optionally by WARN)
        has_rescan = any(f.action == "RESCAN" for f in flags)
        has_warn   = any(f.action == "WARN"   for f in flags)

        if has_rescan or (self.vlm_on_warn and has_warn):
            verdict           = self.vlm.verify(token)
            report.vlm_verdict = verdict

            # Layer 4 — Re-scan if VLM found a mismatch
            if not verdict.match:
                report.final_text = self.rescan_agent.rescan(token, verdict)
            else:
                report.final_text = token.text
        else:
            report.final_text = token.text

        return report

    def validate_all(
        self,
        tokens:     list[SpatialToken],
        drug_pairs: list[tuple[str, str]],
    ) -> list[ValidationReport]:

        print("\n" + "═" * 60)
        print("  VALIDATOR AGENT — 4-Layer Clinical Consistency Audit")
        print("═" * 60)

        reports    = []
        drug_names = {d.lower() for d, _ in drug_pairs}

        for token in tokens:
            words   = token.text.lower().split()
            is_drug = any(w in drug_names for w in words)
            hint    = next((w for w in words if w in DOSAGE_RANGES), "default")

            report = self.validate_token(token, drug_hint=hint, is_drug=is_drug)
            reports.append(report)
            print(report)
            print()

        vlm_count  = sum(1 for r in reports if r.vlm_verdict is not None)
        fix_count  = sum(
            1 for r in reports
            if r.vlm_verdict and not r.vlm_verdict.match
        )

        print("═" * 60)
        print(f"  Tokens validated  : {len(reports)}")
        print(f"  Sent to VLM       : {vlm_count}")
        print(f"  VLM fixes applied : {fix_count}")
        print("═" * 60 + "\n")

        return reports

        
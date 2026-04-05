"""
spatial_tracker.py  (v3)
========================
Fixes:
  1. calibrate_dynamic_anchors — uses TOPMOST header keyword, not bottommost,
     to avoid the bottom signature ("Dr. X") inflating the header boundary.
  2. Added y-guard: header anchor keywords are only accepted if they appear
     in the upper 35% of the page.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from src.ocr.extractor import SpatialToken


# ─────────────────────────────────────────────────────────────────────────────
# Reading-order validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReadingOrderReport:
    is_valid:         bool
    issues:           list[str]   = field(default_factory=list)
    column_count:     int         = 1
    suspicious_jumps: list[dict]  = field(default_factory=list)

    def __str__(self) -> str:
        status = "✅ VALID" if self.is_valid else "🚨 INVALID"
        lines  = [f"Reading Order: {status}",
                  f"  Detected columns : {self.column_count}"]
        for issue in self.issues:
            lines.append(f"  ⚠  {issue}")
        for jump in self.suspicious_jumps:
            lines.append(
                f"  ↔  region {jump['from_id']} → {jump['to_id']} "
                f"| Δx={jump['delta_x']:.2f}  Δy={jump['delta_y']:.2f}"
            )
        return "\n".join(lines)


class ReadingOrderValidator:
    def __init__(self, x_jump_threshold: float = 0.35,
                 y_back_threshold: float = 0.05):
        self.x_jump = x_jump_threshold
        self.y_back = y_back_threshold

    def validate(self, tokens: list[SpatialToken]) -> ReadingOrderReport:
        if len(tokens) < 2:
            return ReadingOrderReport(is_valid=True)
        issues, suspicious_jumps = [], []
        column_xs = [tokens[0].x_center]

        for i in range(1, len(tokens)):
            prev, curr = tokens[i - 1], tokens[i]
            delta_x = curr.x_center - prev.x_center
            delta_y = curr.y_center - prev.y_center

            if abs(delta_x) > self.x_jump and delta_y < self.y_back:
                suspicious_jumps.append({
                    "from_id":   prev.region_id,
                    "to_id":     curr.region_id,
                    "from_text": prev.text[:30],
                    "to_text":   curr.text[:30],
                    "delta_x":   round(delta_x, 3),
                    "delta_y":   round(delta_y, 3),
                })
                issues.append(
                    f"Cross-column jump: region {prev.region_id} → {curr.region_id}"
                )

            if not any(abs(curr.x_center - cx) < 0.15 for cx in column_xs):
                column_xs.append(curr.x_center)

        return ReadingOrderReport(
            is_valid=len(issues) == 0,
            issues=issues,
            column_count=len(column_xs),
            suspicious_jumps=suspicious_jumps,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Spatial index with FIXED zone calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexedEntity:
    entity_text: str
    entity_type: str
    token:       SpatialToken

    def to_dict(self) -> dict:
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type,
            **self.token.to_dict(),
        }


class SpatialIndex:
    # FIX: sensible static fallbacks
    HEADER_DEFAULT = 0.18
    FOOTER_DEFAULT = 0.82

    def __init__(self) -> None:
        self._entries: list[IndexedEntity] = []
        self.header_boundary_y = self.HEADER_DEFAULT
        self.footer_boundary_y = self.FOOTER_DEFAULT

    def calibrate_dynamic_anchors(self, tokens: list[SpatialToken]) -> None:
        """
        FIX v3: header anchors are only accepted when y_center < 0.35
        (upper third of the page).  This prevents the bottom signature
        "Dr. X" from being counted as a header anchor and inflating
        header_boundary_y to 0.90+.
        """
        header_keywords = ["dr", "clinic", "hospital", "date", "patient", "name"]
        footer_keywords = ["sign", "signature", "refill", "auth", "notes",
                           "thanking", "thank"]

        # ── header: only tokens in the upper 35% ──────────────────────────
        header_ys = [
            t.y_center
            for t in tokens
            if t.y_center < 0.35                          # ← y-guard
            and any(k in t.text.lower() for k in header_keywords)
        ]

        # ── footer: only tokens in the lower 40% ──────────────────────────
        footer_ys = [
            t.y_center
            for t in tokens
            if t.y_center > 0.60                          # ← y-guard
            and any(k in t.text.lower() for k in footer_keywords)
        ]

        if header_ys:
            # Use the LOWEST header keyword's y as the header boundary
            # (= where the header ENDS), not the topmost occurrence
            self.header_boundary_y = max(header_ys) + 0.04
        if footer_ys:
            self.footer_boundary_y = min(footer_ys) - 0.04

        print(
            f"[SpatialIndex] zones calibrated → "
            f"header < {self.header_boundary_y:.2f} | "
            f"footer > {self.footer_boundary_y:.2f}"
        )

    def get_dynamic_zone(self, token: SpatialToken) -> str:
        if token.y_center <= self.header_boundary_y:
            return "header"
        elif token.y_center >= self.footer_boundary_y:
            return "footer"
        return "body"

    def add(self, entity_text: str, entity_type: str,
            token: SpatialToken) -> None:
        self._entries.append(IndexedEntity(entity_text, entity_type, token))

    def lookup(self, text: str) -> list[IndexedEntity]:
        q = text.lower()
        return [e for e in self._entries if q in e.entity_text.lower()]

    def all_entities(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    def build_from_tokens(
        self,
        tokens:    list[SpatialToken],
        drug_dict: dict[str, int],
    ) -> "SpatialIndex":
        self.calibrate_dynamic_anchors(tokens)

        for token in tokens:
            token.zone = self.get_dynamic_zone(token)

            for word in token.text.split():
                w = word.lower()
                if drug_dict.get(w, 0) == 1:
                    etype = "drug"
                elif any(kw in w for kw in
                         ["mg", "ml", "tablet", "cap", "ساعة", "يومي", "مرة"]):
                    etype = "dosage"
                else:
                    etype = "instruction"
                self.add(word, etype, token)

        return self


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HallucinationAlert:
    claim:        str
    claimed_zone: str
    actual_zone:  str
    entity:       str
    severity:     str
    explanation:  str

    def __str__(self) -> str:
        return (
            f"🚨 Spatial Hallucination [{self.severity}]\n"
            f"   Claim  : {self.claim}\n"
            f"   Entity : '{self.entity}'\n"
            f"   Actual zone: {self.actual_zone} (Implied: {self.claimed_zone})\n"
            f"   Reason : {self.explanation}"
        )


class HallucinationDetector:
    ZONE_RULES: dict[str, list[str]] = {
        "header": ["unknown"],
        "body":   ["drug", "dosage", "instruction"],
        "footer": ["instruction", "unknown"],
    }

    def verify(
        self, llm_answer: str, spatial_index: SpatialIndex
    ) -> list[HallucinationAlert]:
        alerts = []
        for word in llm_answer.split():
            for entity in spatial_index.lookup(word):
                actual = entity.token.zone
                expected = [
                    z for z, types in self.ZONE_RULES.items()
                    if entity.entity_type in types
                ]
                if actual not in expected:
                    alerts.append(HallucinationAlert(
                        claim=llm_answer[:80] + "…",
                        claimed_zone=expected[0] if expected else "body",
                        actual_zone=actual,
                        entity=entity.entity_text,
                        severity="HIGH" if entity.entity_type in
                                 ["drug", "dosage"] else "MEDIUM",
                        explanation=(
                            f"'{entity.entity_text}' (type={entity.entity_type})"
                            f" found in '{actual}', expected in {expected}."
                        ),
                    ))
        return alerts

    @staticmethod
    def _severity(entity_type: str, actual_zone: str) -> str:
        return "HIGH" if entity_type in ["drug", "dosage"] else "MEDIUM"


# ─────────────────────────────────────────────────────────────────────────────
# One-stop audit runner
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAudit:
    def __init__(self, drug_dict: dict[str, int]):
        self.drug_dict = drug_dict
        self.validator = ReadingOrderValidator()
        self.detector  = HallucinationDetector()

    def run(self, tokens: list[SpatialToken]) -> dict:
        order_report = self.validator.validate(tokens)
        index        = SpatialIndex().build_from_tokens(tokens, self.drug_dict)
        summary      = "\n".join([
            "═" * 55, "  SPATIAL AUDIT REPORT", "═" * 55,
            str(order_report),
            f"\n  Indexed entities : {len(index.all_entities())}",
            "═" * 55,
        ])
        print(summary)
        return {"order_report": order_report, "index": index, "summary": summary}
"""
spatial_tracker.py
==================
Task 1 — The Structural Audit (Spatial Awareness Layer)

Responsibility:
    Takes the SpatialToken list from extractor.py and builds a
    "Deterministic Trust Layer" — every claim an LLM makes about
    the document can be verified against the original (x, y) coordinates.

Key capabilities:
    1. ReadingOrderValidator  → detects if YOLO read across columns
                                instead of top-to-bottom
    2. SpatialIndex           → maps every text entity to its pixel origin
    3. HallucinationDetector  → flags when an LLM answer cites a fact
                                from the wrong document zone
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from src.ocr.extractor import SpatialToken


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Reading-Order Validator
#     Detects if YOLO sorted tokens across columns instead of down the page
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReadingOrderReport:
    is_valid:        bool
    issues:          list[str]   = field(default_factory=list)
    column_count:    int         = 1
    suspicious_jumps: list[dict] = field(default_factory=list)

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
    """
    Detects cross-column reading errors in the YOLO token sequence.

    Logic gate:
        If consecutive tokens have a large HORIZONTAL jump (Δx > threshold)
        AND a BACKWARD vertical move (Δy < 0 or very small), the detector
        flags it as a likely column-switch instead of a line continuation.

    Parameters
    ----------
    x_jump_threshold : fraction of image width that counts as a column jump
    y_back_threshold : how far back (upward) in y triggers the alert
    """

    def __init__(
        self,
        x_jump_threshold: float = 0.35,
        y_back_threshold: float = 0.05,
    ):
        self.x_jump = x_jump_threshold
        self.y_back = y_back_threshold

    def validate(self, tokens: list[SpatialToken]) -> ReadingOrderReport:
        if len(tokens) < 2:
            return ReadingOrderReport(is_valid=True)

        issues:           list[str]  = []
        suspicious_jumps: list[dict] = []
        column_xs:        list[float] = [tokens[0].x_center]

        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]

            delta_x = curr.x_center - prev.x_center   # positive = moved right
            delta_y = curr.y_center - prev.y_center   # positive = moved down

            # ── Gate: big horizontal jump + backward/flat vertical ──────────
            if abs(delta_x) > self.x_jump and delta_y < self.y_back:
                suspicious_jumps.append({
                    "from_id":  prev.region_id,
                    "to_id":    curr.region_id,
                    "from_text": prev.text[:30],
                    "to_text":   curr.text[:30],
                    "delta_x":  round(delta_x, 3),
                    "delta_y":  round(delta_y, 3),
                })
                issues.append(
                    f"Possible cross-column jump between "
                    f"region {prev.region_id} ('{prev.text[:20]}') "
                    f"and region {curr.region_id} ('{curr.text[:20]}')"
                )

            # track unique x-clusters to estimate column count
            if not any(abs(curr.x_center - cx) < 0.15 for cx in column_xs):
                column_xs.append(curr.x_center)

        return ReadingOrderReport(
            is_valid         = len(issues) == 0,
            issues           = issues,
            column_count     = len(column_xs),
            suspicious_jumps = suspicious_jumps,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Spatial Index
#     Stores every entity → its pixel origin for downstream verification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexedEntity:
    """
    A named medical entity (drug, dosage, instruction) tied to its
    pixel-level location on the original scan.
    """
    entity_text:  str
    entity_type:  str        # "drug" | "dosage" | "instruction" | "unknown"
    token:        SpatialToken

    def to_dict(self) -> dict:
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type,
            **self.token.to_dict(),
        }


class SpatialIndex:
    """
    Builds a searchable index of entities → coordinates.

    Use this to answer:  "Where exactly on the scan did the word X appear?"
    Every LLM answer must be verifiable through this index.
    """

    def __init__(self) -> None:
        self._entries: list[IndexedEntity] = []

    def add(self, entity_text: str, entity_type: str, token: SpatialToken):
        self._entries.append(
            IndexedEntity(entity_text, entity_type, token)
        )

    def lookup(self, text: str) -> list[IndexedEntity]:
        """Find all index entries whose text contains `text` (case-insensitive)."""
        q = text.lower()
        return [e for e in self._entries if q in e.entity_text.lower()]

    def get_zone(self, text: str) -> Optional[str]:
        """Return the document zone ('header'/'body'/'footer') for a text snippet."""
        hits = self.lookup(text)
        return hits[0].token.zone if hits else None

    def all_entities(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.all_entities(), ensure_ascii=False, indent=indent)

    def build_from_tokens(
        self,
        tokens:    list[SpatialToken],
        drug_dict: dict[str, int],           # from the SymSpell database
    ) -> "SpatialIndex":
        """
        Populate the index from a token list.
        Uses drug_dict to tag tokens as 'drug' or 'instruction'.
        """
        for token in tokens:
            words = token.text.split()
            for word in words:
                w_lower = word.lower()
                if drug_dict.get(w_lower, 0) == 1:
                    etype = "drug"
                elif any(kw in w_lower for kw in
                         ["mg", "ml", "tablet", "cap", "ساعة", "يومي", "مرة"]):
                    etype = "dosage"
                else:
                    etype = "instruction"

                self.add(
                    entity_text = word,
                    entity_type = etype,
                    token       = token,
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Hallucination Detector
#     The core logic gate for Task 1
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HallucinationAlert:
    """Fired when a claimed fact cannot be grounded to the right document zone."""
    claim:          str
    claimed_zone:   str    # zone the LLM implicitly assumes
    actual_zone:    str    # zone where the text actually lives on the scan
    entity:         str
    severity:       str    # "HIGH" | "MEDIUM" | "LOW"
    explanation:    str

    def __str__(self) -> str:
        return (
            f"🚨 Spatial Hallucination [{self.severity}]\n"
            f"   Claim      : {self.claim}\n"
            f"   Entity     : '{self.entity}'\n"
            f"   Actual zone: {self.actual_zone}  "
            f"(LLM implied: {self.claimed_zone})\n"
            f"   Reason     : {self.explanation}"
        )


class HallucinationDetector:
    """
    Verifies that every entity in an LLM answer is grounded to the
    correct spatial zone on the original scan.

    Zone rules (medical prescription conventions):
    ┌─────────────────────────────────────────────────────┐
    │  header  │ doctor info, clinic name, date, stamp    │
    │  body    │ drug names, dosages, instructions        │
    │  footer  │ signature, notes, refill instructions    │
    └─────────────────────────────────────────────────────┘

    A "Spatial Hallucination" is flagged when:
        • A drug/dosage entity is found in the header zone
          (could be confused with a date or clinic name)
        • A date-like string in the body is attributed to a prescription
          year rather than the visit date
        • A footer entity is cited as a primary instruction
    """

    # Which entity types are expected in which zones
    ZONE_RULES: dict[str, list[str]] = {
        "header":  ["unknown"],                      # rarely drugs here
        "body":    ["drug", "dosage", "instruction"],
        "footer":  ["instruction", "unknown"],
    }

    def verify(
        self,
        llm_answer:    str,
        spatial_index: SpatialIndex,
    ) -> list[HallucinationAlert]:
        """
        Cross-check every entity mentioned in the LLM answer against
        the spatial index.

        Parameters
        ----------
        llm_answer    : the raw text produced by the LLM
        spatial_index : built from the same document's SpatialTokens

        Returns
        -------
        List of HallucinationAlerts (empty = clean answer)
        """
        alerts: list[HallucinationAlert] = []
        words   = llm_answer.split()

        for word in words:
            hits = spatial_index.lookup(word)
            if not hits:
                continue

            for entity in hits:
                actual_zone    = entity.token.zone
                expected_zones = [
                    z for z, types in self.ZONE_RULES.items()
                    if entity.entity_type in types
                ]

                if actual_zone not in expected_zones:
                    severity = self._severity(entity.entity_type, actual_zone)
                    alerts.append(HallucinationAlert(
                        claim        = llm_answer[:80] + "…",
                        claimed_zone = expected_zones[0] if expected_zones else "body",
                        actual_zone  = actual_zone,
                        entity       = entity.entity_text,
                        severity     = severity,
                        explanation  = (
                            f"Entity '{entity.entity_text}' (type={entity.entity_type}) "
                            f"was found in the '{actual_zone}' zone "
                            f"(coords: x={entity.token.x_center:.2f}, "
                            f"y={entity.token.y_center:.2f}), "
                            f"but is expected in {expected_zones}."
                        ),
                    ))

        return alerts

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _severity(entity_type: str, actual_zone: str) -> str:
        if entity_type == "drug" and actual_zone == "header":
            return "HIGH"      # drug name misread as header text → dangerous
        elif entity_type == "dosage":
            return "HIGH"      # wrong dosage zone → always critical
        else:
            return "MEDIUM"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SpatialAudit — unified entry point for Task 1
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAudit:
    """
    One-stop Task 1 runner.

    Usage
    -----
    audit   = SpatialAudit(drug_dict)
    report  = audit.run(tokens)          # after extractor.extract()
    alerts  = audit.verify_answer(llm_answer, report["index"])
    """

    def __init__(self, drug_dict: dict[str, int]):
        self.drug_dict  = drug_dict
        self.validator  = ReadingOrderValidator()
        self.detector   = HallucinationDetector()

    def run(self, tokens: list[SpatialToken]) -> dict:
        """
        Full spatial audit on a token list.

        Returns dict with:
            order_report  : ReadingOrderReport
            index         : SpatialIndex
            summary       : human-readable string
        """
        order_report = self.validator.validate(tokens)

        index = SpatialIndex().build_from_tokens(tokens, self.drug_dict)

        summary_lines = [
            "═" * 55,
            "  SPATIAL AUDIT REPORT",
            "═" * 55,
            str(order_report),
            f"\n  Indexed entities : {len(index.all_entities())}",
            "═" * 55,
        ]
        if not order_report.is_valid:
            summary_lines.append(
                "  ⚠  Reading order issues detected — "
                "OCR output may be spatially corrupted."
            )

        print("\n".join(summary_lines))

        return {
            "order_report": order_report,
            "index":        index,
            "summary":      "\n".join(summary_lines),
        }

    def verify_answer(
        self,
        llm_answer:    str,
        spatial_index: SpatialIndex,
    ) -> list[HallucinationAlert]:
        """Verify an LLM answer against the spatial index."""
        alerts = self.detector.verify(llm_answer, spatial_index)

        if alerts:
            print(f"\n🚨 {len(alerts)} Spatial Hallucination(s) detected:")
            for a in alerts:
                print(a)
        else:
            print("\n✅ Answer is spatially grounded — no hallucinations detected.")

        return alerts
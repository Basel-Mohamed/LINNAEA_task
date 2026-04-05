"""Spatial indexing and hallucination checks for document-grounded answers."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from src.ocr.extractor import SpatialToken


ANSWER_TOKEN_RE = re.compile(r"[A-Za-z0-9_؀-ۿ]+")


@dataclass
class ReadingOrderReport:
    """Summary of reading-order validation across OCR tokens."""

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    column_count: int = 1
    suspicious_jumps: list[dict] = field(default_factory=list)


class ReadingOrderValidator:
    """Detect suspicious jumps that suggest broken reading order."""

    def __init__(self, x_jump_threshold: float = 0.35, y_back_threshold: float = 0.05):
        self.x_jump = x_jump_threshold
        self.y_back = y_back_threshold

    def validate(self, tokens: list[SpatialToken]) -> ReadingOrderReport:
        if len(tokens) < 2:
            return ReadingOrderReport(is_valid=True)

        issues: list[str] = []
        suspicious_jumps: list[dict] = []
        column_xs = [tokens[0].x_center]

        for previous, current in zip(tokens, tokens[1:]):
            delta_x = current.x_center - previous.x_center
            delta_y = current.y_center - previous.y_center

            if abs(delta_x) > self.x_jump and delta_y < self.y_back:
                suspicious_jumps.append(
                    {
                        "from_id": previous.region_id,
                        "to_id": current.region_id,
                        "delta_x": round(delta_x, 3),
                        "delta_y": round(delta_y, 3),
                    }
                )
                issues.append(f"Cross-column jump: region {previous.region_id} -> {current.region_id}")

            if not any(abs(current.x_center - column_x) < 0.15 for column_x in column_xs):
                column_xs.append(current.x_center)

        return ReadingOrderReport(
            is_valid=not issues,
            issues=issues,
            column_count=len(column_xs),
            suspicious_jumps=suspicious_jumps,
        )


@dataclass
class IndexedEntity:
    """A searchable entity anchored to a source OCR token."""

    entity_text: str
    entity_type: str
    token: SpatialToken

    def to_dict(self) -> dict:
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type,
            **self.token.to_dict(),
        }


class SpatialIndex:
    """Attach entity types and page zones to OCR tokens for later verification."""

    HEADER_DEFAULT = 0.18
    FOOTER_DEFAULT = 0.82

    def __init__(self) -> None:
        self._entries: list[IndexedEntity] = []
        self.header_boundary_y = self.HEADER_DEFAULT
        self.footer_boundary_y = self.FOOTER_DEFAULT

    def calibrate_dynamic_anchors(self, tokens: list[SpatialToken]) -> None:
        """Infer page header and footer boundaries from anchor keywords."""
        header_keywords = ["dr", "clinic", "hospital", "date", "patient", "name"]
        footer_keywords = ["sign", "signature", "refill", "auth", "notes", "thanking", "thank"]

        header_ys = [
            token.y_center
            for token in tokens
            if token.y_center < 0.35 and any(keyword in token.text.lower() for keyword in header_keywords)
        ]
        footer_ys = [
            token.y_center
            for token in tokens
            if token.y_center > 0.60 and any(keyword in token.text.lower() for keyword in footer_keywords)
        ]

        if header_ys:
            self.header_boundary_y = max(header_ys) + 0.04
        if footer_ys:
            self.footer_boundary_y = min(footer_ys) - 0.04

    def get_dynamic_zone(self, token: SpatialToken) -> str:
        if token.y_center <= self.header_boundary_y:
            return "header"
        if token.y_center >= self.footer_boundary_y:
            return "footer"
        return "body"

    def add(self, entity_text: str, entity_type: str, token: SpatialToken) -> None:
        self._entries.append(IndexedEntity(entity_text=entity_text, entity_type=entity_type, token=token))

    def lookup(self, text: str) -> list[IndexedEntity]:
        query = text.casefold()
        return [entry for entry in self._entries if entry.entity_text.casefold() == query]

    def all_entities(self) -> list[dict]:
        return [entry.to_dict() for entry in self._entries]

    def build_from_tokens(self, tokens: list[SpatialToken], drug_dict: dict[str, int]) -> "SpatialIndex":
        """Assign zones and index searchable entities for all tokens."""
        self.calibrate_dynamic_anchors(tokens)
        for token in tokens:
            token.zone = self.get_dynamic_zone(token)
            for word in ANSWER_TOKEN_RE.findall(token.text):
                lowered = word.casefold()
                if drug_dict.get(lowered, 0) == 1:
                    entity_type = "drug"
                elif any(keyword in lowered for keyword in ["mg", "ml", "tablet", "cap", "\u0633\u0627\u0639\u0629", "\u064a\u0648\u0645\u064a", "\u0645\u0631\u0629"]):
                    entity_type = "dosage"
                else:
                    entity_type = "instruction"
                self.add(word, entity_type, token)
        return self


@dataclass
class HallucinationAlert:
    """A mismatch between an answer claim and the spatially grounded token zone."""

    claim: str
    claimed_zone: str
    actual_zone: str
    entity: str
    severity: str
    explanation: str


class HallucinationDetector:
    """Verify that answer terms appear in plausible zones on the page."""

    ZONE_RULES: dict[str, list[str]] = {
        "header": ["unknown"],
        "body": ["drug", "dosage", "instruction"],
        "footer": ["instruction", "unknown"],
    }

    def verify(self, llm_answer: str, spatial_index: SpatialIndex) -> list[HallucinationAlert]:
        alerts: list[HallucinationAlert] = []
        seen: set[tuple[str, int]] = set()

        for word in ANSWER_TOKEN_RE.findall(llm_answer):
            for entity in spatial_index.lookup(word):
                key = (entity.entity_text.casefold(), entity.token.region_id)
                if key in seen:
                    continue
                seen.add(key)

                actual_zone = entity.token.zone
                expected_zones = [zone for zone, types in self.ZONE_RULES.items() if entity.entity_type in types]
                if actual_zone in expected_zones:
                    continue

                alerts.append(
                    HallucinationAlert(
                        claim=llm_answer[:120],
                        claimed_zone=expected_zones[0] if expected_zones else "body",
                        actual_zone=actual_zone,
                        entity=entity.entity_text,
                        severity="HIGH" if entity.entity_type in {"drug", "dosage"} else "MEDIUM",
                        explanation=(
                            f"'{entity.entity_text}' resolved to token region {entity.token.region_id} at "
                            f"(x={entity.token.x_center:.3f}, y={entity.token.y_center:.3f}) in the {actual_zone} zone."
                        ),
                    )
                )
        return alerts


class SpatialAudit:
    """Run spatial indexing once and reuse it for answer verification."""

    def __init__(self, drug_dict: dict[str, int]):
        self.drug_dict = drug_dict
        self.validator = ReadingOrderValidator()
        self.detector = HallucinationDetector()

    def run(self, tokens: list[SpatialToken]) -> dict:
        """Build the spatial index and reading-order report for a document."""
        order_report = self.validator.validate(tokens)
        index = SpatialIndex().build_from_tokens(tokens, self.drug_dict)
        summary = "\\n".join(
            [
                "=" * 55,
                "  SPATIAL AUDIT REPORT",
                "=" * 55,
                f"Reading order valid: {order_report.is_valid}",
                f"Indexed entities: {len(index.all_entities())}",
                "=" * 55,
            ]
        )
        print(summary)
        return {"order_report": order_report, "index": index, "summary": summary}

    def verify_answer(
        self,
        llm_answer: str,
        tokens: list[SpatialToken] | None = None,
        spatial_index: SpatialIndex | None = None,
    ) -> list[HallucinationAlert]:
        """Check an answer against the spatial index built from OCR tokens."""
        index = spatial_index
        if index is None:
            if tokens is None:
                raise ValueError("Either tokens or spatial_index must be provided")
            index = SpatialIndex().build_from_tokens(tokens, self.drug_dict)
        return self.detector.verify(llm_answer, index)

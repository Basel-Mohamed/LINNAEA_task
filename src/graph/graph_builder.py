from __future__ import annotations

"""Neo4j persistence for document-scoped clinical entities."""

from dataclasses import asdict, dataclass
from typing import Sequence

from neo4j import GraphDatabase

from src.ocr.extractor import SpatialToken


@dataclass(frozen=True)
class ClinicalEntityRecord:
    """Serialized graph payload for one verified clinical entity."""

    text: str
    zone: str
    x_center: float
    y_center: float
    confidence: float
    region_id: int
    source_path: str


class ClinicalGraphBuilder:
    """Persist document-scoped clinical entities into Neo4j."""

    _SCHEMA_STATEMENTS: tuple[str, ...] = (
        "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (r:PolicyRider) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT clinical_entity_scope IF NOT EXISTS FOR (e:ClinicalEntity) REQUIRE (e.text, e.doc_id, e.patient_id) IS UNIQUE",
        "CREATE INDEX clinical_entity_text IF NOT EXISTS FOR (e:ClinicalEntity) ON (e.text)",
        "CREATE INDEX clinical_entity_patient_doc IF NOT EXISTS FOR (e:ClinicalEntity) ON (e.patient_id, e.doc_id)",
        "CREATE INDEX document_patient_lookup IF NOT EXISTS FOR (d:Document) ON (d.patient_id)",
    )

    _UPSERT_DOCUMENT_QUERY = """
    MERGE (p:Patient {id: $patient_id})
    MERGE (d:Document {id: $doc_id})
    SET d.date = $date,
        d.patient_id = $patient_id
    MERGE (p)-[:HAS_RECORD]->(d)
    WITH d
    UNWIND $entities AS entity
    MERGE (e:ClinicalEntity {
        text: entity.text,
        doc_id: $doc_id,
        patient_id: $patient_id
    })
    SET e.type = entity.zone,
        e.region_id = entity.region_id,
        e.source_path = entity.source_path,
        e.updated_at = datetime()
    MERGE (d)-[rel:CONTAINS_ENTITY]->(e)
    SET rel.x_center = entity.x_center,
        rel.y_center = entity.y_center,
        rel.confidence = entity.confidence
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._schema_ready = False

    def close(self) -> None:
        self.driver.close()

    @staticmethod
    def build_entity_payload(tokens: Sequence[SpatialToken]) -> list[dict]:
        """Convert verified OCR tokens into Neo4j-ready entity payloads."""
        payload: list[dict] = []
        for token in tokens:
            text = token.text.strip()
            if not text:
                continue
            entity = ClinicalEntityRecord(
                text=text,
                zone=token.zone,
                x_center=token.x_center,
                y_center=token.y_center,
                confidence=token.confidence,
                region_id=token.region_id,
                source_path=token.source_path,
            )
            payload.append(asdict(entity))
        return payload

    def ensure_schema(self) -> None:
        """Create required constraints and indexes once per builder instance."""
        if self._schema_ready:
            return
        with self.driver.session() as session:
            for statement in self._SCHEMA_STATEMENTS:
                session.run(statement)
        self._schema_ready = True

    def ingest_document(
        self,
        patient_id: str,
        doc_id: str,
        date: str,
        tokens: Sequence[SpatialToken],
    ) -> None:
        """Persist one document and its entities with patient/document scoping."""
        entities = self.build_entity_payload(tokens)
        self.ensure_schema()
        with self.driver.session() as session:
            session.run(
                self._UPSERT_DOCUMENT_QUERY,
                patient_id=patient_id,
                doc_id=doc_id,
                date=date,
                entities=entities,
            )

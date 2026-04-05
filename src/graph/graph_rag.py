from __future__ import annotations

from typing import Any


class SovereignGraphRAG:
    def __init__(self, builder):
        self.builder = builder

    def query_active_medications(self, patient_id: str, doc_id: str | None = None) -> list[dict[str, Any]]:
        query = """
            MATCH (:Patient {id: $patient_id})-[:HAS_RECORD]->(d:Document)
            MATCH (d)-[rel:CONTAINS_ENTITY]->(e:ClinicalEntity {patient_id: $patient_id, doc_id: d.id})
            WHERE e.type = $entity_type
              AND ($doc_id IS NULL OR d.id = $doc_id)
            RETURN e.text AS medication,
                   rel.x_center AS x_center,
                   rel.y_center AS y_center,
                   rel.confidence AS confidence,
                   d.id AS source_doc
            ORDER BY source_doc, y_center, x_center
        """
        with self.builder.driver.session() as session:
            result = session.run(
                query,
                patient_id=patient_id,
                doc_id=doc_id,
                entity_type="body",
            )
            return [record.data() for record in result]

    def query_policy_status(self, patient_id: str) -> list[dict[str, Any]]:
        query = """
            MATCH (p:Patient {id: $patient_id})-[:BOUND_TO_POLICY]->(r:PolicyRider)
            WHERE r.status = $status
            RETURN r.year AS year, r.id AS policy_id
            ORDER BY r.year DESC
        """
        with self.builder.driver.session() as session:
            result = session.run(query, patient_id=patient_id, status="ACTIVE")
            return [record.data() for record in result]

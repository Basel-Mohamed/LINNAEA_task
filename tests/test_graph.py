from __future__ import annotations

from src.graph.graph_builder import ClinicalGraphBuilder
from src.ocr.extractor import SpatialToken


class FakeSession:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeDriver:
    def __init__(self):
        self.sessions: list[FakeSession] = []

    def session(self):
        session = FakeSession()
        self.sessions.append(session)
        return session

    def close(self):
        return None


def _token(text: str) -> SpatialToken:
    return SpatialToken(
        text=text,
        x_min=0,
        y_min=0,
        x_max=10,
        y_max=10,
        x_center=0.5,
        y_center=0.5,
        width_norm=0.1,
        height_norm=0.1,
        confidence=0.9,
        region_id=1,
        source_path="mock.png",
        zone="body",
    )


def test_build_entity_payload_preserves_document_scope_fields():
    payload = ClinicalGraphBuilder.build_entity_payload([_token("amoxicillin")])
    assert payload == [
        {
            "text": "amoxicillin",
            "zone": "body",
            "x_center": 0.5,
            "y_center": 0.5,
            "confidence": 0.9,
            "region_id": 1,
            "source_path": "mock.png",
        }
    ]


def test_ingest_document_uses_patient_and_document_scoped_entity_merge():
    builder = ClinicalGraphBuilder.__new__(ClinicalGraphBuilder)
    builder.driver = FakeDriver()
    builder._schema_ready = False

    builder.ingest_document(
        patient_id="PATIENT_A",
        doc_id="DOC_1",
        date="2026-04-05",
        tokens=[_token("amoxicillin")],
    )

    ingest_query, ingest_params = builder.driver.sessions[-1].calls[0]
    assert "MERGE (e:ClinicalEntity {" in ingest_query
    assert "doc_id: $doc_id" in ingest_query
    assert "patient_id: $patient_id" in ingest_query
    assert ingest_params["patient_id"] == "PATIENT_A"
    assert ingest_params["doc_id"] == "DOC_1"


def test_same_text_isolation_depends_on_doc_and_patient_ids():
    builder = ClinicalGraphBuilder.__new__(ClinicalGraphBuilder)
    builder.driver = FakeDriver()
    builder._schema_ready = True

    builder.ingest_document("PATIENT_A", "DOC_1", "2026-04-05", [_token("metformin")])
    builder.ingest_document("PATIENT_B", "DOC_2", "2026-04-05", [_token("metformin")])

    first_params = builder.driver.sessions[0].calls[0][1]
    second_params = builder.driver.sessions[1].calls[0][1]
    assert first_params["entities"][0]["text"] == second_params["entities"][0]["text"]
    assert first_params["patient_id"] != second_params["patient_id"]
    assert first_params["doc_id"] != second_params["doc_id"]

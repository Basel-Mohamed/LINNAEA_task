from __future__ import annotations

import asyncio
import os
from pathlib import Path

import cv2
from dotenv import load_dotenv

from src.drug_dictionary import load_drug_dictionary_from_excel
from src.graph.graph_builder import ClinicalGraphBuilder
from src.graph.graph_rag import SovereignGraphRAG
from src.ocr.extractor import PrescriptionExtractor
from src.ocr.spatial_tracker import SpatialAudit
from src.ocr.validator import ValidatorAgent

load_dotenv()


def visualize_extractions(image_path: str, tokens: list, output_path: str = "visualized_output.jpg") -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    for token in tokens:
        cv2.rectangle(image, (token.x_min, token.y_min), (token.x_max, token.y_max), (0, 255, 0), 2)
        y_text = max(token.y_min - 5, 15)
        cv2.putText(image, token.text, (token.x_min, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")


def render_active_medications_answer(records: list[dict]) -> str:
    if not records:
        return "No active medications were found for this patient."
    medications = ", ".join(record["medication"] for record in records)
    return f"Active medications documented in the body of the record: {medications}."


def main() -> None:
    print("Initializing LINNAEA pipeline\n")

    project_root = Path(__file__).resolve().parent
    drug_db_path = project_root / "data" / "database.xlsx"
    drug_dict = load_drug_dictionary_from_excel(drug_db_path)
    print(f"Loaded {len(drug_dict)} entries from {drug_db_path.name}")

    patient_id = "PATIENT_8847"
    doc_id = "DOC_1985_A"
    doc_date = "1985-10-12"

    test_image_path = "data/raw/sample_prescription_1985.jpg"
    if not os.path.exists(test_image_path):
        print(f"Image not found: {test_image_path}")
        return

    extractor = PrescriptionExtractor(
        hf_token=os.getenv("HF_TOKEN"),
        trocr_model="microsoft/trocr-large-handwritten",
    )
    audit_agent = SpatialAudit(drug_dict=drug_dict)
    validator_agent = ValidatorAgent(
        processor=extractor.processor,
        trocr_model=extractor.trocr,
        device=extractor.device,
        drug_dict=drug_dict,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        vlm_on_warn=True,
    )
    graph_builder = ClinicalGraphBuilder(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    print("--- Phase 1: Extraction ---")
    raw_tokens_list = extractor.extract([test_image_path])
    raw_tokens = raw_tokens_list[0] if raw_tokens_list else []
    if not raw_tokens:
        print("No text detected. Aborting.")
        return
    visualize_extractions(test_image_path, raw_tokens, "output_phase1_ocr.jpg")

    print("--- Phase 2: Spatial Audit ---")
    audit_agent.run(raw_tokens)

    print("--- Phase 3: Validation ---")
    drug_pairs = [(key, str(value)) for key, value in drug_dict.items()]
    validated_reports = asyncio.run(validator_agent.validate_all_async(raw_tokens, drug_pairs))

    verified_tokens = []
    for report in validated_reports:
        report.token.text = report.final_text
        verified_tokens.append(report.token)
    visualize_extractions(test_image_path, verified_tokens, "output_phase3_verified.jpg")

    verified_audit = audit_agent.run(verified_tokens)

    print("--- Phase 4: Graph Ingestion ---")
    try:
        graph_builder.ingest_document(
            patient_id=patient_id,
            doc_id=doc_id,
            date=doc_date,
            tokens=verified_tokens,
        )
        rag = SovereignGraphRAG(graph_builder)
        answer = render_active_medications_answer(rag.query_active_medications(patient_id=patient_id, doc_id=doc_id))
        print(f"Graph answer: {answer}")

        hallucination_alerts = audit_agent.verify_answer(answer, spatial_index=verified_audit["index"])
        if hallucination_alerts:
            raise RuntimeError(
                "Spatial hallucination detected: "
                + "; ".join(alert.explanation for alert in hallucination_alerts)
            )
    except Exception as exc:
        print(f"Graph stage bypassed: {exc}")
    finally:
        graph_builder.close()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()

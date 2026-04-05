"""
main.py  (v2 — Groq backend)
=============================
LINNAEΛ Dark Data Processing Pipeline — End-to-End Orchestrator
"""

import os
import asyncio
import cv2
from dotenv import load_dotenv

from src.ocr.extractor import PrescriptionExtractor
from src.ocr.spatial_tracker import SpatialAudit
from src.ocr.validator import ValidatorAgent
from src.graph.graph_builder import ClinicalGraphBuilder

load_dotenv()


def visualize_extractions(image_path: str, tokens: list,
                           output_path: str = "visualized_output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️  Could not read image: {image_path}")
        return

    for token in tokens:
        cv2.rectangle(img,
                      (token.x_min, token.y_min),
                      (token.x_max, token.y_max),
                      (0, 255, 0), 2)
        y_text = max(token.y_min - 5, 15)
        cv2.putText(img, token.text,
                    (token.x_min, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"👁️  Saved: {output_path}")


def main():
    print("🚀 Initializing LINNAEΛ Sovereign Infrastructure Pipeline…\n")

    drug_dict = {
        "amoxicillin": 1, "paracetamol": 1,
        "omeprazole":  1, "hydrocortisone": 1,
        "tablet": 0,      "daily": 0,
    }

    test_image_path = "data/raw/sample_prescription_1985.jpg"
    if not os.path.exists(test_image_path):
        print(f"⚠️  Image not found: {test_image_path}")
        return

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading Extractor (YOLO + TrOCR)…")
    extractor = PrescriptionExtractor(
        hf_token=os.getenv("HF_TOKEN"),
        yolo_model="RoyRud1902/yolo11n-text",
        trocr_model="microsoft/trocr-large-handwritten",
    )

    audit_agent = SpatialAudit(drug_dict=drug_dict)

    print("Loading VLM Validator (Groq — Llama-4-Scout)…")
    validator_agent = ValidatorAgent(
        processor=extractor.processor,
        trocr_model=extractor.trocr,
        device=extractor.device,
        drug_dict=drug_dict,
        groq_api_key=os.getenv("GROQ_API_KEY"),   
        vlm_on_warn=True,
    )

    print("Connecting to Neo4j…")
    graph_builder = ClinicalGraphBuilder(
        uri=os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER",    "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    # ── pipeline ──────────────────────────────────────────────────────────────
    print("\n--- Phase 1: Extraction & Spatial Mapping ---")
    raw_tokens_list = extractor.extract([test_image_path])
    raw_tokens      = raw_tokens_list[0] if raw_tokens_list else []

    if not raw_tokens:
        print("❌ No text detected. Aborting.")
        return

    visualize_extractions(test_image_path, raw_tokens, "output_phase1_ocr.jpg")

    print("\n--- Phase 2: Structural Audit (Dynamic Anchors) ---")
    audit_report = audit_agent.run(raw_tokens)

    print("\n--- Phase 3: VLM Validation (Groq — Async) ---")
    drug_pairs       = [(k, str(v)) for k, v in drug_dict.items()]
    validated_reports = asyncio.run(
        validator_agent.validate_all_async(raw_tokens, drug_pairs)
    )

    verified_tokens = []
    for rep in validated_reports:
        rep.token.text = rep.final_text
        verified_tokens.append(rep.token)

    visualize_extractions(test_image_path, verified_tokens, "output_phase3_verified.jpg")

    print("\n--- Phase 4: Graph Ingestion ---")
    try:
        graph_builder.ingest_document(
            patient_id="PATIENT_8847",
            doc_id="DOC_1985_A",
            date="1985-10-12",
            tokens=verified_tokens,
        )
        print("✅ Ingested into Neo4j.")
    except Exception as e:
        print(f"⚠️  Graph ingestion bypassed: {e}")
    finally:
        graph_builder.close()

    print("\n🏁 Pipeline complete.")


if __name__ == "__main__":
    main()
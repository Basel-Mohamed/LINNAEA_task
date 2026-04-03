"""
main.py
=======
LINNAEΛ Dark Data Processing Pipeline
End-to-End Orchestrator

This script simulates the ingestion of a faded 1985 medical record, 
running it through the Extractor, Spatial Audit, VLM Validator, and Graph Builder.
"""

import os
from dotenv import load_dotenv

# Import our custom modules
from src.ocr.extractor import PrescriptionExtractor
from src.ocr.spatial_tracker import SpatialAudit
from src.ocr.validator import ValidatorAgent
from src.graph.graph_builder import ClinicalGraphBuilder

# Load environment variables (OpenAI Key, Neo4j credentials, HF Token)
load_dotenv()

def main():
    print("🚀 Initializing LINNAEΛ Sovereign Infrastructure Pipeline...\n")

    # 1. Initialize Configuration & Mock Database
    # In a real scenario, drug_dict is loaded from database.xlsx
    drug_dict = {
        "amoxicillin": 1,
        "paracetamol": 1,
        "omeprazole": 1,
        "hydrocortisone": 1,
        "mupirocin": 1,
        "tablet": 0,
        "daily": 0
    }
    
    # Define test parameters
    test_image_path = "data/raw/sample_prescription_1985.jpg" # Replace with an actual test image
    patient_id = "PATIENT_8847"
    doc_id = "DOC_1985_A"
    
    # Ensure file exists before proceeding
    if not os.path.exists(test_image_path):
        print(f"⚠️ Test image not found at {test_image_path}. Please place a sample image there to run the pipeline.")
        return

    # 2. Initialize Agents
    print("Loading Extractor Models (YOLO + TrOCR)...")
    extractor = PrescriptionExtractor(hf_token=os.getenv("HF_TOKEN"))
    
    audit_agent = SpatialAudit(drug_dict=drug_dict)
    
    print("Loading VLM Validator Agent (Gemini 2.5 Flash via AI Studio)...")
    validator_agent = ValidatorAgent(
        processor=extractor.processor,
        trocr_model=extractor.trocr,
        device=extractor.device,
        drug_dict=drug_dict,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        vlm_on_warn=True
    )
    
    print("Connecting to Knowledge Graph (Neo4j)...")
    graph_builder = ClinicalGraphBuilder(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )

    # ---------------------------------------------------------
    # THE PIPELINE EXECUTION
    # ---------------------------------------------------------
    
    print(f"\n--- Phase 1: Extraction & Spatial Mapping ---")
    # Extract spatial tokens from the image
    raw_tokens_list = extractor.extract([test_image_path])
    raw_tokens = raw_tokens_list[0] if raw_tokens_list else []
    
    if not raw_tokens:
        print("❌ No text detected. Aborting pipeline.")
        return

    print(f"\n--- Phase 2: The Structural Audit (Task 1) ---")
    # Verify reading order and build the spatial index
    audit_report = audit_agent.run(raw_tokens)
    
    print(f"\n--- Phase 3: The VLLM Validation Engine (Task 2) ---")
    # Run tokens through clinical rules and the VLM "Mass Spectrometer"
    drug_pairs = [(k, str(v)) for k, v in drug_dict.items()]
    validated_reports = validator_agent.validate_all(raw_tokens, drug_pairs)
    
    # Extract the final, verified text tokens
    verified_tokens = []
    for rep in validated_reports:
        # Update the token's text with the VLM/ReScan corrected text
        rep.token.text = rep.final_text 
        verified_tokens.append(rep.token)

    print(f"\n--- Phase 4: Graph Ingestion (Task 3) ---")
    # Bind the verified spatial tokens into the Knowledge Graph
    try:
        graph_builder.ingest_document(
            patient_id=patient_id, 
            doc_id=doc_id, 
            date="1985-10-12", 
            tokens=verified_tokens
        )
        print("✅ Successfully ingested document into Neo4j Knowledge Graph.")
    except Exception as e:
        print(f"⚠️ Graph ingestion bypassed (Neo4j not running): {e}")
    finally:
        graph_builder.close()

    print("\n🏁 Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
## LINNAEA Task

Medical document AI pipeline for extracting, validating, grounding, and querying clinical information from scanned records.

This project is built around one core idea: OCR output alone is not trustworthy enough for medical records. The pipeline combines deterministic checks, blind visual validation, spatial grounding, and graph persistence so extracted data can be traced back to a specific document region before it is used downstream.

## What It Does

- Detects text regions with EasyOCR detection
- Merges nearby detections into line-level crops
- Recognizes each crop with TrOCR
- Audits spatial layout and reading order
- Applies rule-based OCR validation
- Uses a VLM for blind crop rereading and mismatch detection
- Stores verified entities in Neo4j with patient/document scoping
- Runs graph-grounded retrieval for downstream answers
- Checks generated answers against spatial grounding rules
- Includes a minimal prompt-optimization loop for extraction prompts

## Pipeline

1. `EasyOCR` detects candidate text boxes.
2. `LineMerger` groups adjacent boxes into line-level regions.
3. `TrOCR` reads each merged crop.
4. `SpatialAudit` assigns page zones and builds a searchable spatial index.
5. `ValidatorAgent` applies deterministic rules such as confidence, dosage, garbage-text, and frequency checks.
6. `VLMValidator` performs a blind reread of suspicious crops and compares that result to OCR.
7. `ClinicalGraphBuilder` ingests verified entities into Neo4j using `(text, patient_id, doc_id)` scoping.
8. `SovereignGraphRAG` queries graph-backed entities only from the target patient/document context.
9. `HallucinationDetector` validates the final answer against token coordinates and page zones.

## Why This Design

The repository is intentionally not "OCR only."

- Medical records are high-risk: a wrong drug or dosage is not an acceptable silent failure.
- OCR alone is brittle on low-quality scans, handwriting, stamps, and multilingual forms.
- Pure text extraction can leak information across records if graph entities are not document-scoped.
- LLM or VLM answers need grounding checks before they can be trusted.

The codebase therefore favors:

- explicit data contracts
- document-scoped graph integrity
- blind visual cross-checking
- deterministic validation before correction
- testable helpers around geometry and graph ingestion

## Current Stack

### OCR

- Detection: `easyocr`
- Recognition: `microsoft/trocr-large-handwritten`
- Image preprocessing: `Pillow`, `OpenCV`

### Validation

- Deterministic OCR rules
- Blind VLM verification via `groq`
- Spatial hallucination checks

### Storage and Retrieval

- Graph database: `Neo4j`
- Query layer: graph-grounded retrieval with document/patient scoping

### Prompt Optimization

- Minimal AdalFlow-compatible optimization wrapper

## Repository Layout

```text
LINNAEA_task/
|-- data/
|   `-- database.xlsx
|-- src/
|   |-- drug_dictionary.py
|   |-- graph/
|   |   |-- graph_builder.py
|   |   |-- graph_rag.py
|   |   `-- schema.cypher
|   |-- ocr/
|   |   |-- extractor.py
|   |   |-- spatial_tracker.py
|   |   |-- validator.py
|   |   `-- vision_utils.py
|   `-- prompt/
|       |-- evaluator.py
|       `-- optimizer.py
|-- tests/
|   |-- test_graph.py
|   |-- test_ocr.py
|   `-- test_validator.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Core Modules

### [`src/ocr/extractor.py`](./src/ocr/extractor.py)

- `PrescriptionExtractor`
- EasyOCR detection plus TrOCR recognition
- line-level crop generation
- startup warning suppression for harmless TrOCR pooler noise

### [`src/ocr/validator.py`](./src/ocr/validator.py)

- rule-based OCR validation
- Arabic-safe garbage text handling
- blind VLM reread and OCR comparison
- crop rescanning logic

### [`src/ocr/spatial_tracker.py`](./src/ocr/spatial_tracker.py)

- reading-order validation
- zone detection
- spatial entity index
- hallucination detection against grounded tokens

### [`src/graph/graph_builder.py`](./src/graph/graph_builder.py)

- Neo4j schema bootstrapping
- document-scoped entity ingestion
- indexes and constraints for query safety and performance

### [`src/graph/graph_rag.py`](./src/graph/graph_rag.py)

- patient/document-scoped graph queries

### [`src/prompt/optimizer.py`](./src/prompt/optimizer.py)

- minimal prompt optimization loop
- safe fallback when AdalFlow imports are unavailable

### [`src/drug_dictionary.py`](./src/drug_dictionary.py)

- loads normalized token-to-label drug mappings from physical Excel storage
- strict schema enforcement for required properties
- automatic type safety conversions

## Setup

### 1. Start Neo4j Database

You will need a running Neo4j instance for the knowledge graph stages. The easiest method is via Docker:

```powershell
docker run -d --name neo4j-linnaea -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

*(Allow 10-15 seconds for the database to fully initialize).*

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Prepare environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Notes:

- `HF_TOKEN` may be needed for model download access.
- `GROQ_API_KEY` is required only if you want VLM validation enabled.

## Running The Pipeline

The current demo entrypoint is:

```powershell
python main.py
```

By default it expects:

```text
data/raw/sample_prescription_1985.jpg
```

The main stages are:

- extraction
- spatial audit
- validation
- graph ingestion
- graph-grounded answer generation
- spatial hallucination enforcement

The pipeline also writes visual debug images:

- `output_phase1_ocr.jpg`
- `output_phase3_verified.jpg`

## Testing

Run the focused regression suite:

```powershell
.\venv\Scripts\python -m pytest tests\test_graph.py tests\test_ocr.py tests\test_validator.py
```

What is covered:

- graph entity scoping and ingestion payloads
- OCR geometry and line-merging helpers
- validator rules including Arabic-safe handling

## Data Integrity Guarantees

The repository now protects against a critical class of graph contamination bugs.

Clinical entities are not merged globally by text alone. Instead, graph nodes are scoped to:

- `text`
- `patient_id`
- `doc_id`

This prevents different patients with the same medication text from collapsing onto the same node.

## Validation Guarantees

### Blind VLM validation

The VLM is not shown the OCR text before reading the crop. It first performs an independent crop read, then the system compares that output to OCR using normalized edit distance.

### Spatial answer checks

Graph answers are verified against the spatial index after answer generation. If the answer contradicts grounded page zones, the pipeline raises a hallucination error instead of silently accepting the result.

## Known Limitations

- `main.py` is a single-sample demo entrypoint, not a batch CLI yet
- EasyOCR detection confidence is currently normalized to `1.0` because the detect API returns boxes, not per-box scores in this path
- Neo4j usage is optional at runtime but required for full graph stages
- the prompt optimizer is intentionally minimal and not yet wired into a full training workflow
- model downloads can be large and may require credentials on first run
- `easyocr` must be installed in the active environment before first run

## Recommended Next Steps

- add a proper config layer instead of hardcoded demo values in `main.py`
- expose batch processing and CLI arguments
- add integration tests for the end-to-end pipeline with mocks
- persist validation reports as structured artifacts
- add better multilingual medication dictionaries and domain rules

## Status

This codebase is now structured around production-minded safeguards:

- document-scoped graph storage
- blind VLM verification
- spatially grounded answer checks
- targeted regression tests

It is a solid foundation for a medical document extraction system, with the biggest remaining work being packaging, configuration, and broader integration coverage rather than architectural correction.

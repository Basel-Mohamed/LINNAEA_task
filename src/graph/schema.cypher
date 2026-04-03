// ─────────────────────────────────────────────────────────────────────────────
// schema.cypher
// Task 3: The "Metabolic Pathway" (Sovereign Truth Schema)
// ─────────────────────────────────────────────────────────────────────────────

// Clear existing constraints
DROP CONSTRAINT patient_id IF EXISTS;
DROP CONSTRAINT doc_id IF EXISTS;

// Create Constraints for Deterministic Indexing
CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (r:PolicyRider) REQUIRE r.id IS UNIQUE;

// Note: In Neo4j, we establish relationships.
// Example Paths we want to traverse:
// (Patient) -[:HAS_RECORD]-> (Document)
// (Document) -[:CONTAINS_ENTITY {x, y, confidence}]-> (ClinicalEntity)
// (Patient) -[:BOUND_TO_POLICY]-> (PolicyRider)
// (Document) -[:SUPERSEDED_BY]-> (Document)
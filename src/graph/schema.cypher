CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (r:PolicyRider) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT clinical_entity_scope IF NOT EXISTS
FOR (e:ClinicalEntity)
REQUIRE (e.text, e.doc_id, e.patient_id) IS UNIQUE;

CREATE INDEX clinical_entity_text IF NOT EXISTS
FOR (e:ClinicalEntity)
ON (e.text);

CREATE INDEX clinical_entity_patient_doc IF NOT EXISTS
FOR (e:ClinicalEntity)
ON (e.patient_id, e.doc_id);

CREATE INDEX document_patient_lookup IF NOT EXISTS
FOR (d:Document)
ON (d.patient_id);

"""
graph_rag.py
============
Task 3 — The Deterministic Query

Why a Graph prevents the "2018 Rider" hallucination:
A flat vector search retrieves chunks based on cosine similarity. If the query 
mentions a "Rider", the DB returns the 2018 chunk, and the LLM assumes it applies.
In Graph RAG, we only traverse edges that are 'ACTIVE'.
"""

class SovereignGraphRAG:
    def __init__(self, builder):
        self.builder = builder

    def query_active_medications(self, patient_id: str):
        """
        Only returns medications tied to the patient's documents, 
        and extracts exactly WHERE they were on the page.
        """
        query = """
            MATCH (p:Patient {id: $patient_id})-[:HAS_RECORD]->(d:Document)
            MATCH (d)-[rel:CONTAINS_ENTITY]->(e:ClinicalEntity)
            WHERE e.type = 'body'
            RETURN e.text AS Medication, rel.x_center AS X, rel.y_center AS Y, d.id AS SourceDoc
        """
        with self.builder.driver.session() as session:
            result = session.run(query, patient_id=patient_id)
            return [record.data() for record in result]

    def query_policy_status(self, patient_id: str):
        """
        Prevents hallucinating an obsolete 2018 rider into a 2026 claim.
        """
        query = """
            MATCH (p:Patient {id: $patient_id})-[:SUBJECT_TO]->(r:PolicyRider)
            WHERE r.status = 'ACTIVE'
            RETURN r.year, r.id
        """
        with self.builder.driver.session() as session:
            result = session.run(query, patient_id=patient_id)
            return [record.data() for record in result]
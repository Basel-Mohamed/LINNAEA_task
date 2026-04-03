"""
graph_builder.py
================
Task 3 — Translating Spatial Tokens into Graph Edges

Converts the verified `SpatialToken`s into a Neo4j Knowledge Graph.
This prevents the "hallucination" of Vector DBs by creating hard physical bonds.
"""

from neo4j import GraphDatabase
import sys
import os

# Append src to path to import extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocr.extractor import SpatialToken

class ClinicalGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ingest_document(self, patient_id: str, doc_id: str, date: str, tokens: list[SpatialToken]):
        """
        Creates the 'Reaction Map'. 
        Bonds a Patient to a Document, and the Document to its verified Entities.
        """
        with self.driver.session() as session:
            # 1. Create Patient and Document Nodes
            session.run("""
                MERGE (p:Patient {id: $patient_id})
                MERGE (d:Document {id: $doc_id, date: $date})
                MERGE (p)-[:HAS_RECORD]->(d)
            """, patient_id=patient_id, doc_id=doc_id, date=date)

            # 2. Attach Spatial Entities with coordinates as Edge Properties
            for token in tokens:
                # We only ingest validated, spatially aware tokens
                session.run("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (e:ClinicalEntity {text: $text, type: $zone})
                    MERGE (d)-[rel:CONTAINS_ENTITY {
                        x_center: $x, 
                        y_center: $y, 
                        confidence: $conf
                    }]->(e)
                """, doc_id=doc_id, text=token.text, zone=token.zone, 
                     x=token.x_center, y=token.y_center, conf=token.confidence)
                
    def link_policy_rider(self, patient_id: str, rider_id: str, year: str, status: str):
        """
        Demonstrates the solution to the "2018 Rider" problem.
        The graph explicitly labels the Rider as OBSOLETE or ACTIVE.
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (p:Patient {id: $patient_id})
                MERGE (r:PolicyRider {id: $rider_id, year: $year, status: $status})
                MERGE (p)-[:SUBJECT_TO]->(r)
            """, patient_id=patient_id, rider_id=rider_id, year=year, status=status)
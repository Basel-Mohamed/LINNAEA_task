"""
evaluator.py
============
Task 4 — The "Loss Function" for Prompt Evolution
"""

class ClinicalExtractionEvaluator:
    """
    Evaluates how well an LLM prompt extracted data from OCR text.
    In AdalFlow, this acts as our loss function.
    """
    
    def __init__(self, exact_match_weight=0.6, clinical_safety_weight=0.4):
        self.em_weight = exact_match_weight
        self.safety_weight = clinical_safety_weight

    def evaluate(self, prediction: dict, ground_truth: dict) -> float:
        """
        Calculates a score between 0.0 and 1.0.
        prediction: {"drug": "Amoxicillin", "dosage": "500mg"}
        """
        score = 0.0
        
        # 1. Exact Match (Stereochemical fit)
        if prediction.get("drug", "").lower() == ground_truth.get("drug", "").lower():
            score += self.em_weight
            
        # 2. Clinical Safety (Is the dosage hallucinated?)
        if prediction.get("dosage") == ground_truth.get("dosage"):
            score += self.safety_weight
            
        return score

    def compute_textual_gradient(self, prediction: dict, ground_truth: dict, prompt: str) -> str:
        """
        If the model fails, we generate a Textual Gradient (Feedback).
        This tells AdalFlow's optimizer *why* the prompt failed so it can mutate it.
        """
        if self.evaluate(prediction, ground_truth) == 1.0:
            return "Perfect extraction."
            
        feedback = "The prompt failed because: "
        if prediction.get("drug") != ground_truth.get("drug"):
            feedback += f"It confused the drug name. Extracted {prediction.get('drug')} instead of {ground_truth.get('drug')}. "
        if prediction.get("dosage") != ground_truth.get("dosage"):
            feedback += "It failed to parse the dosage string properly, likely getting confused by a handwritten medical stamp overlapping the text."
            
        return feedback + "\nSuggestion: Instruct the model to explicitly ignore capitalized bold STAMPS (like 'CANCELLED') and focus on the handwritten ink layer."
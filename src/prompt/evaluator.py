from __future__ import annotations


class ClinicalExtractionEvaluator:
    def __init__(self, exact_match_weight: float = 0.6, clinical_safety_weight: float = 0.4):
        self.em_weight = exact_match_weight
        self.safety_weight = clinical_safety_weight

    def evaluate(self, prediction: dict, ground_truth: dict) -> float:
        score = 0.0
        if prediction.get("drug", "").strip().lower() == ground_truth.get("drug", "").strip().lower():
            score += self.em_weight
        if prediction.get("dosage", "").strip() == ground_truth.get("dosage", "").strip():
            score += self.safety_weight
        return score

    def compute_textual_gradient(self, prediction: dict, ground_truth: dict, prompt: str) -> str:
        if self.evaluate(prediction, ground_truth) == 1.0:
            return "Perfect extraction."

        issues: list[str] = []
        if prediction.get("drug") != ground_truth.get("drug"):
            issues.append(
                f"drug mismatch: extracted '{prediction.get('drug')}' instead of '{ground_truth.get('drug')}'"
            )
        if prediction.get("dosage") != ground_truth.get("dosage"):
            issues.append(
                f"dosage mismatch: extracted '{prediction.get('dosage')}' instead of '{ground_truth.get('dosage')}'"
            )

        guidance = (
            "Revise the prompt to separate handwritten clinical content from stamps and force field-by-field extraction."
        )
        return f"""Prompt: {prompt}
Failure analysis: {'; '.join(issues)}
Suggested revision: {guidance}"""

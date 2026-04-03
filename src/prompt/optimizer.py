"""
optimizer.py
============
Task 4 — AdalFlow Optimization Pipeline

How to ensure it evolved and didn't just get lucky (Overfitting):
We use Cross-Validation across vastly different subsets of "Dark Data" 
(e.g., a batch of 1985 blurry carbons, and a batch of 2010 scanned faxes).
If the textual gradient improves the score on ALL batches, the prompt has generalized.
"""

from adalflow.optim import TextualGradientDescent
from evaluator import ClinicalExtractionEvaluator

class PromptCatalyst:
    def __init__(self, initial_prompt: str):
        self.current_prompt = initial_prompt
        self.evaluator = ClinicalExtractionEvaluator()
        
    def optimize_step(self, sample_inputs: list, predictions: list, ground_truths: list):
        """
        Simulates a step of AdalFlow's Textual Gradient Descent.
        """
        gradients = []
        total_loss = 0
        
        for pred, truth in zip(predictions, ground_truths):
            score = self.evaluator.evaluate(pred, truth)
            loss = 1.0 - score
            total_loss += loss
            
            if loss > 0:
                grad = self.evaluator.compute_textual_gradient(pred, truth, self.current_prompt)
                gradients.append(grad)
                
        # In a full AdalFlow implementation, the LLM takes `gradients` and rewrites `self.current_prompt`
        print(f"Batch Loss: {total_loss / len(sample_inputs)}")
        print("Aggregated Textual Gradients for LLM Optimizer:")
        for g in gradients:
            print("-", g)
            
        # Return gradients to be passed to the LLM Meta-Optimizer
        return gradients
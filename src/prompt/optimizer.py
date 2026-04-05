"""Minimal prompt optimization loop compatible with AdalFlow-style components."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from src.prompt.evaluator import ClinicalExtractionEvaluator
from adalflow.optim import Parameter, Trainer
from adalflow.optim.grad_component import GradComponent
from adalflow.optim.types import ParameterType
ADALFLOW_IMPORT_ERROR = None


@dataclass
class OptimizationStepResult:
    """Result of one prompt optimization step over a batch."""

    average_score: float
    gradients: list[str] = field(default_factory=list)
    updated_prompt: str = ""
    used_adalflow: bool = False


if Parameter is not None and ParameterType is not None:
    class PromptGradientComponent(GradComponent):
        """Thin AdalFlow component wrapper around the extraction evaluator."""

        def __init__(self, evaluator: ClinicalExtractionEvaluator):
            super().__init__(desc="Evaluate extraction quality for prompt optimization")
            self.evaluator = evaluator

        def call(self, prediction: dict, ground_truth: dict) -> float:
            return self.evaluator.evaluate(prediction, ground_truth)
else:  # pragma: no cover - fallback type for broken environments
    class PromptGradientComponent:  # type: ignore[no-redef]
        """Fallback evaluator wrapper when AdalFlow is unavailable."""

        def __init__(self, evaluator: ClinicalExtractionEvaluator):
            self.evaluator = evaluator

        def call(self, prediction: dict, ground_truth: dict) -> float:
            return self.evaluator.evaluate(prediction, ground_truth)


class PromptCatalyst:
    """Maintain and iteratively refine the active extraction prompt."""

    def __init__(self, initial_prompt: str, evaluator: ClinicalExtractionEvaluator | None = None):
        self.current_prompt = initial_prompt
        self.evaluator = evaluator or ClinicalExtractionEvaluator()
        self.grad_component = PromptGradientComponent(self.evaluator)
        self.trainer = Trainer if Trainer is not None else None
        self.prompt_parameter = self._build_prompt_parameter(initial_prompt)

    @property
    def adalflow_available(self) -> bool:
        return self.prompt_parameter is not None and ADALFLOW_IMPORT_ERROR is None

    def _build_prompt_parameter(self, prompt: str):
        """Create the AdalFlow prompt parameter when the dependency is available."""
        if Parameter is None or ParameterType is None:
            return None
        return Parameter(
            name="clinical_extraction_prompt",
            data=prompt,
            requires_opt=True,
            role_desc="Clinical extraction system prompt",
            param_type=ParameterType.PROMPT,
        )

    def optimize_step(
        self,
        sample_inputs: list[Any],
        predictions: list[dict],
        ground_truths: list[dict],
    ) -> OptimizationStepResult:
        """Score a batch, collect gradients, and update the active prompt."""
        if not sample_inputs:
            raise ValueError("sample_inputs must not be empty")
        if not (len(sample_inputs) == len(predictions) == len(ground_truths)):
            raise ValueError("sample_inputs, predictions, and ground_truths must have the same length")

        gradients: list[str] = []
        total_score = 0.0
        for prediction, ground_truth in zip(predictions, ground_truths):
            score = self.grad_component.call(prediction, ground_truth)
            total_score += score
            if score < 1.0:
                gradients.append(
                    self.evaluator.compute_textual_gradient(prediction, ground_truth, self.current_prompt)
                )

        average_score = total_score / len(sample_inputs)
        updated_prompt = self.current_prompt
        if gradients:
            gradient_block = "\\n".join(f"- {gradient}" for gradient in gradients)
            updated_prompt = (
                f"{self.current_prompt.strip()}\\n\\n"
                "Additional optimizer guidance:\\n"
                "- Read each crop independently before inferring structure.\\n"
                "- Extract drug and dosage as separate fields.\\n"
                "- Ignore stamps, headers, and footer signatures unless explicitly requested.\\n"
                f"- Address recent failure modes:\\n{gradient_block}"
            )

        self.current_prompt = updated_prompt
        if self.prompt_parameter is not None:
            self.prompt_parameter.update_value(updated_prompt)

        return OptimizationStepResult(
            average_score=average_score,
            gradients=gradients,
            updated_prompt=updated_prompt,
            used_adalflow=self.adalflow_available,
        )

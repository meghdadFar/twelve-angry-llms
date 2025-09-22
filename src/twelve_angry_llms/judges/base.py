from abc import ABC, abstractmethod
from ..results.base import EvaluationResult

class Judge(ABC):
    """
    Abstract base class for all judges.
    """

    @abstractmethod
    def evaluate(self, *, generation: str, reference: str | None = None, context: str | None = None) -> EvaluationResult:
        """
        Evaluate a generation.

        :param generation: The generated text to evaluate.
        :param reference: The reference text to compare against.
        :param context: The context in which the generation was made.
        :return: An EvaluationResult object containing the evaluation results.
        """
        pass

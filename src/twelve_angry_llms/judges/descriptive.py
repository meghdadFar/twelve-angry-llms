from .base import Judge
from ..providers.base import LLMProvider
from ..results.descriptive import DescriptiveResult

class DescriptiveJudge(Judge):
    """
    A judge that uses an LLM to provide a descriptive evaluation.
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        :param llm_provider: The LLM provider to use for evaluation.
        """
        self.llm_provider = llm_provider

    def evaluate(self, *, generation: str, reference: str | None = None, context: str | None = None) -> DescriptiveResult:
        """
        Evaluate a generation using an LLM.

        :param generation: The generated text to evaluate.
        :param reference: The reference text to compare against.
        :param context: The context in which the generation was made.
        :return: A DescriptiveResult containing the LLM's descriptive evaluation.
        """
        prompt = f"Please evaluate the following generation. "
        if context:
            prompt += f"Context: {context}. "
        if reference:
            prompt += f"Reference: {reference}. "
        prompt += f"Generation: {generation}"

        evaluation = self.llm_provider.get_response(prompt)
        return DescriptiveResult(evaluation=evaluation)

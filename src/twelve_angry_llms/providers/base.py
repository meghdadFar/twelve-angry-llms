from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.

        :param prompt: The prompt to send to the LLM.
        :return: The LLM's response.
        """
        pass

import os
from openai import OpenAI
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    """
    An LLM provider that uses the OpenAI API.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-3.5-turbo"):
        """
        :param api_key: The OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
        :param model: The OpenAI model to use.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or pass it to the constructor.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def get_response(self, prompt: str) -> str:
        """
        Get a response from the OpenAI API.

        :param prompt: The prompt to send to the LLM.
        :return: The LLM's response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

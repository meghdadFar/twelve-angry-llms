from typing import Protocol, Optional

class LLMClient(Protocol):
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str: ...

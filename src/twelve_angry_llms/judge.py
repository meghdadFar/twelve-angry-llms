from typing import Any, List, Optional, Sequence, Set, Union
from .clients.base import LLMClient
from .tasks import GenerationTask, ClassificationTask, RankingTask, Task
from .types import JudgeOutput

class Judge:
    def __init__(self, name: str, client: LLMClient, temperature: float = 0.0):
        self.name = name
        self.client = client
        self.temperature = temperature

    def predict(self, task: Task) -> JudgeOutput:
        if isinstance(task, GenerationTask):
            prompt = self._prompt_generation(task)
            raw = self.client.generate(prompt, temperature=self.temperature)
            normalized = raw.strip()
            return JudgeOutput(judge=self.name, raw=raw, normalized=normalized)

        if isinstance(task, ClassificationTask):
            prompt = self._prompt_classification(task)
            raw = self.client.generate(prompt, temperature=self.temperature)
            normalized = self._parse_classification(raw, task.labels, task.multi_label)
            return JudgeOutput(judge=self.name, raw=raw, normalized=normalized)

        if isinstance(task, RankingTask):
            prompt = self._prompt_ranking(task)
            raw = self.client.generate(prompt, temperature=self.temperature)
            normalized = self._parse_ranking(raw, task.items)
            return JudgeOutput(judge=self.name, raw=raw, normalized=normalized)

        raise TypeError(f"Unsupported task type: {type(task)}")

    # Prompt builders (simple, provider-agnostic)
    def _prompt_generation(self, task: GenerationTask) -> str:
        guidance = f"\nGuidance: {task.guidance}" if task.guidance else ""
        return (
            "You are a careful assistant. Generate a helpful, concise response.\n"
            f"Input:\n{task.input_text}\n{guidance}\n"
            "Answer:\n"
        )

    def _prompt_classification(self, task: ClassificationTask) -> str:
        choices = ", ".join(task.labels)
        multi = "You may output multiple labels separated by commas." if task.multi_label else "Output exactly one label."
        return (
            "Classify the input into the provided labels.\n"
            f"Labels: {choices}\n{multi}\n"
            f"Input:\n{task.input_text}\n"
            "Answer with label(s) only:\n"
        )

    def _prompt_ranking(self, task: RankingTask) -> str:
        items = "\n".join(f"- {it}" for it in task.items)
        criteria = f" according to: {task.criteria}" if task.criteria else ""
        return (
            f"Rank the following items{criteria} from best to worst.\n"
            f"Items:\n{items}\n"
            "Return the ordered list, one per line, top to bottom.\n"
        )

    # Parsers (robust to extra text)
    def _parse_classification(
        self, text: str, labels: Sequence[str], multi: bool
    ) -> Union[str, Set[str]]:
        lower = text.lower()
        lbl_map = {l.lower(): l for l in labels}
        found = []
        for l in labels:
            if l.lower() in lower:
                found.append(lbl_map[l.lower()])
        if multi:
            return set(found)
        # single-label: pick the earliest matched label; fallback to best fuzzy contains
        if found:
            return found[0]
        # fallback: naive heuristic - choose label with maximum token overlap
        tokens = set(lower.split())
        best = None
        best_score = -1
        for l in labels:
            score = len(set(l.lower().split()) & tokens)
            if score > best_score:
                best, best_score = l, score
        return best or labels[0]

    def _parse_ranking(self, text: str, items: List[str]) -> List[str]:
        lower = text.lower()
        positions = []
        for it in items:
            idx = lower.find(it.lower())
            positions.append((idx if idx >= 0 else 10_000_000, it))
        positions.sort(key=lambda x: x[0])
        ordered = [it for _, it in positions]
        # If nothing matched, return original order
        return ordered if any(p < 10_000_000 for p, _ in positions) else list(items)

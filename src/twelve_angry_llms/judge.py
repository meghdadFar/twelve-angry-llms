from typing import Any, List, Optional, Sequence, Set, Union
from .clients.base import LLMClient
from .types import JudgeOutput
import re
import json

class Judge:
    def __init__(self, name: str, client: LLMClient, settings: dict[str, Any]):
        self.name = name
        self.client = client
        self.settings = settings

    # --- Public evaluation methods (consume already-produced outputs) ---

    def evaluate_generation(
        self,
        input_text: str,
        candidate: str,
        guidance: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> JudgeOutput:
        prompt = self._build_generation_eval_prompt(input_text, candidate, guidance, rubric)
        raw = self.client.generate(prompt, settings=self.settings)
        normalized = self._extract_score_and_reason(raw)
        return JudgeOutput(judge=self.name, raw=raw, normalized=normalized, reasoning=normalized.get("reasoning"))

    def evaluate_classification(
        self,
        input_text: str,
        candidate_labels: Union[str, Sequence[str], Set[str]],
        allowed_labels: Sequence[str],
        multi_label: bool = False,
        gold_labels: Optional[Sequence[str]] = None,
    ) -> JudgeOutput:
        if isinstance(candidate_labels, str):
            cand_norm = {candidate_labels} if multi_label else candidate_labels
        else:
            cand_norm = set(candidate_labels) if multi_label else list(candidate_labels)[0]
        prompt = self._build_classification_eval_prompt(
            input_text=input_text,
            candidate=cand_norm,
            allowed=allowed_labels,
            multi_label=multi_label,
            gold=gold_labels,
        )
        raw = self.client.generate(prompt, settings=self.settings)
        normalized = self._extract_score_and_reason(raw, extra={"candidate": cand_norm})
        return JudgeOutput(judge=self.name, raw=raw, normalized=normalized, reasoning=normalized.get("reasoning"))

    def evaluate_ranking(
        self,
        items: Sequence[str],
        candidate_ranking: Sequence[str],
        criteria: Optional[str] = None,
        gold_ranking: Optional[Sequence[str]] = None,
    ) -> JudgeOutput:
        prompt = self._build_ranking_eval_prompt(
            items=list(items),
            candidate=list(candidate_ranking),
            criteria=criteria,
            gold=list(gold_ranking) if gold_ranking else None,
        )
        raw = self.client.generate(prompt, settings=self.settings)
        normalized = self._extract_score_and_reason(raw, extra={"candidate_ranking": list(candidate_ranking)})
        return JudgeOutput(judge=self.name, raw=raw, normalized=normalized, reasoning=normalized.get("reasoning"))

    # --- Prompt builders (evaluation, not generation) ---

    def _build_generation_eval_prompt(
        self,
        input_text: str,
        candidate: str,
        guidance: Optional[str],
        rubric: Optional[str],
    ) -> str:
        guidance_part = f"\nGuidance:\n{guidance}" if guidance else ""
        rubric_part = f"\nRubric (use to justify score):\n{rubric}" if rubric else ""
        return (
            "You are an impartial evaluator of a model's response.\n"
            "Task Input:\n"
            f"{input_text}\n"
            f"{guidance_part}"
            "\nCandidate Response:\n"
            f"{candidate}\n"
            f"{rubric_part}\n"
            "Provide:\n"
            "1. A score 0-10 (higher is better)\n"
            "2. A concise reasoning\n"
            "Respond in JSON with keys: score, reasoning.\n"
        )

    def _build_classification_eval_prompt(
        self,
        input_text: str,
        candidate: Union[str, Set[str]],
        allowed: Sequence[str],
        multi_label: bool,
        gold: Optional[Sequence[str]],
    ) -> str:
        gold_part = f"\nGold Label(s): {list(gold)}" if gold else ""
        cand_display = list(candidate) if isinstance(candidate, set) else candidate
        return (
            "You are evaluating a classification decision.\n"
            f"Input Text:\n{input_text}\n"
            f"Allowed Labels: {list(allowed)}\n"
            f"Multi-label: {multi_label}\n"
            f"Candidate Prediction: {cand_display}"
            f"{gold_part}\n"
            "Assess correctness (if gold provided) and label suitability. Return JSON:\n"
            "{ \"score\": <0-10>, \"reasoning\": \"...\", \"valid\": <true|false> }\n"
        )

    def _build_ranking_eval_prompt(
        self,
        items: List[str],
        candidate: List[str],
        criteria: Optional[str],
        gold: Optional[List[str]],
    ) -> str:
        crit = f"Criteria: {criteria}\n" if criteria else ""
        gold_part = f"Gold Ranking: {gold}\n" if gold else ""
        return (
            "Evaluate a provided ranking of items.\n"
            f"{crit}"
            f"Items (unordered set): {items}\n"
            f"Candidate Ranking: {candidate}\n"
            f"{gold_part}"
            "Judge coherence, adherence to criteria (if any), and plausibility.\n"
            "Return JSON: { \"score\": 0-10, \"reasoning\": \"...\" }\n"
        )

    # --- Parsing / normalization helpers ---

    def _extract_score_and_reason(self, text: str, extra: Optional[dict] = None) -> dict:
        extra = extra or {}
        # Try JSON first
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            blob = json_match.group(0)
            try:
                data = json.loads(blob)
                score = self._coerce_score(data.get("score"))
                reasoning = data.get("reasoning") or data.get("reason") or ""
                data_norm = {
                    "score": score,
                    "reasoning": reasoning.strip(),
                    **{k: v for k, v in data.items() if k not in ("score", "reasoning", "reason")},
                    **extra,
                }
                return data_norm
            except Exception:
                pass
        # Fallback: regex for "score"
        score = None
        m = re.search(r"score[^0-9]{0,10}(\d{1,2}(\.\d+)?)", text.lower())
        if m:
            score = self._coerce_score(m.group(1))
        reasoning = text.strip()
        return {"score": score, "reasoning": reasoning, **extra}

    def _coerce_score(self, val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            # Clamp to 0-10 if clearly in that band
            if f < 0:
                return 0.0
            if f > 10 and f <= 100:  # maybe percent
                return round(f / 10.0, 2)
            if f > 10:
                return 10.0
            return round(f, 2)
        except Exception:
            return None

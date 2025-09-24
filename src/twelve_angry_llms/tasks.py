from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass(frozen=True)
class GenerationTask:
    input_text: str
    guidance: Optional[str] = None  # optional rubric or constraints

@dataclass(frozen=True)
class ClassificationTask:
    input_text: str
    labels: List[str]
    multi_label: bool = False

@dataclass(frozen=True)
class RankingTask:
    items: List[str]
    criteria: Optional[str] = None  # e.g., "most informative to least"

Task = Union[GenerationTask, ClassificationTask, RankingTask]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class JudgeOutput:
    judge: str
    raw: Any                  # raw model output (string)
    normalized: Any           # parsed/normalized output: str | set[str] | list[str]
    reasoning: Optional[str] = None

@dataclass
class JuryResult:
    task_type: str
    outputs: List[JudgeOutput]
    agreement: float
    details: Dict[str, Any]   # extra info per-metric (pairwise scores, etc.)

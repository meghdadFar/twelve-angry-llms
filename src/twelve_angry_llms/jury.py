from typing import List, Tuple, Union
from .judge import Judge
from .tasks import GenerationTask, ClassificationTask, RankingTask
from .types import JudgeOutput, JuryResult
from . import metrics

class Jury:
    def __init__(self, judges: List[Judge]):
        if not judges:
            raise ValueError("At least one judge is required")
        self.judges = judges

    def evaluate(self, task: Union[GenerationTask, ClassificationTask, RankingTask]) -> JuryResult:
        outputs: List[JudgeOutput] = [j.predict(task) for j in self.judges]

        if isinstance(task, GenerationTask):
            texts = [o.normalized for o in outputs]  # type: ignore
            agreement, details = metrics.agreement_generation(texts)
            return JuryResult(
                task_type="generation",
                outputs=outputs,
                agreement=agreement,
                details=details,
            )

        if isinstance(task, ClassificationTask):
            if task.multi_label:
                label_sets = [o.normalized for o in outputs]  # type: ignore
                agreement, details = metrics.agreement_classification_multi(label_sets)
            else:
                labels = [o.normalized for o in outputs]  # type: ignore
                agreement, details = metrics.agreement_classification_single(labels)
            return JuryResult(
                task_type="classification",
                outputs=outputs,
                agreement=agreement,
                details=details,
            )

        if isinstance(task, RankingTask):
            rankings = [o.normalized for o in outputs]  # type: ignore
            agreement, details = metrics.agreement_ranking(rankings)
            return JuryResult(
                task_type="ranking",
                outputs=outputs,
                agreement=agreement,
                details=details,
            )

        raise TypeError(f"Unsupported task type: {type(task)}")

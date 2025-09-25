from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math
import re

def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t]

def _pairwise_avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def agreement_generation(texts: List[str]) -> Tuple<float, Dict[str, Any]]:  # type: ignore[valid-type]
    """
    Average pairwise Jaccard similarity over token sets.
    Returns (agreement, details).
    """
    pair_scores: List[float] = []
    token_sets = [set(_tokenize(t)) for t in texts]
    for i, j in combinations(range(len(token_sets)), 2):
        a, b = token_sets[i], token_sets[j]
        if not a and not b:
            sim = 1.0
        else:
            sim = len(a & b) / max(1, len(a | b))
        pair_scores.append(sim)
    return _pairwise_avg(pair_scores), {"pairwise": pair_scores}

def agreement_classification_single(labels: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Average pairwise exact match for single-label predictions.
    labels: each element is the predicted label.
    """
    pair_scores: List[float] = []
    for i, j in combinations(range(len(labels)), 2):
        pair_scores.append(1.0 if labels[i] == labels[j] else 0.0)
    return _pairwise_avg(pair_scores), {"pairwise": pair_scores}

def agreement_classification_multi(label_sets: List[Iterable[str]]) -> Tuple[float, Dict[str, Any]]:
    """
    Average pairwise Jaccard similarity for multi-label predictions.
    """
    sets = [set(s) for s in label_sets]
    pair_scores: List[float] = []
    for i, j in combinations(range(len(sets)), 2):
        a, b = sets[i], sets[j]
        if not a and not b:
            sim = 1.0
        else:
            sim = len(a & b) / max(1, len(a | b))
        pair_scores.append(sim)
    return _pairwise_avg(pair_scores), {"pairwise": pair_scores}

def _spearman_rank(r1: Sequence[int], r2: Sequence[int]) -> float:
    """
    Spearman rho for two complete rankings without ties.
    r1, r2 are parallel rank vectors for same items.
    """
    n = len(r1)
    if n < 2:
        return 1.0
    d2 = sum((a - b) ** 2 for a, b in zip(r1, r2))
    return 1 - (6 * d2) / (n * (n * n - 1))

def agreement_ranking(rankings: List[List[str]]) -> Tuple[float, Dict[str, Any]]:
    """
    Average pairwise Spearman rank correlation across judge rankings.
    Each ranking is an ordered list of the same items.
    """
    if not rankings:
        return 0.0, {"pairwise": []}
    items = rankings[0]
    item_set = set(items)
    # Validate all rankings contain same items
    for r in rankings:
        if set(r) != item_set or len(r) != len(items):
            raise ValueError("All rankings must contain the same items (no ties)")

    def to_rank_vec(order: List[str]) -> Dict[str, int]:
        return {item: idx for idx, item in enumerate(order)}

    vecs = [to_rank_vec(r) for r in rankings]
    pair_scores: List[float] = []
    for i, j in combinations(range(len(vecs)), 2):
        v1, v2 = vecs[i], vecs[j]
        r1 = [v1[item] for item in items]
        r2 = [v2[item] for item in items]
        pair_scores.append(_spearman_rank(r1, r2))
    return _pairwise_avg(pair_scores), {"pairwise": pair_scores}

"""Transparent configurable statistical scores and candidate generation."""
from itertools import combinations
from typing import Any
import random

def number_scores(freq: dict[str, Any], gaps: dict[str, Any], weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = weights or {"frequency": 0.6, "gap": 0.4}
    scores = {}
    for n, value in freq["normalized_frequency"].items():
        gap = gaps["gap"][n]["normalized"]
        scores[n] = {"frequency": value, "gap": 1 - gap,
                     "score": round(weights["frequency"] * value + weights["gap"] * (1 - gap), 6)}
    return scores

def candidates(scores: dict[str, float], main_count: int, max_number: int, pair: dict[str, Any], top: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    ranked = sorted((int(n) for n in scores), key=lambda n: (-scores[str(n)], n))
    result, used = [], set()
    rng = random.Random(seed)
    for rank in range(1, top + 1):
        numbers = tuple(sorted(ranked[(rank - 1) % main_count:(rank - 1) % main_count + main_count]))
        if len(numbers) < main_count or numbers in used:
            numbers = tuple(sorted(rng.sample(range(1, max_number + 1), main_count)))
            while numbers in used:
                numbers = tuple(sorted(rng.sample(range(1, max_number + 1), main_count)))
        used.add(numbers)
        pair_score = sum(pair.get("pair_strength", {}).get(f"{a}-{b}", 0) for a, b in combinations(numbers, 2)) / max(1, len(list(combinations(numbers, 2))))
        balance = 1 - abs(sum(n % 2 for n in numbers) - main_count / 2) / main_count
        stat = sum(scores[str(n)] for n in numbers) / main_count
        result.append({"rank": rank, "numbers": list(numbers), "score": round(.7 * stat + .3 * pair_score, 6),
                       "statistical_score": round(stat, 6), "ml_score": 0.0, "pair_score": round(pair_score, 6),
                       "balance_score": round(balance, 6), "human_choice_penalty": 0.0})
    return result

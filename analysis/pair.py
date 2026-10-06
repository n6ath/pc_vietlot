"""Pair and optional triple descriptive statistics."""
from collections import Counter
from itertools import combinations
from typing import Any

def analyze(draws: list[dict[str, Any]], include_triples: bool = False) -> dict[str, Any]:
    pairs = Counter(pair for d in draws for pair in combinations(sorted(d["numbers"]), 2))
    total = max(1, len(draws))
    result: dict[str, Any] = {
        "pair_frequency": {f"{a}-{b}": c for (a, b), c in pairs.most_common(50)},
        "pair_support": {f"{a}-{b}": round(c / total, 6) for (a, b), c in pairs.most_common(50)},
        "pair_recency": {},
        "pair_gap": {},
        "pair_strength": {f"{a}-{b}": round(c / max(1, max(pairs.values(), default=1)), 6) for (a, b), c in pairs.most_common(50)},
    }
    if include_triples and len(draws) >= 30:
        triples = Counter(t for d in draws for t in combinations(sorted(d["numbers"]), 3))
        result["triple_frequency"] = {f"{a}-{b}-{c}": v for (a, b, c), v in triples.most_common(30)}
    return result

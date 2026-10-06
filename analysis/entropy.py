"""Entropy measures for descriptive draw randomness analysis."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _shannon(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def analyze(
    draws: list[dict[str, Any]],
    max_number: int,
    window: int = 20,
) -> dict[str, Any]:
    """Return entropy scores; these describe history and do not predict outcomes."""
    distributions = [n for draw in draws for n in draw["numbers"]]
    maximum = math.log2(max_number) if max_number > 1 else 1.0
    number_entropy = _shannon(distributions)
    rolling = [_shannon([n for draw in draws[i:i + window] for n in draw["numbers"]])
               for i in range(max(0, len(draws) - window + 1))]
    changes = [rolling[i] - rolling[i - 1] for i in range(1, len(rolling))]
    stability = 1.0 - min(1.0, (max(changes) - min(changes)) / maximum) if changes else 1.0
    per_number = {}
    for number in range(1, max_number + 1):
        appearances = [number in draw["numbers"] for draw in draws]
        per_number[str(number)] = round(_shannon(appearances), 6)
    return {
        "entropy_score": round(number_entropy / maximum if maximum else 0.0, 6),
        "entropy_trend": round((rolling[-1] - rolling[0]) / maximum, 6) if rolling else 0.0,
        "entropy_stability": round(max(0.0, stability), 6),
        "entropy_per_number": per_number,
        "rolling_entropy": [round(value, 6) for value in rolling],
        "draw_to_draw_entropy_change": [round(value, 6) for value in changes],
        "warning": "Entropy mô tả tính đa dạng của lịch sử, không chứng minh xác suất thắng.",
    }

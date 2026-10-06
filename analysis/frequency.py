"""Descriptive frequency statistics."""
from collections import Counter
from typing import Any

def analyze(draws: list[dict[str, Any]], max_number: int) -> dict[str, Any]:
    numbers = [n for d in draws for n in d["numbers"]]
    total = Counter(numbers)
    windows = {str(w): dict(Counter(n for d in draws[-w:] for n in d["numbers"])) for w in (10, 20, 50, 100, 300)}
    max_count = max(total.values(), default=1)
    ordered = sorted(range(1, max_number + 1), key=lambda n: (-total[n], n))
    return {
        "frequency": {str(n): total[n] for n in range(1, max_number + 1)},
        "recent_frequency": {k: {str(n): v for n, v in value.items()} for k, value in windows.items()},
        "hot_numbers": ordered[:min(10, max_number)],
        "cold_numbers": sorted(range(1, max_number + 1), key=lambda n: (total[n], n))[:min(10, max_number)],
        "normalized_frequency": {str(n): round(total[n] / max_count, 6) for n in range(1, max_number + 1)},
        "frequency_trend": {str(n): windows["10"].get(str(n), 0) - windows["50"].get(str(n), 0) / 5 for n in range(1, max_number + 1)},
    }

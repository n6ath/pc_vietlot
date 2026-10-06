"""Past-only rolling features."""
from statistics import mean
from typing import Any

def analyze(draws: list[dict[str, Any]], max_number: int, window: int = 20) -> dict[str, Any]:
    rows = []
    for end in range(max(1, len(draws) - window + 1), len(draws) + 1):
        sample = draws[max(0, end - window):end]
        rows.append({"end": end, "mean_sum": mean(sum(d["numbers"]) for d in sample),
                     "number_activity": len({n for d in sample for n in d["numbers"]})})
    return {"window": window, "rolling": rows}

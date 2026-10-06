"""Gap and recency statistics."""
from collections import defaultdict
from typing import Any

def analyze(draws: list[dict[str, Any]], max_number: int) -> dict[str, Any]:
    positions: dict[int, list[int]] = defaultdict(list)
    for i, draw in enumerate(draws):
        for n in draw["numbers"]:
            positions[n].append(i)
    gaps: dict[str, dict[str, float | int]] = {}
    for n in range(1, max_number + 1):
        p = positions[n]
        intervals = [b - a for a, b in zip(p, p[1:])]
        current = len(draws) - 1 - p[-1] if p else len(draws)
        gaps[str(n)] = {"current": current, "average": round(sum(intervals) / len(intervals), 4) if intervals else None,
                        "minimum": min(intervals) if intervals else None, "maximum": max(intervals) if intervals else None,
                        "appearances": len(p), "normalized": round(current / max(1, len(draws)), 6)}
    return {"gap": gaps, "large_gap_numbers": sorted(range(1, max_number + 1), key=lambda n: (-gaps[str(n)]["current"], n))[:10]}

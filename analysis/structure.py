"""Draw composition and structural statistics."""
from collections import Counter
from statistics import mean
from typing import Any

def analyze(draws: list[dict[str, Any]], max_number: int) -> dict[str, Any]:
    sums = [sum(d["numbers"]) for d in draws]
    ranges = [max(d["numbers"]) - min(d["numbers"]) for d in draws]
    odd_even = Counter(f"{sum(n % 2 for n in d['numbers'])} odd/{sum(n % 2 == 0 for n in d['numbers'])} even" for d in draws)
    high_low = Counter(f"{sum(n > max_number / 2 for n in d['numbers'])} high/{sum(n <= max_number / 2 for n in d['numbers'])} low" for d in draws)
    consecutive = Counter(sum(b == a + 1 for a, b in zip(d["numbers"], d["numbers"][1:])) for d in draws)
    repeated = [len(set(draws[i]["numbers"]) & set(draws[i - 1]["numbers"])) for i in range(1, len(draws))]
    bands = Counter(f"{((n - 1) * 5) // max_number + 1}" for d in draws for n in d["numbers"])
    return {"odd_even_distribution": dict(odd_even), "high_low_distribution": dict(high_low),
            "sum_statistics": {"min": min(sums) if sums else None, "max": max(sums) if sums else None, "mean": mean(sums) if sums else None},
            "range_statistics": {"min": min(ranges) if ranges else None, "max": max(ranges) if ranges else None, "mean": mean(ranges) if ranges else None},
            "consecutive_distribution": dict(consecutive), "repeated_between_draws": dict(Counter(repeated)),
            "band_distribution": dict(bands)}

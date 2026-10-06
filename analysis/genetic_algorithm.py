"""Genetic optimization of candidate tickets using supplied scores only."""
from __future__ import annotations

import random
from typing import Any, Callable


def optimize(
    max_number: int,
    main_count: int,
    number_scores: dict[str, float],
    pair_strength: dict[str, float] | None = None,
    entropy_scores: dict[str, float] | None = None,
    component_scores: dict[str, dict[str, float]] | None = None,
    weights: dict[str, float] | None = None,
    population_size: int = 40,
    generations: int = 20,
    mutation_rate: float = 0.08,
    crossover_rate: float = 0.8,
    elite_count: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Optimize combinations; it does not train a model or estimate odds."""
    rng = random.Random(seed)
    pairs = pair_strength or {}
    entropy = entropy_scores or {}
    components = component_scores or {}
    score_weights = weights or {
        "base": 0.55, "pair": 0.15, "entropy": 0.10,
        "frequency": 0.10, "gap": 0.05, "ml": 0.05,
    }

    def fitness(ticket: tuple[int, ...]) -> float:
        individual = sum(number_scores.get(str(n), 0.0) for n in ticket) / main_count
        pair_score = sum(pairs.get(f"{a}-{b}", 0.0) for a in ticket for b in ticket if a < b)
        entropy_score = sum(entropy.get(str(n), 0.0) for n in ticket) / main_count
        frequency_score = sum(components.get(str(n), {}).get("frequency", 0.0) for n in ticket) / main_count
        gap_score = sum(components.get(str(n), {}).get("gap", 0.0) for n in ticket) / main_count
        ml_score = sum(components.get(str(n), {}).get("ml", 0.0) for n in ticket) / main_count
        return round(
            score_weights["base"] * individual
            + score_weights["pair"] * pair_score / max(1, main_count)
            + score_weights["entropy"] * entropy_score
            + score_weights["frequency"] * frequency_score
            + score_weights["gap"] * gap_score
            + score_weights["ml"] * ml_score,
            6,
        )

    def random_ticket() -> tuple[int, ...]:
        return tuple(sorted(rng.sample(range(1, max_number + 1), main_count)))

    def crossover(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        pool = list(dict.fromkeys(left[:main_count // 2] + right[main_count // 2:]))
        pool += [n for n in range(1, max_number + 1) if n not in pool]
        return tuple(sorted(pool[:main_count]))

    population = [random_ticket() for _ in range(max(2, population_size))]
    history = []
    for _ in range(max(1, generations)):
        ranked = sorted(population, key=fitness, reverse=True)
        history.append({"best": fitness(ranked[0]), "mean": round(sum(fitness(t) for t in ranked) / len(ranked), 6)})
        next_population = ranked[:max(1, elite_count)]
        while len(next_population) < len(population):
            parents = rng.sample(ranked[:max(2, len(ranked) // 2)], 2)
            child = crossover(*parents) if rng.random() < crossover_rate else parents[0]
            if rng.random() < mutation_rate:
                values = set(child)
                values.remove(rng.choice(tuple(values)))
                values.add(rng.choice([n for n in range(1, max_number + 1) if n not in values]))
                child = tuple(sorted(values))
            next_population.append(child)
        population = next_population
    best = sorted(set(population), key=fitness, reverse=True)[:10]
    return {"best_combinations": [{"numbers": list(ticket), "fitness": fitness(ticket)} for ticket in best],
            "fitness_history": history,
            "convergence_metrics": {"generations": generations, "population_size": population_size, "seed": seed},
            "warning": "Genetic Algorithm chỉ tối ưu bộ vé theo điểm đầu vào, không học xác suất xổ số."}

"""Generic rectangular-grid analysis for lottery number layouts."""
from __future__ import annotations

from collections import Counter
from typing import Any


def analyze(
    draws: list[dict[str, Any]],
    max_number: int,
    columns: int | None = None,
) -> dict[str, Any]:
    columns = columns or max(1, round(max_number ** 0.5))
    rows = (max_number + columns - 1) // columns
    row_counts = Counter()
    column_counts = Counter()
    diagonal_hits = 0
    cluster_scores = []
    for draw in draws:
        numbers = draw["numbers"]
        cells = [((number - 1) // columns, (number - 1) % columns) for number in numbers]
        row_counts.update(row for row, _ in cells)
        column_counts.update(column for _, column in cells)
        diagonal_hits += sum(row == column or row + column == columns - 1 for row, column in cells)
        adjacent = sum(abs(a - b) <= 1 for a in numbers for b in numbers if a < b)
        cluster_scores.append(adjacent / max(1, len(numbers) * (len(numbers) - 1) / 2))
    balance = 1.0 - (
        (max(row_counts.values(), default=0) - min(row_counts.values(), default=0)) +
        (max(column_counts.values(), default=0) - min(column_counts.values(), default=0))
    ) / max(1, len(draws) * 2)
    return {
        "layout": {"rows": rows, "columns": columns, "max_number": max_number},
        "row_distribution": {str(k): v for k, v in sorted(row_counts.items())},
        "column_distribution": {str(k): v for k, v in sorted(column_counts.items())},
        "diagonal_hits": diagonal_hits,
        "cluster_density_score": round(sum(cluster_scores) / len(cluster_scores), 6) if cluster_scores else 0.0,
        "grid_balance_score": round(max(0.0, min(1.0, balance)), 6),
        "spatial_score": round((max(0.0, min(1.0, balance)) + (diagonal_hits / max(1, len(draws) * 6))), 6),
        "warning": "Bố cục lưới là mô tả hình học; không phải bằng chứng dự đoán.",
    }

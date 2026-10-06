"""Ticket covering and abbreviated-wheel construction utilities."""
from __future__ import annotations

from itertools import combinations
from typing import Any


def generate(
    selected_numbers: list[int],
    main_count: int,
    mode: str = "abbreviated",
    max_tickets: int = 1000,
) -> dict[str, Any]:
    """Generate tickets from selected numbers; this is not a prediction method."""
    numbers = sorted(set(int(number) for number in selected_numbers))
    if len(numbers) < main_count:
        return {"generated_tickets": [], "coverage_rate": 0.0, "ticket_count": 0,
                "expected_match_coverage": None, "mode": mode}
    all_tickets = list(combinations(numbers, main_count))
    tickets = all_tickets if mode == "full" else all_tickets[:max_tickets]
    total = len(all_tickets)
    coverage = len(tickets) / total if total else 0.0
    return {
        "generated_tickets": [list(ticket) for ticket in tickets],
        "coverage_rate": round(coverage, 6),
        "ticket_count": len(tickets),
        "expected_match_coverage": round(coverage, 6),
        "mode": mode,
        "warning": "Bao phủ vé là tối ưu hóa tổ hợp, không làm tăng xác suất của từng kỳ quay.",
    }

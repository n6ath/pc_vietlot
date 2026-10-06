"""Public wheel-system facade for full, abbreviated, and covering designs."""
from __future__ import annotations

from typing import Any

from .covering_design import generate


def analyze(
    selected_numbers: list[int],
    main_count: int,
    mode: str = "abbreviated",
    max_tickets: int = 1000,
) -> dict[str, Any]:
    return generate(selected_numbers, main_count, mode=mode, max_tickets=max_tickets)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EasyTask:
    task_id: str = "rename_variables"
    description: str = (
        "Refactor the function by renaming generic variables (`x`, `tmp`, `i`) "
        "into descriptive names while preserving behavior."
    )
    input_code: str = """\
def compute(x, y, tmp):
    tmp = x + y
    x = tmp * 2
    result = x
    return result
"""
    expected_output: str = """\
def compute(left, right, sum_value):
    sum_value = left + right
    doubled = sum_value * 2
    result = doubled
    return result
"""


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MediumTask:
    task_id: str = "remove_dead_code"
    description: str = (
        "Remove dead code patterns (unreachable statements, `if False` blocks, and "
        "obviously unused assignments) while keeping functional behavior intact."
    )
    input_code: str = """\
def process(data):
    result = []
    for item in data:
        result.append(item * 2)
    if False:
        print("never runs")
    unused_var = 42
    return result
    print("unreachable")
"""
    expected_output: str = """\
def process(data):
    return [item * 2 for item in data]
"""


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardTask:
    task_id: str = "full_refactor"
    description: str = (
        "Perform a full refactor: rename generic variables, remove dead branches, "
        "simplify loops into comprehensions, optimize boolean conditions, and inline "
        "trivial helpers where appropriate."
    )
    input_code: str = """\
def add(p, q):
    return p + q

def compute(x, data, tmp):
    result = []
    for item in data:
        result.append(item * 2)
    if False:
        y = 999
    if True:
        val = add(x, tmp)
    unused = 0
    flag = not not True
    return val
    print("dead")
"""
    expected_output: str = """\
def compute(value, data, offset):
    _ = [item * 2 for item in data]
    return value + offset
"""


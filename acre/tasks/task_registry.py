"""
Three OpenEnv tasks with AST-based graders scoring 0.0-1.0.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class Task:
    id: str
    name: str
    description: str
    difficulty: str
    initial_code: str
    _grade_fn: Callable[[str], float]

    def grade(self, code: str) -> float:
        """Return a score in [0.0, 1.0]."""
        try:
            return float(min(1.0, max(0.0, self._grade_fn(code))))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Task 1 — Easy: Rename generic variables
# ---------------------------------------------------------------------------
_EASY_CODE = """\
def compute(x, y, tmp):
    tmp = x + y
    x = tmp * 2
    result = x
    return result
"""


def _grade_easy(code: str) -> float:
    """Score = fraction of generic names (x, tmp) removed from all scopes."""
    generic = {"x", "tmp"}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    remaining: set[str] = set()

    class _Collector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id in generic:
                remaining.add(node.id)
            self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> None:
            if node.arg in generic:
                remaining.add(node.arg)
            self.generic_visit(node)

    _Collector().visit(tree)
    renamed = len(generic - remaining)
    return renamed / len(generic)


# ---------------------------------------------------------------------------
# Task 2 — Medium: Remove dead code
# ---------------------------------------------------------------------------
_MEDIUM_CODE = """\
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


def _grade_medium(code: str) -> float:
    """Score = fraction of dead-code patterns eliminated (3 checks, ~0.33 each)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    source = ast.unparse(tree)
    score = 0.0

    # Check 1: if-False block removed
    if "if False" not in source:
        score += 1 / 3

    # Check 2: unused_var assignment removed
    if "unused_var" not in source:
        score += 1 / 3

    # Check 3: list comprehension used (loop simplified)
    has_listcomp = any(isinstance(n, ast.ListComp) for n in ast.walk(tree))
    if has_listcomp:
        score += 1 / 3

    return score


# ---------------------------------------------------------------------------
# Task 3 — Hard: Full refactor
# ---------------------------------------------------------------------------
_HARD_CODE = """\
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


def _grade_hard(code: str) -> float:
    """Score = fraction of 5 quality checks passed."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    source = ast.unparse(tree)
    checks = 0

    # 1. No generic variable names x/tmp in function signature or body
    has_generic = False

    class _GenCheck(ast.NodeVisitor):
        def visit_arg(self, node: ast.arg) -> None:
            nonlocal has_generic
            if node.arg in {"x", "tmp"}:
                has_generic = True

    _GenCheck().visit(tree)
    if not has_generic:
        checks += 1

    # 2. No if False block
    if "if False" not in source:
        checks += 1

    # 3. if True removed (body inlined)
    if "if True" not in source:
        checks += 1

    # 4. List comprehension used
    if any(isinstance(n, ast.ListComp) for n in ast.walk(tree)):
        checks += 1

    # 5. add() call inlined (no call to 'add')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    fn_names = {c.func.id for c in calls if isinstance(c.func, ast.Name)}
    if "add" not in fn_names:
        checks += 1

    return checks / 5


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._register_all()

    def _register_all(self) -> None:
        self._tasks["rename_variables"] = Task(
            id="rename_variables",
            name="Rename Variables (Easy)",
            description="Rename generic variable names (x, tmp) to descriptive ones",
            difficulty="easy",
            initial_code=_EASY_CODE,
            _grade_fn=_grade_easy,
        )
        self._tasks["remove_dead_code"] = Task(
            id="remove_dead_code",
            name="Remove Dead Code (Medium)",
            description="Remove unreachable code, if False blocks, and unused variables",
            difficulty="medium",
            initial_code=_MEDIUM_CODE,
            _grade_fn=_grade_medium,
        )
        self._tasks["full_refactor"] = Task(
            id="full_refactor",
            name="Full Refactor (Hard)",
            description="Apply all transformations: rename, dead code, loops, conditions, inlining",
            difficulty="hard",
            initial_code=_HARD_CODE,
            _grade_fn=_grade_hard,
        )

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[dict]:
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "difficulty": t.difficulty,
                "initial_code": t.initial_code,
            }
            for t in self._tasks.values()
        ]

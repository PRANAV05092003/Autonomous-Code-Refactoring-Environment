"""
Three OpenEnv tasks with AST-based graders scoring 0.0-1.0.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from acre.tasks.easy_task import EasyTask
from acre.tasks.hard_task import HardTask
from acre.tasks.medium_task import MediumTask
from acre.tasks.grader import grade_task


@dataclass
class Task:
    id: str
    name: str
    description: str
    difficulty: str
    samples: List[str]
    expected_outputs: List[str]
    _grade_fn: Callable[[str], float]

    @property
    def initial_code(self) -> str:
        return str(self.samples[0]) if self.samples else ""

    def expected_output_for_index(self, idx: int) -> str:
        if 0 <= idx < len(self.expected_outputs):
            return str(self.expected_outputs[idx])
        return str(self.expected_outputs[0]) if self.expected_outputs else ""

    def grade(self, code: str) -> float:
        """Return a score in [0.0, 1.0]."""
        try:
            return float(min(1.0, max(0.0, self._grade_fn(code))))
        except Exception:
            return 0.0

    def grade_against_expected(self, code: str) -> float:
        """
        Deterministic grader comparing against this task's expected outputs.

        Since the HTTP `grade` endpoint doesn't know which sample was active, we
        score against the best-matching expected output (still deterministic).
        """
        if not self.expected_outputs:
            return 0.0
        return float(max(grade_task(code, exp) for exp in self.expected_outputs))


def _safe_unparse(tree: ast.AST) -> str:
    try:
        return ast.unparse(tree)
    except Exception:
        return ""


def _has_unreachable_after_terminator(stmts: Sequence[ast.stmt]) -> bool:
    unreachable = False
    for s in stmts:
        if unreachable:
            # ignore empty docstrings as "unreachable" noise
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                continue
            return True
        if isinstance(s, (ast.Return, ast.Raise)):
            unreachable = True
    return False


def _tree_has_unreachable(tree: ast.AST) -> bool:
    class _Scan(ast.NodeVisitor):
        def __init__(self) -> None:
            self.bad = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            if _has_unreachable_after_terminator(node.body):
                self.bad = True
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            if _has_unreachable_after_terminator(node.body):
                self.bad = True
            self.generic_visit(node)

    s = _Scan()
    s.visit(tree)
    return bool(s.bad)


# ---------------------------------------------------------------------------
# Task 1 — Easy: Rename generic variables
# ---------------------------------------------------------------------------
_EASY_SAMPLES: List[str] = [
    """\
def compute(x, y, tmp):
    tmp = x + y
    x = tmp * 2
    result = x
    return result
""",
    """\
def normalize(tmp, x):
    for i in range(3):
        tmp = tmp + i
    return tmp * x
""",
    """\
def score(items):
    tmp = 0
    for i in items:
        tmp += i
    x = tmp
    return x
""",
    """\
def transform(x):
    tmp = x
    if tmp > 10:
        tmp = tmp - 1
    return tmp
""",
    """\
def merge(a, b):
    x = a
    tmp = b
    return x + tmp
""",
]

_EASY_EXPECTED: List[str] = [
    EasyTask.expected_output,
    """\
def normalize(temp_value, value):
    for index in range(3):
        temp_value = temp_value + index
    return temp_value * value
""",
    """\
def score(items):
    total = 0
    for item in items:
        total += item
    value = total
    return value
""",
    """\
def transform(value):
    temp_value = value
    if temp_value > 10:
        temp_value = temp_value - 1
    return temp_value
""",
    """\
def merge(a, b):
    left = a
    right = b
    return left + right
""",
]


def _grade_easy(code: str) -> float:
    """Score = fraction of generic names removed from all scopes."""
    generic = {"x", "tmp", "i"}
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
_MEDIUM_SAMPLES: List[str] = [
    """\
def process(data):
    result = []
    for item in data:
        result.append(item * 2)
    if False:
        print("never runs")
    unused_var = 42
    return result
    print("unreachable")
""",
    """\
def build(values):
    out = []
    for v in values:
        out.append(v + 1)
    while False:
        out.append(999)
    dead = 0
    return out
    dead += 1
""",
    """\
def route(flag):
    if False:
        return 1
    if True:
        x = 2
    y = x
    return y
""",
    """\
def clean(xs):
    res = []
    for x in xs:
        res.append(x * 2)
    unused = "remove me"
    if False:
        unused2 = 123
    return res
""",
    """\
def calc(n):
    total = 0
    for i in range(n):
        total += i
    return total
    print("dead")
""",
]

_MEDIUM_EXPECTED: List[str] = [
    MediumTask.expected_output,
    """\
def build(values):
    return [v + 1 for v in values]
""",
    """\
def route(flag):
    x = 2
    y = x
    return y
""",
    """\
def clean(xs):
    return [x * 2 for x in xs]
""",
    """\
def calc(n):
    total = 0
    for index in range(n):
        total += index
    return total
""",
]


def _grade_medium(code: str) -> float:
    """Score = fraction of dead-code patterns eliminated (4 checks, 0.25 each)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    source = _safe_unparse(tree)
    score = 0.0

    # Check 1: if/while-False removed
    if ("if False" not in source) and ("while False" not in source):
        score += 0.25

    # Check 2: no unreachable statements after return/raise
    if not _tree_has_unreachable(tree):
        score += 0.25

    # Check 3: list comprehension used (loop simplified)
    has_listcomp = any(isinstance(n, ast.ListComp) for n in ast.walk(tree))
    if has_listcomp:
        score += 0.25

    # Check 4: obvious dead/unused sentinel names removed
    if all(name not in source for name in ["unused_var", "unused", "dead", "unused2"]):
        score += 0.25

    return score


# ---------------------------------------------------------------------------
# Task 3 — Hard: Full refactor
# ---------------------------------------------------------------------------
_HARD_SAMPLES: List[str] = [
    """\
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
""",
    """\
def helper(a, b):
    return a + b

def pipeline(tmp, xs, x):
    out = []
    for i in xs:
        out.append(i * 2)
    if True:
        y = helper(tmp, x)
    if False:
        y = 0
    return y
    y = 123
""",
    """\
def add(p, q):
    return p + q

def compute(x, data, tmp):
    result = []
    for item in data:
        result.append(item * 2)
    if False:
        print("never")
    val = add(x, tmp)
    return val
""",
    """\
def add(p, q):
    return p + q

def compute(x, data, tmp):
    res = []
    for item in data:
        res.append(item * 2)
    flag = not not True
    if True:
        return add(x, tmp)
""",
    """\
def plus(p, q):
    return p + q

def compute(tmp, data, x):
    out = []
    for item in data:
        out.append(item * 2)
    if False:
        tmp = 999
    if True:
        val = plus(x, tmp)
    return val
""",
]

_HARD_EXPECTED: List[str] = [
    HardTask.expected_output,
    """\
def pipeline(offset, xs, value):
    _ = [item * 2 for item in xs]
    return offset + value
""",
    """\
def compute(value, data, offset):
    _ = [item * 2 for item in data]
    return value + offset
""",
    """\
def compute(value, data, offset):
    _ = [item * 2 for item in data]
    return value + offset
""",
    """\
def compute(offset, data, value):
    _ = [item * 2 for item in data]
    return value + offset
""",
]


def _grade_hard(code: str) -> float:
    """Score = fraction of 7 quality checks passed."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    source = _safe_unparse(tree)
    checks = 0

    # 1. No generic variable names x/tmp/i in function signature
    has_generic = False

    class _GenCheck(ast.NodeVisitor):
        def visit_arg(self, node: ast.arg) -> None:
            nonlocal has_generic
            if node.arg in {"x", "tmp", "i"}:
                has_generic = True

    _GenCheck().visit(tree)
    if not has_generic:
        checks += 1

    # 2. No if/while False block
    if ("if False" not in source) and ("while False" not in source):
        checks += 1

    # 3. if True removed (body inlined)
    if "if True" not in source:
        checks += 1

    # 4. List comprehension used
    if any(isinstance(n, ast.ListComp) for n in ast.walk(tree)):
        checks += 1

    # 5. helper calls inlined (no call sites remain)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    fn_names = {c.func.id for c in calls if isinstance(c.func, ast.Name)}
    if not ({"add", "plus", "helper"} & fn_names):
        checks += 1

    # 6. no unreachable after return/raise
    if not _tree_has_unreachable(tree):
        checks += 1

    # 7. remove double-not
    if "not not" not in source:
        checks += 1

    return checks / 7


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
            description=EasyTask.description,
            difficulty="easy",
            samples=_EASY_SAMPLES,
            expected_outputs=_EASY_EXPECTED,
            _grade_fn=_grade_easy,
        )
        self._tasks["remove_dead_code"] = Task(
            id="remove_dead_code",
            name="Remove Dead Code (Medium)",
            description=MediumTask.description,
            difficulty="medium",
            samples=_MEDIUM_SAMPLES,
            expected_outputs=_MEDIUM_EXPECTED,
            _grade_fn=_grade_medium,
        )
        self._tasks["full_refactor"] = Task(
            id="full_refactor",
            name="Full Refactor (Hard)",
            description=HardTask.description,
            difficulty="hard",
            samples=_HARD_SAMPLES,
            expected_outputs=_HARD_EXPECTED,
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

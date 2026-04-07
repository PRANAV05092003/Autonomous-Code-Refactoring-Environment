from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class TransformationResult:
    """Output of applying a transformation (placeholder)."""

    code: str
    changed: bool
    metadata: Dict[str, Any]


class Transformation(Protocol):
    """Protocol for a code transformation."""

    name: str

    def apply(self, code: str) -> TransformationResult: ...


def noop_transformation(code: str) -> TransformationResult:
    """Baseline transformation that leaves code unchanged."""
    return TransformationResult(code=code, changed=False, metadata={"kind": "noop"})


def _finalize_result(*, original: str, out: str, meta: Dict[str, Any]) -> TransformationResult:
    """
    Standardize metadata across transformations.

    - Adds `lines_changed` and `impact` for explainability/metrics.
    - Ensures formatting-only changes don't count as `changed`.
    """

    def _count_lines_changed(a: str, b: str) -> int:
        a_lines = a.splitlines()
        b_lines = b.splitlines()
        changed = 0
        for x, y in zip_longest(a_lines, b_lines, fillvalue=None):
            if x != y:
                changed += 1
        return int(changed)

    lines_changed = _count_lines_changed(original, out)

    # Fallback identity check: AST round-trips can reformat without changing meaning.
    # If the textual content is the same after stripping, treat it as unchanged.
    if out.strip() == original.strip():
        meta["success"] = False
        meta["lines_changed"] = 0
        meta["impact"] = "low"
        return TransformationResult(code=original, changed=False, metadata=meta)

    meta["lines_changed"] = lines_changed
    meta["impact"] = "high" if lines_changed >= 3 else "low"
    meta["success"] = True
    return TransformationResult(code=out, changed=True, metadata=meta)


def _unchanged(*, code: str, meta: Dict[str, Any]) -> TransformationResult:
    meta.setdefault("success", False)
    meta.setdefault("lines_changed", 0)
    meta.setdefault("impact", "low")
    return TransformationResult(code=code, changed=False, metadata=meta)


def rename_variable(code: str) -> TransformationResult:
    """
    Rename simple, generic variable names to more descriptive ones.

    Hackathon-scope heuristic:
    - Rename generic names in priority order: x, tmp, i.
    - Uses descriptive base names and avoids collisions.
    - Applies to Name nodes and function args.
    """
    meta: Dict[str, Any] = {"type": "rename_variable", "success": False}
    try:
        tree = ast.parse(code)

        class _NameCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.names: set[str] = set()

            def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
                self.names.add(node.id)

            def visit_arg(self, node: ast.arg) -> None:  # noqa: N802
                self.names.add(node.arg)

        collector = _NameCollector()
        collector.visit(tree)

        rename_plan = [
            ("x", "value"),
            ("tmp", "temp_value"),
            ("i", "index"),
        ]

        old = ""
        base_new = "value"
        for candidate_old, candidate_base in rename_plan:
            if candidate_old in collector.names:
                old = candidate_old
                base_new = candidate_base
                break

        if not old:
            return _unchanged(code=code, meta=meta)

        new = base_new
        i = 1
        while new in collector.names:
            new = f"{base_new}{i}"
            i += 1

        class _Renamer(ast.NodeTransformer):
            def __init__(self, old_name: str, new_name: str) -> None:
                self.old_name = old_name
                self.new_name = new_name
                self.changed = False

            def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
                if node.id == self.old_name:
                    self.changed = True
                    return ast.copy_location(ast.Name(id=self.new_name, ctx=node.ctx), node)
                return node

            def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
                if node.arg == self.old_name:
                    self.changed = True
                    new_node = copy.copy(node)
                    new_node.arg = self.new_name
                    return new_node
                return node

        renamer = _Renamer(old, new)
        tree = renamer.visit(tree)
        ast.fix_missing_locations(tree)

        if not renamer.changed:
            return _unchanged(code=code, meta=meta)

        out = ast.unparse(tree)
        meta["old"] = old
        meta["new"] = new
        # Renames tend to be small diffs; label as low impact unless the diff is large.
        return _finalize_result(original=code, out=out, meta=meta)
    except Exception:
        return _unchanged(code=code, meta=meta)


def remove_dead_code(code: str) -> TransformationResult:
    """
    Remove simple dead code patterns.

    Hackathon-scope heuristics:
    - Drop statements after `return` / `raise` in the same block.
    - Remove `if False: ...` blocks (keep `else` if present).
    - Remove assignments to unused names in a block (very simple check).
    """
    meta: Dict[str, Any] = {"type": "remove_dead_code", "success": False}

    try:
        tree = ast.parse(code)

        def _is_const_bool(expr: ast.AST, value: bool) -> bool:
            return isinstance(expr, ast.Constant) and isinstance(expr.value, bool) and expr.value is value

        class _LoadNameCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.loaded: set[str] = set()

            def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
                if isinstance(node.ctx, ast.Load):
                    self.loaded.add(node.id)

        class _DeadCode(ast.NodeTransformer):
            def __init__(self) -> None:
                self.changed = False

            def _prune_unreachable(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
                out: list[ast.stmt] = []
                unreachable = False
                for s in stmts:
                    if unreachable:
                        self.changed = True
                        continue
                    out.append(s)
                    if isinstance(s, (ast.Return, ast.Raise)):
                        unreachable = True
                return out

            def _remove_unused_assigns(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
                collector = _LoadNameCollector()
                for s in stmts:
                    collector.visit(s)
                used = collector.loaded

                out: list[ast.stmt] = []
                for s in stmts:
                    if isinstance(s, ast.Assign) and all(isinstance(t, ast.Name) for t in s.targets):
                        targets = [t.id for t in s.targets if isinstance(t, ast.Name)]
                        # Remove only if *all* assigned names are unused.
                        if targets and all(t not in used for t in targets):
                            self.changed = True
                            continue
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                        if s.target.id not in used:
                            self.changed = True
                            continue
                    out.append(s)
                return out

            def _clean_block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
                # First apply transformations inside statements.
                visited = [self.visit(s) for s in stmts]
                flat: list[ast.stmt] = []
                for s in visited:
                    if s is None:
                        self.changed = True
                        continue
                    if isinstance(s, list):
                        flat.extend([x for x in s if isinstance(x, ast.stmt)])
                        self.changed = True
                    else:
                        flat.append(s)

                flat = self._prune_unreachable(flat)
                flat = self._remove_unused_assigns(flat)
                return flat

            def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
                node.body = self._clean_block(node.body)
                return node

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
                node.body = self._clean_block(node.body)
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
                node.body = self._clean_block(node.body)
                return node

            def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:  # noqa: N802
                node = self.generic_visit(node)
                if _is_const_bool(node.test, False):
                    self.changed = True
                    return node.orelse or []
                return node

            def visit_While(self, node: ast.While) -> ast.AST | None:  # noqa: N802
                node = self.generic_visit(node)
                if _is_const_bool(node.test, False):
                    self.changed = True
                    return None
                return node

        dc = _DeadCode()
        tree = dc.visit(tree)
        ast.fix_missing_locations(tree)
        if not dc.changed:
            return _unchanged(code=code, meta=meta)

        out = ast.unparse(tree)
        return _finalize_result(original=code, out=out, meta=meta)
    except Exception:
        return _unchanged(code=code, meta=meta)


def simplify_loops(code: str) -> TransformationResult:
    """
    Simplify very basic loop patterns into more pythonic forms.

    Supported pattern (only when adjacent in the same block):
    - xs = []
      for t in it:
          xs.append(expr)
      => xs = [expr for t in it]
    """
    meta: Dict[str, Any] = {"type": "simplify_loops", "success": False}
    try:
        tree = ast.parse(code)

        class _LoopSimplifier(ast.NodeTransformer):
            def __init__(self) -> None:
                self.changed = False

            def _simplify_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
                out: list[ast.stmt] = []
                i = 0
                while i < len(body):
                    cur = body[i]
                    nxt = body[i + 1] if i + 1 < len(body) else None

                    if (
                        isinstance(cur, ast.Assign)
                        and len(cur.targets) == 1
                        and isinstance(cur.targets[0], ast.Name)
                        and isinstance(cur.value, ast.List)
                        and cur.value.elts == []
                        and isinstance(nxt, ast.For)
                        and len(nxt.body) == 1
                        and isinstance(nxt.body[0], ast.Expr)
                        and isinstance(nxt.body[0].value, ast.Call)
                    ):
                        list_name = cur.targets[0].id
                        call = nxt.body[0].value
                        if (
                            isinstance(call.func, ast.Attribute)
                            and isinstance(call.func.value, ast.Name)
                            and call.func.value.id == list_name
                            and call.func.attr == "append"
                            and len(call.args) == 1
                            and not call.keywords
                        ):
                            # Build list comprehension: [call.args[0] for <target> in <iter>]
                            comp = ast.ListComp(
                                elt=call.args[0],
                                generators=[
                                    ast.comprehension(
                                        target=nxt.target,
                                        iter=nxt.iter,
                                        ifs=[],
                                        is_async=0,
                                    )
                                ],
                            )
                            new_assign = ast.Assign(targets=[ast.Name(id=list_name, ctx=ast.Store())], value=comp)
                            out.append(ast.copy_location(new_assign, cur))
                            self.changed = True
                            i += 2
                            continue

                    out.append(cur)
                    i += 1

                return out

            def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
                node = self.generic_visit(node)
                node.body = self._simplify_body(node.body)
                return node

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
                node = self.generic_visit(node)
                node.body = self._simplify_body(node.body)
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
                node = self.generic_visit(node)
                node.body = self._simplify_body(node.body)
                return node

        simp = _LoopSimplifier()
        tree = simp.visit(tree)
        ast.fix_missing_locations(tree)
        if not simp.changed:
            return _unchanged(code=code, meta=meta)

        out = ast.unparse(tree)
        return _finalize_result(original=code, out=out, meta=meta)
    except Exception:
        return _unchanged(code=code, meta=meta)


def simplify_loop(code: str) -> TransformationResult:
    # Backwards-compatible alias for the environment's action mapping.
    return simplify_loops(code)


def optimize_condition(code: str) -> TransformationResult:
    """
    Simplify redundant boolean conditions.

    Hackathon-scope heuristics:
    - Replace `if True:` with its body; `if False:` with `else` (if present).
    - Simplify `not not X` -> `X`.
    - Simplify comparisons to True/False: `X == True` -> `X`, `X == False` -> `not X`.
    """
    meta: Dict[str, Any] = {"type": "optimize_condition", "success": False}
    try:
        tree = ast.parse(code)

        def _is_bool_const(node: ast.AST, value: bool) -> bool:
            return isinstance(node, ast.Constant) and isinstance(node.value, bool) and node.value is value

        class _CondOpt(ast.NodeTransformer):
            def __init__(self) -> None:
                self.changed = False

            def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:  # noqa: N802
                node = self.generic_visit(node)
                if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.Not):
                    self.changed = True
                    return node.operand.operand
                return node

            def visit_Compare(self, node: ast.Compare) -> ast.AST:  # noqa: N802
                node = self.generic_visit(node)
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    op = node.ops[0]
                    rhs = node.comparators[0]
                    if isinstance(op, (ast.Eq, ast.Is)) and _is_bool_const(rhs, True):
                        self.changed = True
                        return node.left
                    if isinstance(op, (ast.Eq, ast.Is)) and _is_bool_const(rhs, False):
                        self.changed = True
                        return ast.UnaryOp(op=ast.Not(), operand=node.left)
                return node

            def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:  # noqa: N802
                node = self.generic_visit(node)
                if _is_bool_const(node.test, True):
                    self.changed = True
                    return node.body
                if _is_bool_const(node.test, False):
                    self.changed = True
                    return node.orelse or []
                return node

        opt = _CondOpt()
        tree = opt.visit(tree)
        ast.fix_missing_locations(tree)
        if not opt.changed:
            return _unchanged(code=code, meta=meta)

        out = ast.unparse(tree)
        return _finalize_result(original=code, out=out, meta=meta)
    except Exception:
        return _unchanged(code=code, meta=meta)


def inline_function(code: str) -> TransformationResult:
    """
    Inline very simple functions into their call sites.

    Supported pattern:
    - def f(a, b): return <expr using only a,b>
    - Replace calls: f(x, y) -> <expr with a->x, b->y>
    Only handles module-level functions and positional args.
    """
    meta: Dict[str, Any] = {"type": "inline_function", "success": False}
    try:
        tree = ast.parse(code)

        simple_fns: Dict[str, tuple[list[str], ast.AST]] = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.decorator_list:
                continue
            args = node.args
            if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.posonlyargs:
                continue
            if len(node.body) != 1 or not isinstance(node.body[0], ast.Return) or node.body[0].value is None:
                continue
            arg_names = [a.arg for a in args.args]
            # Ensure the return expression only references the function's args.
            referenced: set[str] = set()

            class _Ref(ast.NodeVisitor):
                def visit_Name(self, n: ast.Name) -> None:  # noqa: N802
                    if isinstance(n.ctx, ast.Load):
                        referenced.add(n.id)

            _Ref().visit(node.body[0].value)
            if not referenced.issubset(set(arg_names)):
                continue
            simple_fns[node.name] = (arg_names, node.body[0].value)

        if not simple_fns:
            return _unchanged(code=code, meta=meta)

        class _Substitute(ast.NodeTransformer):
            def __init__(self, mapping: Dict[str, ast.AST]) -> None:
                self.mapping = mapping

            def visit_Name(self, n: ast.Name) -> ast.AST:  # noqa: N802
                if isinstance(n.ctx, ast.Load) and n.id in self.mapping:
                    return copy.deepcopy(self.mapping[n.id])
                return n

        class _Inliner(ast.NodeTransformer):
            def __init__(self) -> None:
                self.changed = False

            def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
                node = self.generic_visit(node)
                if not isinstance(node.func, ast.Name):
                    return node
                fn = simple_fns.get(node.func.id)
                if fn is None:
                    return node
                arg_names, expr = fn
                if node.keywords or len(node.args) != len(arg_names):
                    return node
                mapping = {name: arg for name, arg in zip(arg_names, node.args, strict=True)}
                new_expr = _Substitute(mapping).visit(copy.deepcopy(expr))
                self.changed = True
                return ast.copy_location(new_expr, node)

        inliner = _Inliner()
        tree = inliner.visit(tree)
        ast.fix_missing_locations(tree)
        if not inliner.changed:
            return _unchanged(code=code, meta=meta)

        out = ast.unparse(tree)
        meta["inlined"] = sorted(simple_fns.keys())
        return _finalize_result(original=code, out=out, meta=meta)
    except Exception:
        return _unchanged(code=code, meta=meta)


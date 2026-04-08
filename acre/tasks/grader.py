from __future__ import annotations

import ast
import difflib
from typing import Tuple


def _normalize(code: str) -> Tuple[str, str]:
    """
    Deterministic normalization for grading.

    Returns:
      (ast_unparsed, stripped_source)
    """
    src = (code or "").replace("\r\n", "\n").strip()
    try:
        tree = ast.parse(src)
        normalized = ast.unparse(tree).strip()
        return normalized, src
    except Exception:
        return "", src


def grade_task(output: str, expected_output: str) -> float:
    """
    Deterministic score in [0.0, 1.0] comparing output vs expected_output.

    - If both parse as Python, we compare normalized AST-unparse strings.
    - Otherwise, we fall back to a whitespace-stripped diff similarity.
    """
    out_norm, out_src = _normalize(output)
    exp_norm, exp_src = _normalize(expected_output)

    if out_norm and exp_norm:
        if out_norm == exp_norm:
            return 1.0
        ratio = difflib.SequenceMatcher(a=exp_norm, b=out_norm).ratio()
        return float(max(0.0, min(1.0, ratio)))

    # Fallback: compare raw text (still deterministic).
    a = " ".join(exp_src.split())
    b = " ".join(out_src.split())
    if not a and not b:
        return 1.0
    ratio = difflib.SequenceMatcher(a=a, b=b).ratio()
    return float(max(0.0, min(1.0, ratio)))


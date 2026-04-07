from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Metric:
    """Single scalar metric value (placeholder)."""

    name: str
    value: float


@dataclass
class MetricLogger:
    """Tiny metric logger stub."""

    _history: Dict[str, List[float]] = field(default_factory=dict)

    def log(self, metric: Metric) -> None:
        self._history.setdefault(metric.name, []).append(metric.value)

    def latest(self) -> Dict[str, float]:
        return {k: v[-1] for k, v in self._history.items() if v}

    def as_series(self) -> Dict[str, Tuple[float, ...]]:
        return {k: tuple(v) for k, v in self._history.items()}

    def extend(self, metrics: Iterable[Metric]) -> None:
        for m in metrics:
            self.log(m)


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class CodeSample:
    """A single code sample (placeholder)."""

    id: str
    language: str
    code: str


class CodeSampleDataset:
    """
    Minimal in-memory dataset stub.

    Later versions can back this with files, Git repos, or benchmark suites.
    """

    def __init__(self, samples: Optional[Iterable[CodeSample]] = None) -> None:
        self._samples: List[CodeSample] = list(samples or [])

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[CodeSample]:
        return iter(self._samples)

    def add(self, sample: CodeSample) -> None:
        self._samples.append(sample)


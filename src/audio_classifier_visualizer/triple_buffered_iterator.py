from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, Optional, Tuple, TypeVar

T = TypeVar("T")


class TripleBufferedIterator(Iterator[Tuple[Optional[T], T, Optional[T]]]):
    def __init__(self, iterable_or_iterator: Iterable[T]) -> None:
        self.iter: Iterator[T] = iter(iterable_or_iterator)
        self.buffer: deque[T | None] = deque(maxlen=3)
        self.__iter__()

    def __iter__(self) -> TripleBufferedIterator[T]:
        self.iter = self.iter.__iter__()
        self.buffer.clear()
        self.buffer.append(None)
        return self

    def __next__(self) -> tuple[T | None, T, T | None]:
        while len(self.buffer) < 3:  # noqa: PLR2004
            try:
                self.buffer.append(next(self.iter))
            except StopIteration:
                break
        if len(self.buffer) < 2:  # noqa: PLR2004
            raise StopIteration
        prv = self.buffer.popleft()
        cur = self.buffer[0]
        nxt = self.buffer[1] if len(self.buffer) > 1 else None
        return prv, cur, nxt

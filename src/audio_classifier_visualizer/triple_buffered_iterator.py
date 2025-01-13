from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, Optional, Tuple, TypeVar

T = TypeVar("T")


class TripleBufferedIterator(Iterator[Tuple[Optional[T], T, Optional[T]]]):
    """Iterator that provides access to previous, current, and next items.

    This iterator maintains a sliding window of three items from the source iterator,
    allowing access to the previous, current, and next items at each step. This is
    useful for processing that requires context from adjacent items.
    """

    def __init__(self, iterable_or_iterator: Iterable[T]) -> None:
        """Initialize the triple buffered iterator.

        Args:
            iterable_or_iterator: The source iterator to wrap with triple buffering
        """
        self.iter: Iterator[T] = iter(iterable_or_iterator)
        self.buffer: deque[T | None] = deque(maxlen=3)
        self.__iter__()

    def __iter__(self) -> TripleBufferedIterator[T]:
        """Return self as iterator.

        Returns:
            Self as iterator object
        """
        self.iter = self.iter.__iter__()
        self.buffer.clear()
        self.buffer.append(None)
        return self

    def __next__(self) -> tuple[T | None, T, T | None]:
        """Get next item with its surrounding context.

        Returns:
            Tuple of (previous_item, current_item, next_item), where items may be None
            at the boundaries of the iteration.

        Raises:
            StopIteration: When the source iterator is exhausted
        """
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

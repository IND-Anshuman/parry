"""Metering for SDK streams without changing what the caller receives.

The OpenAI and Anthropic ``Stream`` / ``AsyncStream`` objects read from a
private ``_iterator`` generator. parry swaps in a generator that feeds each
event to an accumulator and reports when the stream ends, and wraps
``close()``. The caller gets the SDK's own object back, so ``isinstance``,
attributes and context-manager behaviour are exactly the SDK's. (Some SDK
versions define an ``isinstance`` hook that rejects anything but the exact
class, so a proxy could not pass.)

If a stream object has no ``_iterator``, parry falls back to a transparent
proxy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from ._adapters import StreamAccumulator

Finish = Callable[[StreamAccumulator], None]


class _Completion:
    __slots__ = ("acc", "finish", "done")

    def __init__(self, acc: StreamAccumulator, finish: Finish) -> None:
        self.acc = acc
        self.finish = finish
        self.done = False

    def __call__(self, completed: bool) -> None:
        if self.done:
            return
        self.done = True
        self.acc.complete = completed
        self.finish(self.acc)


def instrument_stream(stream: Any, acc: StreamAccumulator, finish: Finish) -> Any:
    source = getattr(stream, "_iterator", None)
    if source is None or not hasattr(source, "__next__"):
        return StreamWrapper(stream, acc, finish)
    complete = _Completion(acc, finish)

    def metered() -> Iterator[Any]:
        try:
            for item in source:
                acc.feed(item)
                yield item
        except BaseException:
            complete(False)
            raise
        complete(True)

    original_close = stream.close

    def close() -> None:
        try:
            original_close()
        finally:
            complete(False)

    stream._iterator = metered()
    stream.close = close
    return stream


def instrument_async_stream(stream: Any, acc: StreamAccumulator, finish: Finish) -> Any:
    source = getattr(stream, "_iterator", None)
    if source is None or not hasattr(source, "__anext__"):
        return AsyncStreamWrapper(stream, acc, finish)
    complete = _Completion(acc, finish)

    async def metered() -> AsyncIterator[Any]:
        try:
            async for item in source:
                acc.feed(item)
                yield item
        except BaseException:
            complete(False)
            raise
        complete(True)

    original_close = stream.close

    async def close() -> None:
        try:
            await original_close()
        finally:
            complete(False)

    stream._iterator = metered()
    stream.close = close
    return stream


# --- fallback proxies -------------------------------------------------------------


class _Proxy:
    __slots__ = ("_parry_stream", "_parry_acc", "_parry_complete")

    def __init__(self, stream: Any, acc: StreamAccumulator, finish: Finish) -> None:
        object.__setattr__(self, "_parry_stream", stream)
        object.__setattr__(self, "_parry_acc", acc)
        object.__setattr__(self, "_parry_complete", _Completion(acc, finish))

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # lets plain isinstance() checks see the SDK type
        return type(self._parry_stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parry_stream, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._parry_stream, name, value)

    def __repr__(self) -> str:
        return f"<parry stream over {self._parry_stream!r}>"


class StreamWrapper(_Proxy):
    __slots__ = ()

    def __iter__(self) -> StreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            item = next(self._parry_stream)
        except StopIteration:
            self._parry_complete(True)
            raise
        except BaseException:
            self._parry_complete(False)
            raise
        self._parry_acc.feed(item)
        return item

    def __enter__(self) -> StreamWrapper:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._parry_stream.close()
        finally:
            self._parry_complete(False)


class AsyncStreamWrapper(_Proxy):
    __slots__ = ()

    def __aiter__(self) -> AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            item = await self._parry_stream.__anext__()
        except StopAsyncIteration:
            self._parry_complete(True)
            raise
        except BaseException:
            self._parry_complete(False)
            raise
        self._parry_acc.feed(item)
        return item

    async def __aenter__(self) -> AsyncStreamWrapper:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        try:
            await self._parry_stream.close()
        finally:
            self._parry_complete(False)

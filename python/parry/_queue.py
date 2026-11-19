"""A background retry queue: run callables off-thread and retry transient
failures with backoff, without blocking the caller.

parry's guard retries a call *inline* — the caller waits through every backoff.
For fire-and-forget or batch work (enrichment, embeddings, offline jobs) you
often don't want to wait. ``RetryQueue`` runs each submitted callable on a pool
of worker threads and, on a retryable error, re-schedules it after a backoff
delay *without* holding a worker, using the same ``Retry`` classifier the guard
uses so retry behaviour matches. Each ``submit`` returns a standard
``concurrent.futures.Future``.

    q = parry.RetryQueue(workers=4)
    futures = [q.submit(enrich, row) for row in rows]
    q.close()                      # waits for everything (and its retries)
    results = [f.result() for f in futures]

If you submit a guarded SDK call, let the queue own the retries — build the
guard with ``Parry(retry=False)`` so the two layers don't both back off.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from typing import Any

from ._parry_native import Retry

_SHUTDOWN = object()


class _Job:
    __slots__ = ("fn", "args", "kwargs", "future", "attempt")

    def __init__(self, fn: Callable[..., Any], args: tuple, kwargs: dict, future: Future) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.attempt = 0


class RetryQueue:
    """A pool of worker threads that run callables and retry transient failures.

    ``workers`` sets the concurrency. ``max_attempts`` / ``base_delay`` /
    ``max_delay`` mirror :class:`parry.Retry`. ``max_pending`` bounds the queue
    (0 = unbounded); when full, ``submit`` blocks, giving backpressure. Use it as
    a context manager, or call :meth:`close` to drain and stop.
    """

    def __init__(
        self,
        workers: int = 4,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        max_pending: int = 0,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._retry = Retry(max_attempts=max_attempts, base_delay=base_delay, max_delay=max_delay)
        self._max_attempts = max_attempts
        self._q: queue.Queue[Any] = queue.Queue(maxsize=max_pending)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._outstanding = 0
        self._closed = False
        self._timers: set[threading.Timer] = set()
        self._counts = {"submitted": 0, "succeeded": 0, "failed": 0, "retried": 0}
        self._threads = [
            threading.Thread(target=self._work, name=f"parry-retryq-{i}", daemon=True)
            for i in range(workers)
        ]
        for t in self._threads:
            t.start()

    # --- submission --------------------------------------------------------

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        """Queue ``fn(*args, **kwargs)`` to run on a worker; returns its Future."""
        future: Future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("RetryQueue is closed")
            self._outstanding += 1
            self._counts["submitted"] += 1
        self._q.put(_Job(fn, args, kwargs, future))
        return future

    def map(self, fn: Callable[[Any], Any], iterable: Iterable[Any]) -> list[Future]:
        """Submit ``fn(item)`` for each item; returns the Futures in order."""
        return [self.submit(fn, item) for item in iterable]

    # --- workers -----------------------------------------------------------

    def _work(self) -> None:
        while True:
            job = self._q.get()
            try:
                if job is _SHUTDOWN:
                    return
                self._run(job)
            finally:
                self._q.task_done()

    def _run(self, job: _Job) -> None:
        # A Future goes PENDING -> RUNNING once; a retried job is already RUNNING,
        # so only transition (and honour a cancel) on the first attempt.
        if job.attempt == 0 and not job.future.set_running_or_notify_cancel():
            self._finalize()  # the future was cancelled before it ran
            return
        try:
            result = job.fn(*job.args, **job.kwargs)
        except BaseException as exc:  # noqa: BLE001 - deliberately route every error
            self._handle_error(job, exc)
            return
        try:
            job.future.set_result(result)
        finally:
            with self._lock:
                self._counts["succeeded"] += 1
            self._finalize()

    def _handle_error(self, job: _Job, exc: BaseException) -> None:
        try:
            job.attempt += 1
            retryable, retry_after = (
                self._retry.classify(exc) if isinstance(exc, Exception) else (False, None)
            )
            if retryable and job.attempt < self._max_attempts:
                delay = self._retry.delay(job.attempt, retry_after)
                with self._lock:
                    self._counts["retried"] += 1
                self._schedule(job, delay)
                return
        except BaseException:  # noqa: BLE001 - never let retry bookkeeping lose a job
            pass
        with self._lock:
            self._counts["failed"] += 1
        if not job.future.done():
            job.future.set_exception(exc)
        self._finalize()

    def _schedule(self, job: _Job, delay: float) -> None:
        # Re-enqueue after the backoff without tying up a worker during the wait.
        timer: threading.Timer

        def requeue() -> None:
            with self._lock:
                self._timers.discard(timer)
            self._q.put(job)

        timer = threading.Timer(max(delay, 0.0), requeue)
        timer.daemon = True
        with self._lock:
            self._timers.add(timer)
        timer.start()

    def _finalize(self) -> None:
        with self._lock:
            self._outstanding -= 1
            if self._outstanding == 0:
                self._idle.notify_all()

    # --- lifecycle ---------------------------------------------------------

    def join(self, timeout: float | None = None) -> bool:
        """Block until every submitted job has finished (retries included).
        Returns False on timeout."""
        with self._lock:
            return self._idle.wait_for(lambda: self._outstanding == 0, timeout)

    def close(self, *, wait: bool = True, timeout: float | None = None) -> None:
        """Stop accepting work and shut the workers down. With ``wait`` (default)
        it first drains outstanding jobs; otherwise pending retries are dropped."""
        with self._lock:
            self._closed = True
        if wait:
            self.join(timeout)
        else:
            with self._lock:
                timers = list(self._timers)
                self._timers.clear()
            for timer in timers:
                timer.cancel()
        for _ in self._threads:
            self._q.put(_SHUTDOWN)
        if wait:
            for t in self._threads:
                t.join(timeout)

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {**self._counts, "pending": self._outstanding}

    def __enter__(self) -> RetryQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close(wait=True)

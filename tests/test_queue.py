"""The background retry queue (parry.RetryQueue)."""

from __future__ import annotations

import threading

import pytest

from parry import RetryQueue


def test_runs_and_returns_results():
    with RetryQueue(workers=3) as q:
        futures = q.map(lambda n: n * n, range(5))
        q.join()
        assert sorted(f.result() for f in futures) == [0, 1, 4, 9, 16]
    assert q.stats["succeeded"] == 5 and q.stats["pending"] == 0


def test_retries_a_retryable_failure_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")  # retryable per the classifier
        return "ok"

    with RetryQueue(workers=1, max_attempts=5, base_delay=0.0, max_delay=0.0) as q:
        fut = q.submit(flaky)
        assert fut.result(timeout=5) == "ok"
    assert calls["n"] == 3
    assert q.stats["retried"] == 2 and q.stats["succeeded"] == 1 and q.stats["failed"] == 0


def test_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise ConnectionError("still down")

    with RetryQueue(workers=1, max_attempts=3, base_delay=0.0, max_delay=0.0) as q:
        fut = q.submit(always_fails)
        with pytest.raises(ConnectionError):
            fut.result(timeout=5)
    assert calls["n"] == 3  # one initial + two retries
    assert q.stats["failed"] == 1 and q.stats["retried"] == 2


def test_non_retryable_error_is_not_retried():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("permanent")  # not retryable

    with RetryQueue(workers=1, base_delay=0.0) as q:
        fut = q.submit(bad)
        with pytest.raises(ValueError):
            fut.result(timeout=5)
    assert calls["n"] == 1 and q.stats["retried"] == 0


def test_backoff_does_not_block_other_jobs():
    # A job that is waiting out its backoff must not occupy the single worker.
    started = threading.Event()

    def slow_retry():
        raise ConnectionError("retry me")

    def quick():
        started.set()
        return "done"

    with RetryQueue(workers=1, max_attempts=3, base_delay=0.2, max_delay=0.2) as q:
        q.submit(slow_retry)  # fails, then waits 0.2s off-worker before retrying
        fut = q.submit(quick)  # should run during that wait, not after all retries
        assert started.wait(1.0)
        assert fut.result(timeout=2) == "done"


def test_submit_after_close_raises():
    q = RetryQueue(workers=1)
    q.close()
    with pytest.raises(RuntimeError):
        q.submit(lambda: 1)


def test_close_waits_for_outstanding_work():
    done = []

    def work(n):
        done.append(n)
        return n

    q = RetryQueue(workers=2)
    for i in range(10):
        q.submit(work, i)
    q.close()  # wait=True by default
    assert len(done) == 10 and q.stats["pending"] == 0


def test_cancelled_future_is_not_run():
    gate = threading.Event()

    def blocker():
        gate.wait(2)
        return "x"

    with RetryQueue(workers=1) as q:
        q.submit(blocker)  # occupies the single worker
        fut = q.submit(lambda: "later")
        assert fut.cancel()  # cancel while still queued
        gate.set()
        q.join(timeout=5)
        assert fut.cancelled()
    assert q.stats["pending"] == 0


def test_validation():
    with pytest.raises(ValueError):
        RetryQueue(workers=0)
    with pytest.raises(ValueError):
        RetryQueue(max_attempts=0)

from __future__ import annotations

import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from threading import Lock
from time import perf_counter
from typing import Callable

from qrescue.qaoa import QAOAResult, qaoa_error_result, run_qaoa
from qrescue.qubo import DispatchProblem

_executor_lock = Lock()
_executor: ProcessPoolExecutor | None = None


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return _executor


def _discard_executor(executor: ProcessPoolExecutor) -> None:
    global _executor
    with _executor_lock:
        if _executor is executor:
            _executor = None
    executor.shutdown(wait=False, cancel_futures=True)


def shutdown() -> None:
    global _executor
    with _executor_lock:
        executor = _executor
        _executor = None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


atexit.register(shutdown)


def run_qaoa_isolated(
    problem: DispatchProblem,
    *,
    target: Callable[..., QAOAResult] = run_qaoa,
    **kwargs,
) -> QAOAResult:
    started = perf_counter()
    executor = _get_executor()
    try:
        return executor.submit(target, problem, **kwargs).result()
    except BrokenProcessPool:
        _discard_executor(executor)
        return qaoa_error_result(
            problem,
            reps=kwargs.get("reps", 2),
            shots=kwargs.get("shots", 4096),
            seed=kwargs.get("seed", 7),
            message=(
                "QAOA worker process crashed (native fault in Qiskit/Aer simulator); "
                "no QAOA result was produced."
            ),
            runtime_seconds=perf_counter() - started,
        )
    except Exception as error:
        return qaoa_error_result(
            problem,
            reps=kwargs.get("reps", 2),
            shots=kwargs.get("shots", 4096),
            seed=kwargs.get("seed", 7),
            message=f"{type(error).__name__}: {error}",
            runtime_seconds=perf_counter() - started,
        )

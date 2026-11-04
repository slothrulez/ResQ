from __future__ import annotations

import ctypes
import resource

from resq.qaoa import QAOAResult
from resq.qubo import DispatchProblem


def crash_worker(problem: DispatchProblem, **kwargs) -> QAOAResult:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ctypes.string_at(0)
    raise AssertionError("unreachable")

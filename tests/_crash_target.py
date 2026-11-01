from __future__ import annotations

import ctypes
import resource

from qrescue.qaoa import QAOAResult
from qrescue.qubo import DispatchProblem


def crash_worker(problem: DispatchProblem, **kwargs) -> QAOAResult:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ctypes.string_at(0)
    raise AssertionError("unreachable")
